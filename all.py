import math
import time
from functools import partial
import os
import threading
import torch  # Assurez-vous d'avoir torch installé
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import argparse
import pickle  # Pour sauvegarder et charger le vocabulaire

# Import the renamed data loader
def load_dataset_custom(dataname):
    from datasets import load_dataset
    import numpy as np

    if dataname == "enwik8":
        dataset = load_dataset("enwik8")
    elif dataname == "ptb":
        dataset = load_dataset("ptb_text_only")
    elif dataname == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=['train', 'validation', 'test'])
    else:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=['train', 'validation', 'test'])
    
    vocab = set()
    for split in dataset:
        for text in split['text']:
            vocab.update(text.split())

    vocab = {v: i for i, v in enumerate(vocab)}
    
    def tokenize(text):
        return [vocab[word] for word in text.split() if word in vocab]

    train = np.concatenate([tokenize(text) for text in dataset[0]['text']]).astype(np.int32)
    valid = np.concatenate([tokenize(text) for text in dataset[1]['text']]).astype(np.int32)
    test = np.concatenate([tokenize(text) for text in dataset[2]['text']]).astype(np.int32)

    return vocab, train, valid, test

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layers, dims, num_heads, checkpoint):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.Parameter(torch.zeros(1, 5000, dims))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dims, num_heads),
            num_layers
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def forward(self, x):
        L = x.shape[1]
        x = self.embedding(x) + self.pe[:, :L]
        x = self.transformer(x)
        return self.out_proj(x)

def to_samples(context_size, dataset):
    tokens = len(dataset)
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    ).astype(np.int32)
    return X[:, :-1], X[:, 1:]

def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0

def save_checkpoint(model, optimizer, epoch, loss, vocab, filepath='model_checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'vocab': vocab  # Sauvegarder le vocabulaire dans le checkpoint
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = TransformerLM(vocab_size=checkpoint['model_state_dict']['embedding.weight'].size(0),
                          num_layers=3, dims=256, num_heads=4, checkpoint=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    vocab = checkpoint['vocab']  # Charger le vocabulaire du checkpoint
    return model, optimizer, epoch, loss, vocab

def generate_text(model, vocab, seed_text, max_length=100):
    model.eval()
    
    # Inverse vocab to decode tokens back to words
    inv_vocab = {v: k for k, v in vocab.items()}
    
    def tokenize(text):
        return [vocab[word] for word in text.split() if word in vocab]
    
    def detokenize(tokens):
        return ' '.join([inv_vocab[token] for token in tokens])
    
    input_ids = torch.tensor([tokenize(seed_text)], dtype=torch.long).to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token_id), dim=1)
    
    generated_text = detokenize(generated[0].cpu().numpy())
    return generated_text

def main(args):
    batch_size = args.batch_size
    context_size = args.context_size
    steps_per_eval = args.steps_per_eval
    steps_per_report = args.steps_per_report
    num_steps = args.num_steps

    # Load vocab and dataset:
    vocab, train, valid, test = load_dataset_custom(args.dataset)

    # Initialize model:
    model = TransformerLM(
        len(vocab), args.num_blocks, args.dim, args.num_heads, args.checkpoint
    )
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    model.to(device)
    nparams = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    def loss_fn(model, x, y):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def eval_fn(dataset):
        model.eval()
        inputs, targets = map(torch.tensor, to_samples(context_size, dataset))
        inputs, targets = inputs.to(device), targets.to(device)
        loss = 0
        with torch.no_grad():
            for s in range(0, targets.shape[0], batch_size):
                bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
                logits = model(bx)
                loss += F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1)).item()
        model.train()
        return loss / len(targets)

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []
    tic = time.perf_counter()
    for it, (inputs, targets) in zip(range(num_steps), train_iterator):
        inputs, targets = map(torch.tensor, (inputs, targets))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, inputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {steps_per_report / (toc - tic):.3f}"
            )
            losses = []
            tic = time.perf_counter()
        if (it + 1) % steps_per_eval == 0:
            val_loss = eval_fn(valid)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
                f"Val took {(toc - tic):.3f}s, "
            )
            tic = time.perf_counter()

        # Save the model every 1000 steps
        if (it + 1) % 1000 == 0:
            checkpoint_path = f'{args.save_path}_step_{it + 1}.pth'
            save_checkpoint(model, optimizer, it + 1, train_loss, vocab, checkpoint_path)
            print(f'Checkpoint saved at step {it + 1} to {checkpoint_path}')

    if args.eval_test:
        test_loss = eval_fn(test)
        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    # Save the final model and optimizer state at the end of training
    save_checkpoint(model, optimizer, num_steps, train_loss, vocab, args.save_path)

# Arguments setup
args = argparse.Namespace(
    gpu=True,  # Use GPU if available
    seed=42,
    dataset='wikitext2',
    context_size=512,
    num_blocks=3,
    dim=256,
    num_heads=4,
    checkpoint=False,
    batch_size=2,
    num_iters=100000,
    learning_rate=3e-4,
    weight_decay=1e-5,
    lr_warmup=200,
    steps_per_report=10,
    steps_per_eval=1000,
    eval_test=True,
    num_steps=10000,  # Total number of training steps
    save_path='model_checkpoint.pth'  # Path to save the model checkpoint
)

# Execute the training
main(args)

# Load the model and optimizer from the checkpoint
model_checkpoint_path = args.save_path
model, optimizer, epoch, loss, vocab = load_checkpoint(model_checkpoint_path)

# Use the GPU if available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model.to(device)

# Generate text
seed_text = "Once upon a time"
generated_text = generate_text(model, vocab, seed_text, max_length=100)
print(generated_text)
