import torch
from data_loader import create_dataloader_v1
from model import GPTModel
from train import train_model_simple, plot_losses

if __name__ == "__main__":
    # Model and training settings
    GPT_CONFIG_124M = {
        "vocab_size": 50257, "context_length": 256, "emb_dim": 768,
        "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False
    }
    SETTINGS = {
        "learning_rate": 5e-4, "num_epochs": 10, "batch_size": 2, "weight_decay": 0.1
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data, Model, Optimizer setup
    text_data = "Sample text for model"  # Load or prepare data as required
    train_loader = create_dataloader_v1(text_data[:int(len(text_data) * 0.9)], SETTINGS["batch_size"])
    val_loader = create_dataloader_v1(text_data[int(len(text_data) * 0.9):], SETTINGS["batch_size"], shuffle=False)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])

    # Train model and plot losses
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device,
                                                               SETTINGS["num_epochs"], eval_freq=5, eval_iter=10)
    plot_losses(range(SETTINGS["num_epochs"]), tokens_seen, train_losses, val_losses)
