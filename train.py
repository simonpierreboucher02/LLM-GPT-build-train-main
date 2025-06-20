import torch
from matplotlib import pyplot as plt

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    # Training loop implementation

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    plt.figure()
    plt.plot(epochs_seen, train_losses, label="Training Loss")
    plt.plot(epochs_seen, val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
