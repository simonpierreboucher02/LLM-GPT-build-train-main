# GPT Training Implementation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/graphs/commit-activity)
[![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/network)
[![GitHub issues](https://img.shields.io/github/issues/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/pulls)

A PyTorch implementation of GPT (Generative Pre-trained Transformer) model training from scratch. This project provides a complete pipeline for building, training, and evaluating GPT models with customizable configurations.

## 🚀 Features

[![Lines of Code](https://img.shields.io/tokei/lines/github/simonpierreboucher02/LLM-GPT-build-train-main)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main)
[![Code Size](https://img.shields.io/github/languages/code-size/simonpierreboucher02/LLM-GPT-build-train-main)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main)
[![Repo Size](https://img.shields.io/github/repo-size/simonpierreboucher02/LLM-GPT-build-train-main)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main)
[![Last Commit](https://img.shields.io/github/last-commit/simonpierreboucher02/LLM-GPT-build-train-main)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/commits/main)

- **Complete GPT Implementation**: Full implementation of the GPT architecture including:
  - Multi-head self-attention mechanism
  - Feed-forward networks with GELU activation
  - Layer normalization
  - Positional embeddings
  - Token embeddings

- **Flexible Configuration**: Easy-to-modify model configurations for different model sizes
- **Training Pipeline**: Complete training loop with validation and loss plotting
- **Data Loading**: Customizable data loader for text processing
- **GPU Support**: Automatic CUDA detection and utilization

## 📋 Requirements

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-orange.svg)](https://matplotlib.org/)
[![tqdm](https://img.shields.io/badge/tqdm-4.60+-lightgrey.svg)](https://tqdm.github.io/)
[![tiktoken](https://img.shields.io/badge/tiktoken-0.3+-purple.svg)](https://github.com/openai/tiktoken)

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- tqdm
- tiktoken (for tokenization)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/simonpierreboucher02/LLM-GPT-build-train-main.git
cd LLM-GPT-build-train-main
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib tqdm tiktoken
```

## 🎯 Usage

### Basic Training

Run the main training script:

```bash
python main.py
```

### Custom Configuration

You can modify the model configuration in `main.py`:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 256,    # Sequence length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Use bias in QKV projections
}

SETTINGS = {
    "learning_rate": 5e-4,   # Learning rate
    "num_epochs": 10,        # Number of training epochs
    "batch_size": 2,         # Batch size
    "weight_decay": 0.1      # Weight decay for regularization
}
```

## 📁 Project Structure

```
LLM-GPT-build-train-main/
├── main.py          # Main training script
├── model.py         # GPT model implementation
├── train.py         # Training utilities and loss plotting
├── data_loader.py   # Data loading and preprocessing
├── setup.py         # Setup utilities and file downloads
├── all.py           # Additional utilities
└── README.md        # This file
```

## 🔧 Model Architecture

The implementation includes:

- **LayerNorm**: Custom layer normalization implementation
- **MultiHeadAttention**: Multi-head self-attention mechanism
- **FeedForward**: Position-wise feed-forward networks
- **GPTModel**: Complete GPT model with token embeddings and transformer blocks

## 📊 Training

The training process includes:

- Automatic device detection (CPU/GPU)
- Training and validation loss tracking
- Progress monitoring
- Loss visualization with matplotlib

## 📈 Repository Metrics

[![GitHub contributors](https://img.shields.io/github/contributors/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/graphs/contributors)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/graphs/commit-activity)
[![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/m/simonpierreboucher02/LLM-GPT-build-train-main/main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main/graphs/commit-activity)
[![GitHub language count](https://img.shields.io/github/languages/count/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main)
[![GitHub top language](https://img.shields.io/github/languages/top/simonpierreboucher02/LLM-GPT-build-train-main.svg)](https://github.com/simonpierreboucher02/LLM-GPT-build-train-main)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Simon Pierre Boucher**
- GitHub: [@simonpierreboucher02](https://github.com/simonpierreboucher02)

## 🙏 Acknowledgments

This implementation is inspired by the original GPT paper and various open-source implementations in the community.

---

⭐ If you find this project helpful, please give it a star! 