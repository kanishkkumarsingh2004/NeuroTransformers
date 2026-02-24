# Transformer Chatbot

A Transformer-based chatbot implementation using PyTorch for natural language understanding and generation.

## Features

- **Transformer Architecture**: Implements the Transformer model from "Attention Is All You Need" paper
- **Multi-Head Attention**: Multi-head self-attention and cross-attention mechanisms
- **Positional Encoding**: Sinusoidal positional encoding for sequence modeling
- **Customizable Hyperparameters**: Configurable model size, layers, heads, and dimensions
- **Data Loading**: Supports text files with parallel data pairs
- **Training Pipeline**: Includes training loop with checkpointing
- **Interactive Testing**: Real-time chat interface for testing the model
- **GPU Acceleration**: CUDA support for faster training

## Project Structure

```
├── data.py              # Data loading and preprocessing
├── model.py             # Transformer model implementation
├── train.py             # Training pipeline
├── test.py              # Interactive testing interface
├── requirements.txt     # Python dependencies
├── data1/               # Sample training data
│   └── main.txt         # Example conversation pairs
└── model/               # Saved model files (generated)
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Training Data

Create or update your training data in the `data1` folder. The data should be in the format:

```text
input text ||| response text
another input ||| another response
```

**Example data file** (data1/main.txt):
```text
hii bro how are u ||| I'm doing well, thanks!
how's the weather today ||| It's sunny and warm.
what are you doing ||| I'm learning about transformers.
```

### 2. Train the Model

```bash
python train.py
```

Training parameters can be modified in [train.py](train.py:13-17):
- `LEARNING_RATE`: 1e-4
- `EPOCHS`: 5
- `BATCH_SIZE`: 2
- `MAX_LEN`: 20 (in data.py)

### 3. Test the Model

```bash
python test.py
```

This will start an interactive chat interface. Type your messages and the model will respond. Type "quit" or "exit" to end the session.

## Model Architecture

The transformer consists of:

### Encoder
- 32 layers (configurable)
- Multi-head attention (16 heads)
- Position-wise feed-forward networks
- Layer normalization and dropout

### Decoder
- 32 layers (configurable)
- Self-attention mechanism
- Cross-attention with encoder outputs
- Position-wise feed-forward networks
- Layer normalization and dropout

### Hyperparameters

```python
d_model = 512          # Model dimension
num_heads = 16         # Number of attention heads
num_layers = 32        # Number of encoder/decoder layers
d_ff = 2048            # Feed-forward dimension
max_seq_length = 200   # Maximum sequence length
dropout = 0.2          # Dropout probability
```

## Data Processing

The [data.py](data.py) module handles:
- Loading training data from text files
- Tokenization using NLTK
- Vocabulary building
- Encoding/decoding with special tokens (<pad>, <sos>, <eos>, <unk>)
- Padding sequences to fixed length (MAX_LEN)

## Training Details

The training process includes:
- Batch training with CUDA acceleration
- Gradient clipping
- Checkpointing (saves model after each epoch)
- Learning rate optimization with Adam optimizer
- Loss calculation using cross-entropy with padding mask

## Performance

Training time depends on:
- Number of epochs
- Batch size
- GPU memory (CUDA required for efficient training)
- Model size (d_model, num_layers, num_heads)

## Customization

### Modify Model Parameters

Edit the hyperparameters in [model.py](model.py:157-164).

### Change Data Source

Modify the data loading logic in [data.py](data.py:78-124) to support:
- PDF files
- Different data formats
- Additional data sources

### Adjust Training Settings

Update the training parameters in [train.py](train.py:13-17).

## Requirements

```
click==8.2.1
filelock==3.13.1
fsspec==2024.6.1
Jinja2==3.1.4
joblib==1.5.2
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.3
nltk==3.9.1
numpy==2.1.2
PyPDF2==3.0.1
regex==2025.9.1
setuptools==70.2.0
sympy==1.13.3
torch==2.7.1+cu118
torchaudio==2.7.1+cu118
torchsummary==1.5.1
torchvision==0.22.1+cu118
tqdm==4.67.1
triton==3.3.1
typing_extensions==4.12.2
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This implementation is based on the original Transformer paper "Attention Is All You Need"
- PyTorch library for deep learning framework
- NLTK for natural language processing
