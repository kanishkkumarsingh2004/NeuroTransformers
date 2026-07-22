# NeuroTransformers: Decoder-Only Transformer Chatbot with Byte Pair Encoding (BPE)

A PyTorch-based decoder-only transformer chatbot trained with custom Byte Pair Encoding (BPE) tokenization. This repository implements an end-to-end small language model pipeline for dialogue training and interactive inference.

---

## 🌟 Features

- **Decoder-Only Transformer Architecture**: Features causal self-attention, position-wise feed-forward networks, pre-layer normalization (Pre-LN), and residual connections.
- **Custom Byte Pair Encoding (BPE)**: Implements a subword-level BPE tokenizer trained from scratch directly on the dataset characters.
- **Merge Persistence**: Automatically learns, saves, and loads merge rules to/from `model/merges.txt` and vocabs to `model/vocab.json`.
- **Dialogue Masking & Autoregressive Training**: Formats dialogues with special structure tokens (`[BOS]`, `[USER]`, `[ASSISTANT]`, `[EOS]`) and applies target masking (`-100` label masks) to train the model strictly on predicting the assistant's responses.
- **Interactive Chat Interface**: Interactive chat prompt to test the model's generation capacity in real-time.
- **GPU Acceleration**: Built-in CUDA support with automatic mixed-precision training.

---

## 📂 Project Structure

```text
├── data.py              # BPETokenizer class, vocabulary building, and dataloader
├── model.py             # Decoder-only transformer architecture and hyperparameters
├── train.py             # Training pipeline with checkpoint resuming and mixed precision
├── test_chat.py         # Interactive dialogue chatbot loop
├── test_next.py         # Next token/word prediction interface
├── requirements.txt     # Python dependencies
├── data1/               # Training data folder
│   └── main.txt         # Conversation pairs separated by '|||'
└── model/               # Model folder (automatically created)
    ├── vocab.json       # Vocabulary token-to-ID mapping
    ├── merges.txt       # BPE merge rules (id1 id2 merged_id)
    ├── config.json      # Trained model configurations
    └── transformer.pt   # Saved PyTorch checkpoint weights
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU-accelerated training)

### Setup
1. Clone the repository and navigate into the workspace.
2. Initialize and activate your virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Workflow & Usage

### 1. Prepare Training Data
Place conversation pairs in `data1/main.txt`. Each conversation line must use `|||` as a divider:
```text
hii bro how are u ||| I'm doing well, thanks! How about you?
what is the capital of france ||| The capital of France is Paris.
how to cook rice ||| Rinse, add water, boil, then simmer until absorbed.
```

### 2. Train the Model
Run the training pipeline:
```bash
python train.py
```

#### What happens under the hood during training:
1. **BPE Training**: `data.py` reads `data1/main.txt`, collects unique characters, and runs the subword merge training up to a target size of `500` tokens.
2. **Persistence**: Saves BPE rules in `model/merges.txt`, vocab maps in `model/vocab.json`, and configurations in `model/config.json`.
3. **Dialogue Sequence Assembly**: Encodes dialogues into unified tensor sequences formatted as:
   `[BOS][USER]{prompt_text}[ASSISTANT]{response_text}[EOS]`
4. **Target Masking**: Targets for the `[BOS][USER]{prompt_text}` segments are masked to `-100` so that cross-entropy loss is computed exclusively on the assistant response.
5. **Autoregressive Optimization**: Model trains using the AdamW optimizer with AMP (Automatic Mixed Precision) and saves checkpoint weights to `model/transformer.pt`.

### 3. Run Inference

#### Chat Interface
Launch the interactive chat loop:
```bash
python test_chat.py
```
This formats your query inside dialogue structure templates for conversational generation.

#### Next Token Predictor
Launch the next token/word prediction loop:
```bash
python test_next.py
```
This predicts the next single token and short continuation based on raw text inputs.

---

## ⚙️ Hyperparameters

Edit configuration values directly in [model.py](file:///home/kanishk/Desktop/kk-code/NeuroTransformers/model.py):
- `block_size = 256`: Context window length.
- `n_embd = 384`: Dense vector embedding size.
- `n_head = 6`: Number of causal self-attention heads (each head size `64`).
- `n_layer = 6`: Transformer blocks stacked.
- `dropout = 0.2`: Dropout probability.
- `learning_rate = 3e-4`: Optimization step scale.
