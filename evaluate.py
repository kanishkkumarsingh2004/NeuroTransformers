import os
import torch
from model import MiniLanguageModel, HYPERPARAMITER
from data import tokenizer, vocab_size, train_data

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "transformer.pt")


def format_int(value):
    return f"{value:,}"


def format_bytes(value):
    return f"{value / 1024**2:.2f} MB"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint():
    model_path = HYPERPARAMITER.model_path if os.path.exists(HYPERPARAMITER.model_path) else os.path.join(HYPERPARAMITER.model_dir, "transformer.pt")
    if not os.path.exists(model_path):
        print(f"Warning: checkpoint not found at {model_path}. Using default initialized model weights.")
        model = MiniLanguageModel(vocab_size=vocab_size).to(HYPERPARAMITER.device)
        return model, False

    checkpoint = torch.load(model_path, map_location=HYPERPARAMITER.device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

    n_embd = config.get("n_embd")
    n_head = config.get("n_head")
    n_layer = config.get("n_layer")

    if n_embd is None and "token_embedding_table.weight" in model_state:
        n_embd = model_state["token_embedding_table.weight"].shape[1]
    if n_layer is None and any(k.startswith("blocks.") for k in model_state):
        n_layer = max(int(k.split('.')[1]) for k in model_state if k.startswith("blocks.")) + 1

    model = MiniLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer
    ).to(HYPERPARAMITER.device)

    saved_vocab_size = model_state["token_embedding_table.weight"].shape[0]
    if saved_vocab_size != vocab_size:
        print(f"-> Saved vocab size ({saved_vocab_size}) differs from current ({vocab_size}). Resizing embeddings...")
        model.resize_token_embeddings(saved_vocab_size)
        model.load_state_dict(model_state)
        model.resize_token_embeddings(vocab_size)
    else:
        model.load_state_dict(model_state)
    return model, True


def main():
    print("=== Model Evaluation ===")
    print(f"Device: {HYPERPARAMITER.device}")
    print(f"Vocab size: {format_int(vocab_size)}")
    print(f"Block size: {HYPERPARAMITER.block_size}")
    print(f"Embedding dimension: {HYPERPARAMITER.n_embd}")
    print(f"Number of attention heads: {HYPERPARAMITER.n_head}")
    print(f"Number of transformer layers: {HYPERPARAMITER.n_layer}")
    print(f"Dropout: {HYPERPARAMITER.dropout}")
    print(f"Batch size: {HYPERPARAMITER.batch_size}")
    print(f"Learning rate: {HYPERPARAMITER.learning_rate}")
    steps_per_epoch = max(1, len(train_data) // HYPERPARAMITER.batch_size)
    print(f"Epochs: {HYPERPARAMITER.epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Data folder: {HYPERPARAMITER.data_dir}")
    data_files = sorted([f for f in os.listdir(HYPERPARAMITER.data_dir) if f.endswith('.txt')])
    print(f"Data files: {', '.join(data_files) if data_files else 'none found'}")
    print(f"Model path: {HYPERPARAMITER.model_path}")
    print(f"Max training iterations: {HYPERPARAMITER.max_iters}")
    print(f"Evaluation interval: {HYPERPARAMITER.eval_interval}")
    print(f"Evaluation iterations per split: {HYPERPARAMITER.eval_iters}\n")

    model = MiniLanguageModel(vocab_size=vocab_size).to(HYPERPARAMITER.device)
    loaded = load_checkpoint(model)
    total_params, trainable_params = count_parameters(model)
    non_trainable_params = total_params - trainable_params
    estimated_size = total_params * 4

    print("=== Parameter Summary ===")
    print(f"Total parameters: {format_int(total_params)}")
    print(f"Trainable parameters: {format_int(trainable_params)}")
    print(f"Non-trainable parameters: {format_int(non_trainable_params)}")
    print(f"Estimated model size (float32): {format_bytes(estimated_size)}")
    print(f"Estimated model size (float16): {format_bytes(estimated_size / 2)}")

    embedding_params = sum(p.numel() for p in model.token_embedding_table.parameters())
    position_params = sum(p.numel() for p in model.position_embedding_table.parameters())
    head_params = sum(p.numel() for p in model.blocks.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())

    print("\n=== Key Module Parameter Counts ===")
    print(f"Token embedding parameters: {format_int(embedding_params)}")
    print(f"Position embedding parameters: {format_int(position_params)}")
    print(f"Transformer block parameters: {format_int(head_params)}")
    print(f"Language model head parameters: {format_int(lm_head_params)}")

    if loaded:
        checkpoint_size = os.path.getsize(MODEL_PATH)
        print(f"\nCheckpoint path: {MODEL_PATH}")
        print(f"Checkpoint file size: {format_bytes(checkpoint_size)}")

    with torch.no_grad():
        sample_input = torch.zeros((1, min(HYPERPARAMITER.block_size, 8)), dtype=torch.long, device=HYPERPARAMITER.device)
        logits, loss = model(sample_input)
        print(f"\nSample forward output shape: {tuple(logits.shape)}")
        print(f"Sample forward loss (untrained/loaded): {loss.item() if loss is not None else 'None'}")

    print("\nTokenizer special tokens:")
    print("  [BOS]", tokenizer.stoi.get("[BOS]"))
    print("  [EOS]", tokenizer.stoi.get("[EOS]"))
    print("  [USER]", tokenizer.stoi.get("[USER]"))
    print("  [ASSISTANT]", tokenizer.stoi.get("[ASSISTANT]"))


if __name__ == "__main__":
    main()
