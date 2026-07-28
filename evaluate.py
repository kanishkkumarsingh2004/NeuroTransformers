import os
import torch
from model import ModernLLM, ModelConfig, HYPERPARAMITER
from data import tokenizer, get_batch, estimate_loss

# -----------------------------
# Configuration & Device
# -----------------------------
DEVICE = getattr(HYPERPARAMITER, "device", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = getattr(HYPERPARAMITER, "model_dir", os.path.join(os.path.dirname(__file__), "model"))
MODEL_PATH = getattr(HYPERPARAMITER, "model_path", os.path.join(MODEL_DIR, "transformer.pt"))
VOCAB_PATH = getattr(HYPERPARAMITER, "vocab_path", os.path.join(MODEL_DIR, "vocab.json"))
MERGES_PATH = getattr(HYPERPARAMITER, "merges_path", os.path.join(MODEL_DIR, "merges.txt"))


def format_int(value):
    return f"{value:,}"


def format_bytes(value):
    return f"{value / 1024**2:.2f} MB"


def format_params(value):
    """Formats parameter count into Millions (M), Billions (B), Trillions (T), and Crores (Cr)."""
    val = float(value)
    formatted = f"{value:,}"
    
    parts = []
    if val >= 1e12:
        parts.append(f"{val / 1e12:.3f} Trillion (T)")
    elif val >= 1e9:
        parts.append(f"{val / 1e9:.3f} Billion (B)")
    elif val >= 1e6:
        parts.append(f"{val / 1e6:.2f} Million (M)")
    
    if val >= 1e7:
        parts.append(f"{val / 1e7:.2f} Crore (Cr)")

    if parts:
        return f"{formatted}  -->  [{' | '.join(parts)}]"
    return formatted


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_tokenizer_without_retraining():
    """Loads existing tokenizer files without retraining or dataset rebuilds."""
    if os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH):
        try:
            tokenizer.load(VOCAB_PATH, MERGES_PATH)
            print(f"✅ Tokenizer loaded successfully from disk (Vocab size: {len(tokenizer.vocab)}).")
        except Exception as e:
            print(f"⚠️ Error loading tokenizer files from {VOCAB_PATH}: {e}")
    else:
        print(f"⚠️ Tokenizer files not found at {VOCAB_PATH}. Using default initialized tokenizer in memory.")


def load_checkpoint():
    """Loads model weights and config from the saved checkpoint."""
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Warning: Checkpoint not found at '{MODEL_PATH}'. Using default initialized model weights.")
        config = ModelConfig(
            vocab_size=len(tokenizer.vocab),
            dim=getattr(HYPERPARAMITER, "dim", 512),
            n_layers=getattr(HYPERPARAMITER, "n_layers", getattr(HYPERPARAMITER, "n_layer", 8)),
            n_heads=getattr(HYPERPARAMITER, "n_heads", getattr(HYPERPARAMITER, "n_head", 8)),
            max_seq_len=getattr(HYPERPARAMITER, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256)),
        )
        model = ModernLLM(config).to(DEVICE)
        return model, config, False

    print(f"📂 Loading checkpoint from '{MODEL_PATH}'...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
        saved_config = checkpoint.get("config", None)
    else:
        model_state = checkpoint
        saved_config = None

    if isinstance(saved_config, ModelConfig):
        config = saved_config
    elif isinstance(saved_config, dict) and saved_config:
        config = ModelConfig(**saved_config)
    else:
        v_size = len(tokenizer.vocab)
        if "tok_embeddings.weight" in model_state:
            v_size = model_state["tok_embeddings.weight"].shape[0]
            emb_dim = model_state["tok_embeddings.weight"].shape[1]
        elif "token_embedding_table.weight" in model_state:
            v_size = model_state["token_embedding_table.weight"].shape[0]
            emb_dim = model_state["token_embedding_table.weight"].shape[1]
        else:
            emb_dim = getattr(HYPERPARAMITER, "dim", getattr(HYPERPARAMITER, "n_embd", 512))

        config = ModelConfig(
            vocab_size=v_size,
            dim=emb_dim,
            n_layers=getattr(HYPERPARAMITER, "n_layers", getattr(HYPERPARAMITER, "n_layer", 8)),
            n_heads=getattr(HYPERPARAMITER, "n_heads", getattr(HYPERPARAMITER, "n_head", 8)),
            max_seq_len=getattr(HYPERPARAMITER, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256)),
        )

    model = ModernLLM(config).to(DEVICE)

    try:
        model.load_state_dict(model_state, strict=False)
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning loading state dict: {e}")

    return model, config, True


def main():
    print("==========================================")
    print("         LLM MODEL EVALUATION             ")
    print("==========================================")

    # 1. Load Tokenizer strictly from existing files
    load_tokenizer_without_retraining()
    current_vocab_size = len(tokenizer.vocab)

    # 2. Load Model Checkpoint
    model, config, loaded = load_checkpoint()
    model.eval()

    # 3. Print Hyperparameters & Configuration
    block_size = getattr(config, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256))
    batch_size = getattr(HYPERPARAMITER, "batch_size", 64)

    print("\n=== Architectural Configuration ===")
    print(f"Device:                      {DEVICE}")
    print(f"Vocabulary Size:             {format_int(current_vocab_size)}")
    print(f"Max Sequence Length (Block): {block_size}")
    print(f"Embedding Dimension (dim):   {config.dim}")
    print(f"Attention Heads:             {config.n_heads}")
    print(f"KV Attention Heads (GQA):    {config.n_kv_heads}")
    print(f"Transformer Layers:          {config.n_layers}")
    print(f"Hidden Dimension (SwiGLU):   {config.hidden_dim}")
    print(f"Batch Size:                  {batch_size}")
    print(f"Model Path:                  {MODEL_PATH}")

    # 4. Parameter Counts
    total_params, trainable_params = count_parameters(model)
    non_trainable_params = total_params - trainable_params
    estimated_size = total_params * 4  # float32

    print("\n=== Parameter Summary ===")
    print(f"Total Parameters:            {format_params(total_params)}")
    print(f"Trainable Parameters:        {format_params(trainable_params)}")
    print(f"Non-Trainable Parameters:    {format_params(non_trainable_params)}")
    print(f"Estimated VRAM (FP32):       {format_bytes(estimated_size)}")
    print(f"Estimated VRAM (FP16/BF16):  {format_bytes(estimated_size / 2)}")

    # Module Parameter Counts for ModernLLM
    print("\n=== Sub-Module Breakdown ===")
    if hasattr(model, "tok_embeddings"):
        tok_params = sum(p.numel() for p in model.tok_embeddings.parameters())
        layer_params = sum(p.numel() for p in model.layers.parameters())
        out_params = sum(p.numel() for p in model.output.parameters())
        print(f"Token Embeddings:            {format_params(tok_params)}")
        print(f"Transformer Stack ({config.n_layers} blocks): {format_params(layer_params)}")
        print(f"LM Head Output Layer:        {format_params(out_params)}")

    if loaded and os.path.exists(MODEL_PATH):
        checkpoint_size = os.path.getsize(MODEL_PATH)
        print(f"\nCheckpoint Size on Disk:     {format_bytes(checkpoint_size)}")

    # 5. Model Evaluation (Loss & Sample Forward Pass)
    print("\n=== Model Loss Evaluation ===")
    try:
        losses = estimate_loss(model)
        print(f"Train Loss:                  {losses['train']:.4f}")
        print(f"Validation Loss:             {losses['val']:.4f}")
    except Exception as e:
        print(f"⚠️ Could not compute dataset loss: {e}")

    # Sample Forward Pass Test
    with torch.no_grad():
        sample_input = torch.zeros((1, min(block_size, 8)), dtype=torch.long, device=DEVICE)
        logits, loss, _ = model(sample_input)
        print(f"Sample Forward Logits Shape: {tuple(logits.shape)}")

    # 6. Special Tokens Inspection
    print("\n=== Tokenizer Special Tokens ===")
    for token in ["[PAD]", "[BOS]", "[EOS]", "[SYSTEM]", "[USER]", "[ASSISTANT]", "[THOUGHT]", "<|im_start|>", "<|im_end|>"]:
        token_id = tokenizer.stoi.get(token, "Not Defined")
        print(f"  {token:<15} -> ID: {token_id}")

    print("\n✅ Evaluation Completed Successfully.")


if __name__ == "__main__":
    main()