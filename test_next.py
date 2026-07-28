import os
import torch
from model import ModernLLM, ModelConfig, HYPERPARAMITER
from data import tokenizer, vocab_size

# -----------------------------
# Device & Paths Configuration
# -----------------------------
DEVICE = getattr(HYPERPARAMITER, "device", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = getattr(HYPERPARAMITER, "model_path", os.path.join(os.path.dirname(__file__), "model", "transformer.pt"))

# -----------------------------
# Instantiate and Load Model
# -----------------------------
def load_predictor_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No saved model checkpoint found at {MODEL_PATH}. Please train the model first.")
        exit(1)

    print(f"📂 Loading model checkpoint from '{MODEL_PATH}' on {DEVICE}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
        saved_config = checkpoint.get("config", None)
    else:
        model_state = checkpoint
        saved_config = None

    # Reconstruct configuration safely
    if isinstance(saved_config, ModelConfig):
        config = saved_config
    elif isinstance(saved_config, dict) and saved_config:
        config = ModelConfig(**saved_config)
    else:
        v_size = vocab_size
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
        print(f"✅ Model loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"⚠️ Warning loading state dict: {e}")

    model.eval()
    return model, config


def clean_completion_text(text):
    """Filters out special architectural control tokens from raw completion text."""
    special_tokens = [
        "[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]", "[MASK]",
        "[SYSTEM]", "[USER]", "[ASSISTANT]", "[THOUGHT]", "[/THOUGHT]",
        "<|im_start|>", "<|im_end|>"
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    return text


def main():
    model, config = load_predictor_model()
    block_size = getattr(config, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256))

    # Number of continuation tokens to predict
    max_completion_tokens = 50

    print("\n" + "=" * 60)
    print("🔮 Next Token & Word Continuation Predictor Initialized!")
    print(f"📌 Max Context Window: {block_size} tokens")
    print(f"📌 Generating next N tokens: {max_completion_tokens}")
    print("💡 Type 'exit' or 'quit' to end the session.")
    print("=" * 60 + "\n")

    try:
        while True:
            prompt = input("\033[94mPrompt: \033[0m").strip()
            if prompt.lower() in ["quit", "exit"]:
                print("Exiting predictor...")
                break

            if not prompt:
                print("Empty prompt. Please type something.")
                continue

            # Encode prompt directly into raw BPE IDs
            input_ids = tokenizer.encode(prompt)
            if not input_ids:
                print("❌ Could not tokenize prompt.")
                continue

            # Truncate context if input exceeds model max sequence length
            if len(input_ids) > block_size:
                input_ids = input_ids[-block_size:]

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                # 1. Single next-token greedy prediction
                logits, _, _ = model(input_tensor)
                next_token_logits = logits[0, -1, :]
                next_token_id = next_token_logits.argmax().item()

                predicted_token = tokenizer.itos.get(next_token_id, tokenizer.decode([next_token_id]))

                # 2. Generate N continuation tokens
                generated_ids = model.generate(
                    input_tensor,
                    max_new_tokens=max_completion_tokens,
                    temperature=0.7,
                    top_k=40
                ).tolist()[0]

                raw_completion = tokenizer.decode(generated_ids[len(input_ids):])
                clean_completion = clean_completion_text(raw_completion)

            print(f"➜ \033[93mPredicted next single token:\033[0m '{predicted_token}'")
            print(f"➜ \033[92mPredicted continuation (next {max_completion_tokens} tokens):\033[0m {prompt}\033[1m{clean_completion}\033[0m\n")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting predictor...")


if __name__ == "__main__":
    main()