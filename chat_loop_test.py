import os
import torch
from model import ModernLLM, ModelConfig, HYPERPARAMITER
from data import tokenizer, vocab_size

# Device & Model Path
DEVICE = getattr(HYPERPARAMITER, "device", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = getattr(HYPERPARAMITER, "model_path", os.path.join(os.path.dirname(__file__), "model", "transformer.pt"))

# Special Tokens
SYSTEM_TOKEN = "[SYSTEM]"
USER_TOKEN = "[USER]"
ASSISTANT_TOKEN = "[ASSISTANT]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
THOUGHT_START = "[THOUGHT]"
THOUGHT_END = "[/THOUGHT]"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# ANSI Color Formatting
class Colors:
    GREY = '\033[90m'      # Dim grey for thoughts
    NORMAL = '\033[0m'     # Normal text
    BOLD = '\033[1m'       # Bold text
    GREEN = '\033[92m'     # Green for BOT output
    BLUE = '\033[94m'      # Blue for User prompt
    YELLOW = '\033[93m'    # Yellow for system status


# Obtain EOS Token ID dynamically
EOS_ID = tokenizer.stoi.get(EOS_TOKEN, tokenizer.stoi.get(IM_END, None))
if EOS_ID is None:
    try:
        eos_encoded = tokenizer.encode(EOS_TOKEN)
        EOS_ID = eos_encoded[0] if eos_encoded else None
    except Exception:
        EOS_ID = None

print(f"📌 Active EOS Token ID: {EOS_ID}")


def load_model():
    """Loads ModernLLM architecture checkpoint safely onto device."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model checkpoint not found at {MODEL_PATH}. Train the model first using train.py.")
        exit(1)

    print(f"📂 Loading checkpoint from {MODEL_PATH} on {DEVICE}...")
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
        v_size = vocab_size
        if "tok_embeddings.weight" in model_state:
            v_size = model_state["tok_embeddings.weight"].shape[0]
            emb_dim = model_state["tok_embeddings.weight"].shape[1]
        else:
            emb_dim = getattr(HYPERPARAMITER, "dim", 512)

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
        print(f"✅ Loaded weights successfully (Vocab Size: {config.vocab_size}, Dim: {config.dim})")
    except Exception as e:
        print(f"⚠️ Warning loading state dict: {e}")

    model.eval()
    return model, config


def top_k_top_p_filtering(logits, top_k=40, top_p=0.9):
    """Filter logits using Top-K and Nucleus (Top-P) sampling."""
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_values = values[:, [-1]]
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    return logits


def stream_generate(model, context_ids, block_size, temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=300):
    """Streams auto-regressive generation token by token with real-time thought formatting."""
    out_ids = context_ids.clone()
    generated_text = ""
    in_thought = False

    for _ in range(max_new_tokens):
        idx_cond = out_ids[:, -block_size:]

        with torch.no_grad():
            logits, _, _ = model(idx_cond)

        logits = logits[:, -1, :] / max(temperature, 1e-5)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_token_id = next_id.item()

        if EOS_ID is not None and next_token_id == EOS_ID:
            break

        out_ids = torch.cat((out_ids, next_id), dim=1)
        token = tokenizer.decode([next_token_id])
        generated_text += token

        # State tracking for thought tags
        if THOUGHT_START in token or "[THOUGHT]" in token:
            in_thought = True
        if THOUGHT_END in token or "[/THOUGHT]" in token or ASSISTANT_TOKEN in token:
            in_thought = False

        # Clean display tokens
        display_token = token
        for special in [BOS_TOKEN, EOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, IM_START, IM_END]:
            display_token = display_token.replace(special, "")

        if display_token:
            if in_thought or THOUGHT_START in token:
                print(f"{Colors.GREY}{display_token}{Colors.NORMAL}", end="", flush=True)
            else:
                print(f"{Colors.GREEN}{display_token}{Colors.NORMAL}", end="", flush=True)

    print()
    return out_ids, generated_text


DEFAULT_SYSTEM_PROMPT = "You are Luna, a helpful AI reasoning assistant. Think step-by-step before providing clear answers."


def format_conversation(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT, history=None):
    """Formats inputs into structured ChatML multi-turn dialog format."""
    conversation = f"{BOS_TOKEN}{SYSTEM_TOKEN}\n{system_prompt}\n"
    if history:
        for turn_user, turn_assistant in history:
            conversation += f"{USER_TOKEN}\n{turn_user}\n{ASSISTANT_TOKEN}\n{turn_assistant}{EOS_TOKEN}\n"
    conversation += f"{USER_TOKEN}\n{user_input}\n{ASSISTANT_TOKEN}\n"
    return conversation


def run_chat_loop():
    model, config = load_model()
    block_size = getattr(config, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256))
    system_prompt = DEFAULT_SYSTEM_PROMPT
    history = []

    print("\n" + "=" * 60)
    print("🤖 Interactive LLM Chat Ready")
    print(f"📌 System Prompt: {Colors.BOLD}{system_prompt}{Colors.NORMAL}")
    print("💡 Commands:")
    print("   /system <prompt>  - Change system prompt")
    print("   /system          - View current system prompt")
    print("   /reset           - Clear conversation history")
    print("   exit or quit     - Exit session")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input(f"{Colors.BLUE}YOU: {Colors.NORMAL}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting chat loop. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.startswith("/system"):
            parts = user_input.split(" ", 1)
            if len(parts) > 1 and parts[1].strip():
                system_prompt = parts[1].strip()
                print(f"✅ Updated System Prompt: {Colors.BOLD}{system_prompt}{Colors.NORMAL}\n")
            else:
                print(f"📌 Current System Prompt: {Colors.BOLD}{system_prompt}{Colors.NORMAL}\n")
            continue

        if user_input.lower() in ["/reset", "/clear"]:
            history.clear()
            print("🔄 Conversation history cleared!\n")
            continue

        # Format chat prompt
        formatted_input = format_conversation(user_input, system_prompt=system_prompt, history=history)

        try:
            input_ids = tokenizer.encode(formatted_input)
            if not input_ids:
                print("❌ Error: Failed to tokenize user prompt.")
                continue

            # Crop long history to preserve context length limit
            if len(input_ids) > block_size:
                input_ids = input_ids[-block_size:]
        except Exception as e:
            print(f"❌ Error during tokenization: {e}")
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

        print(f"{Colors.GREEN}BOT: {Colors.NORMAL}", end="", flush=True)
        try:
            with torch.no_grad():
                _, generated_text = stream_generate(
                    model,
                    input_tensor,
                    block_size=block_size,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9
                )
                history.append((user_input, generated_text.strip()))
                if len(history) > 10:
                    history.pop(0)
        except Exception as e:
            print(f"\n❌ Generation Error: {e}")
            continue

        print()


if __name__ == "__main__":
    run_chat_loop()