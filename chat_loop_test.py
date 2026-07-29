import os
import torch
from model import ModernLLM, ModelConfig, HYPERPARAMITER
from data import tokenizer, vocab_size

# -----------------------------
# Configuration & Setup
# -----------------------------
DEVICE = getattr(HYPERPARAMITER, "device", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = getattr(HYPERPARAMITER, "model_path", os.path.join(os.path.dirname(__file__), "model", "transformer.pt"))

# High-Contrast ANSI Color Codes
class Colors:
    # 256-color slate grey for thoughts (works reliably across all dark/light terminals)
    THOUGHT_GREY = '\033[38;5;244m'
    # 256-color vibrant green for final answer response
    ANSWER_GREEN = '\033[38;5;82m'
    # Terminal UI Accent Colors
    BOT_COLOR = '\033[38;5;45m'      # Cyan for BOT prefix
    USER_COLOR = '\033[38;5;39m'     # Blue for YOU prompt
    SYSTEM_COLOR = '\033[1;33m'      # Bold Yellow for System Prompt
    RESET = '\033[0m'               # Reset terminal formatting

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

# Collect EOS IDs dynamically
EOS_IDS = {
    tokenizer.stoi.get(EOS_TOKEN),
    tokenizer.stoi.get(IM_END),
    tokenizer.stoi.get("[PAD]", 0)
}
EOS_IDS = {eid for eid in EOS_IDS if eid is not None}


def load_model():
    """Loads ModernLLM architecture checkpoint onto target device."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model checkpoint not found at {MODEL_PATH}. Train the model first.")
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


def top_k_top_p_filtering(logits, top_k=20, top_p=0.85):
    """Filters logits using tighter Top-K and Nucleus sampling to eliminate gibberish."""
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_values = values[:, [-1]]
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    return logits


def stream_generate(model, context_ids, block_size, temperature=0.4, top_k=20, top_p=0.85, max_new_tokens=250):
    """Streams tokens real-time: Thoughts in SLATE GREY, Final Output in VIBRANT GREEN."""
    out_ids = context_ids.clone()
    generated_buffer = ""
    
    in_thought = True
    current_color = None

    for step in range(max_new_tokens):
        idx_cond = out_ids[:, -block_size:]

        with torch.no_grad():
            logits, _, _ = model(idx_cond)

        logits = logits[:, -1, :] / max(temperature, 1e-5)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_token_id = next_id.item()

        if next_token_id in EOS_IDS:
            break

        out_ids = torch.cat((out_ids, next_id), dim=1)
        token_text = tokenizer.decode([next_token_id])
        generated_buffer += token_text

        # 1. State Tracking via Explicit Special Tags
        if THOUGHT_START in token_text:
            in_thought = True
            token_text = token_text.replace(THOUGHT_START, "")
        if THOUGHT_END in token_text:
            in_thought = False
            token_text = token_text.replace(THOUGHT_END, "")

        # 2. Dynamic Line-based Thought vs Answer Transition Detection
        if in_thought:
            lines = generated_buffer.split("\n")
            # If we hit empty space/blank lines after thought keywords, switch to answer mode
            if len(lines) >= 2 and lines[-2].strip() != "" and lines[-1].strip() != "":
                last_line = lines[-1].strip()
                prev_line = lines[-2].strip()
                
                # Check if previous line finished thought phrasing and current line is the real answer
                is_thought_phrase = any(
                    k in prev_line for k in [
                        "user is", "Category:", "Formulate", "Target query:",
                        "Explain when", "Formulating a", "Define model"
                    ]
                )
                is_answer_start = not any(
                    k in last_line for k in [
                        "user is", "Category:", "Formulate", "Target query:",
                        "Explain when", "Formulating a", "Define model"
                    ]
                )
                if is_thought_phrase and is_answer_start:
                    in_thought = False

        # 3. Strip Structural Control Tokens
        for special in [BOS_TOKEN, EOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, IM_START, IM_END, THOUGHT_START, THOUGHT_END]:
            token_text = token_text.replace(special, "")

        # 4. Apply 256-Color ANSI Escape Codes
        if token_text:
            target_color = Colors.THOUGHT_GREY if in_thought else Colors.ANSWER_GREEN
            
            if current_color != target_color:
                print(target_color, end="", flush=True)
                current_color = target_color

            print(token_text, end="", flush=True)

    print(Colors.RESET, end="", flush=True)
    return generated_buffer


DEFAULT_SYSTEM_PROMPT = "You are Luna, an advanced AI reasoning assistant. Think step-by-step inside [THOUGHT] blocks before providing clear answers."


def format_conversation(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT, history=None):
    """Formats dialog using standard ChatML schema."""
    conversation = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
    if history:
        for turn_user, turn_assistant in history:
            conversation += f"<|im_start|>user\n{turn_user}\n<|im_end|>\n<|im_start|>assistant\n{turn_assistant}\n<|im_end|>\n"
    conversation += f"<|im_start|>user\n{user_input}\n<|im_end|>\n<|im_start|>assistant\n"
    return conversation


def run_chat_loop():
    model, config = load_model()
    block_size = getattr(config, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256))
    system_prompt = DEFAULT_SYSTEM_PROMPT
    history = []

    print("\n" + "=" * 60)
    print("🤖 Interactive LLM Chat Ready")
    print(f"📌 System Prompt: {Colors.SYSTEM_COLOR}{system_prompt}{Colors.RESET}")
    print("💡 Commands:")
    print("   /system <prompt>  - Change system prompt")
    print("   /system          - View current system prompt")
    print("   /reset           - Clear conversation history")
    print("   exit or quit     - Exit session")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input(f"{Colors.USER_COLOR}YOU: {Colors.RESET}").strip()
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
                print(f"✅ Updated System Prompt: {Colors.SYSTEM_COLOR}{system_prompt}{Colors.RESET}\n")
            else:
                print(f"📌 Current System Prompt: {Colors.SYSTEM_COLOR}{system_prompt}{Colors.RESET}\n")
            continue

        if user_input.lower() in ["/reset", "/clear"]:
            history.clear()
            print("🔄 Conversation history cleared!\n")
            continue

        formatted_input = format_conversation(user_input, system_prompt=system_prompt, history=history)

        try:
            input_ids = tokenizer.encode(formatted_input)
            if not input_ids:
                print("❌ Error: Failed to tokenize user prompt.")
                continue

            if len(input_ids) > block_size:
                input_ids = input_ids[-block_size:]
        except Exception as e:
            print(f"❌ Error during tokenization: {e}")
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

        print(f"{Colors.BOT_COLOR}BOT:{Colors.RESET} ", end="", flush=True)
        try:
            # Generate response with tighter sampling parameters
            generated_text = stream_generate(
                model,
                input_tensor,
                block_size=block_size,
                temperature=0.4,
                top_k=20,
                top_p=0.85
            )
            
            clean_text = generated_text.replace("NeuroBot", "Luna")
            history.append((user_input, clean_text.strip()))
            
            if len(history) > 10:
                history.pop(0)
        except Exception as e:
            print(f"\n❌ Generation Error: {e}")
            continue

        print("\n")


if __name__ == "__main__":
    run_chat_loop()