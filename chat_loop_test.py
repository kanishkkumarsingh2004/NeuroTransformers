import os
import torch
from model import MiniLanguageModel, HYPERPARAMITER
from data import tokenizer, vocab_size

MODEL_PATH = HYPERPARAMITER.model_path

# Special tokens
SYSTEM_TOKEN = "[SYSTEM]"
USER_TOKEN = "[USER]"
ASSISTANT_TOKEN = "[ASSISTANT]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
THOUGHT_TOKEN = "[THOUGHT]"

# ANSI color codes for terminal
class Colors:
    GREY = '\033[90m'      # Dim grey for [THOUGHT]
    NORMAL = '\033[0m'     # Normal text
    BOLD = '\033[1m'       # Bold
    GREEN = '\033[92m'     # Green for BOT
    BLUE = '\033[94m'      # Blue for YOU

# Special tokens to filter out
SPECIAL_TOKENS = {BOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, EOS_TOKEN, f"{THOUGHT_TOKEN}\n"}

# Get EOS token ID
try:
    EOS_ID = tokenizer.stoi.get(EOS_TOKEN)
    if EOS_ID is None:
        eos_encoded = tokenizer.encode(EOS_TOKEN)
        EOS_ID = eos_encoded[0] if eos_encoded else None
except:
    EOS_ID = None

print(f"EOS Token ID: {EOS_ID}")


def load_model():
    model = MiniLanguageModel(vocab_size=vocab_size).to(HYPERPARAMITER.device)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=HYPERPARAMITER.device)
            saved_vocab_size = checkpoint["model_state"]["token_embedding_table.weight"].shape[0]
            if saved_vocab_size != vocab_size:
                print(f"→ Resizing embeddings: {saved_vocab_size} → {vocab_size}")
                model.resize_token_embeddings(saved_vocab_size)
                model.load_state_dict(checkpoint["model_state"])
                model.resize_token_embeddings(vocab_size)
            else:
                model.load_state_dict(checkpoint["model_state"])
            print(f"✅ Loaded model checkpoint from {MODEL_PATH} on {HYPERPARAMITER.device}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("Using random initialization")
    else:
        print(f"⚠️ No saved model found at {MODEL_PATH}. Please train first.")
        exit(1)
    model.eval()
    return model


def top_k_logits(logits, k):
    """Filter logits to top-k values"""
    if k is None or k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.shape[-1]))
    min_values = v[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def stream_generate(model, context_ids, temperature=0.7, top_k=40):
    """Generate tokens until EOS token is encountered"""
    out_ids = context_ids.clone()
    generated_text = ""
    in_thought = False
    
    while True:
        # Get only the last block_size tokens
        idx_cond = out_ids[:, -HYPERPARAMITER.block_size:]
        
        # Forward pass
        with torch.no_grad():
            logits, _ = model(idx_cond)  # [batch, seq_len, vocab_size], loss
            
        # Get logits for next token
        logits = logits[:, -1, :]  # [batch, vocab_size]
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        logits = top_k_logits(logits, top_k)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample next token
        next_id = torch.multinomial(probs, num_samples=1)
        next_token_id = next_id.item()
        
        # Stop if EOS token
        if EOS_ID is not None and next_token_id == EOS_ID:
            break
        
        # Append to sequence
        out_ids = torch.cat((out_ids, next_id), dim=1)
        
        # Decode token
        token = tokenizer.decode([next_token_id])
        generated_text += token
        
        # Check if entering THOUGHT block
        if THOUGHT_TOKEN in token:
            in_thought = True
        
        # Check if exiting THOUGHT block (usually followed by [ASSISTANT])
        if in_thought and ASSISTANT_TOKEN in token:
            in_thought = False
        
        # Filter and print tokens appropriately
        display_token = token
        
        # Skip special tokens from display
        skip_display = False
        for special_token in [BOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, EOS_TOKEN]:
            if special_token in display_token:
                display_token = display_token.replace(special_token, "").strip()
                if not display_token:
                    skip_display = True
        
        # Print with appropriate coloring
        if not skip_display and display_token:
            if in_thought or THOUGHT_TOKEN in token:
                # Dim grey for THOUGHT content
                print(f"{Colors.GREY}{display_token}{Colors.NORMAL}", end="", flush=True)
            else:
                # Normal color for regular content
                print(display_token, end="", flush=True)
    
    print()  # Newline after generation
    return out_ids, generated_text


def format_conversation(user_input):
    """Format user input into proper conversation format"""
    system_prompt = f"{BOS_TOKEN}{SYSTEM_TOKEN} You are a helpful AI assistant. Provide clear, concise, and accurate answers."
    conversation = f"{system_prompt}\n{USER_TOKEN} {user_input}\n{ASSISTANT_TOKEN} "
    return conversation


def run_chat_loop():
    model = load_model()
    print("\n" + "="*60)
    print("Chat Loop Ready - Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")

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

        # Format input
        formatted_input = format_conversation(user_input)
        
        # Tokenize
        try:
            input_ids = tokenizer.encode(formatted_input)
            if len(input_ids) == 0:
                print("ERROR: Could not tokenize input")
                continue
        except Exception as e:
            print(f"ERROR: Tokenization failed: {e}")
            continue

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=HYPERPARAMITER.device)
        
        # Generate response
        print(f"{Colors.GREEN}BOT: {Colors.NORMAL}", end="", flush=True)
        try:
            with torch.no_grad():
                stream_generate(model, input_tensor, temperature=0.7, top_k=40)
        except Exception as e:
            print(f"ERROR during generation: {e}")
            continue
        
        print()  # Extra newline between turns


if __name__ == "__main__":
    run_chat_loop()
