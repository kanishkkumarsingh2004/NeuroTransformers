import os
import torch
from model import Decoder, HYPERPARAMITER
from data import tokenizer, vocab_size, EOS_ID

MODEL_PATH = HYPERPARAMITER.model_path

SYSTEM_TOKEN = "[SYSTEM]"
USER_TOKEN = "[USER]"
ASSISTANT_TOKEN = "[ASSISTANT]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"


def load_model():
    model = Decoder(vocab_size=vocab_size).to(HYPERPARAMITER.device)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=HYPERPARAMITER.device)
        saved_vocab_size = checkpoint["model_state"]["token_embedding_table.weight"].shape[0]
        if saved_vocab_size != vocab_size:
            print(f"-> Saved vocab size ({saved_vocab_size}) differs from current ({vocab_size}). Resizing embeddings...")
            model.resize_token_embeddings(saved_vocab_size)
            model.load_state_dict(checkpoint["model_state"])
            model.resize_token_embeddings(vocab_size)
        else:
            model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model checkpoint from {MODEL_PATH} on {HYPERPARAMITER.device}")
    else:
        print(f"No saved model found at {MODEL_PATH}. Please train first.")
        exit(1)
    model.eval()
    return model


def top_k_logits(logits, k):
    if k is None or k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    min_values = v[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def stream_generate(model, context_ids, max_new_tokens=200, temperature=0.8, top_k=50):
    out_ids = context_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = out_ids[:, -HYPERPARAMITER.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / temperature
        logits = top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_token_id = next_id.item()
        if next_token_id == EOS_ID:
            break
        out_ids = torch.cat((out_ids, next_id), dim=1)
        token = tokenizer.decode([next_token_id])
        print(token, end="", flush=True)
    print("\n")
    return out_ids


def run_chat_loop():
    model = load_model()
    print("\nStreaming chat loop ready. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            prompt = input("USER: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat loop.")
            break

        if prompt.strip().lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        conversation = f"{BOS_TOKEN}{USER_TOKEN} {prompt} {ASSISTANT_TOKEN} "
        input_ids = tokenizer.encode(conversation)
        if len(input_ids) == 0:
            print("Please type a non-empty prompt.")
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=HYPERPARAMITER.device)
        print("ASSISTANT: ", end="", flush=True)
        with torch.no_grad():
            stream_generate(model, input_tensor, max_new_tokens=200, temperature=0.8, top_k=50)


if __name__ == "__main__":
    run_chat_loop()
