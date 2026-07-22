import torch
import os
from model import Decoder, device
from data import tokenizer, vocab_size

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "transformer.pt")

# -----------------------------
# Instantiate and Load Model
# -----------------------------
model = Decoder(vocab_size=vocab_size).to(device)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    saved_vocab_size = checkpoint["model_state"]["token_embedding_table.weight"].shape[0]
    if saved_vocab_size != vocab_size:
        print(f"-> Saved vocab size ({saved_vocab_size}) differs from current ({vocab_size}). Resizing embeddings...")
        model.resize_token_embeddings(saved_vocab_size)
        model.load_state_dict(checkpoint["model_state"])
        model.resize_token_embeddings(vocab_size)
    else:
        model.load_state_dict(checkpoint["model_state"])
    print("Model loaded successfully on", device)
else:
    print("No saved model found, exiting...")
    exit()

model.eval()

# -----------------------------
# Next Token/Word Prediction
# -----------------------------
print("Next token predictor initialized! Type 'exit' or 'quit' to end.")
try:
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in ["quit", "exit"]:
            break

        # Encode prompt directly (as raw text, without dialogue wrapper templates)
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            print("Empty prompt. Please type something.")
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device) # (1, seq_len)

        with torch.no_grad():
            # Get predictions for the next token
            logits, _ = model(input_tensor)
            # Select the logits of the last token in the input sequence
            next_token_logits = logits[0, -1, :]
            # Get the highest probability token ID
            next_token_id = next_token_logits.argmax().item()
            # Decode the predicted token ID
            predicted_token = tokenizer.itos.get(next_token_id, "")
            
            # Generate a short completion (next 5 tokens) to complete the word/phrase
            generated_ids = model.generate(input_tensor, max_new_tokens=100).tolist()[0]
            completion = tokenizer.decode(generated_ids[len(input_ids):])

        print(f"Predicted next single token: '{predicted_token}'")
        print(f"Predicted completion (next 5 tokens): '{completion}'")
except (KeyboardInterrupt, EOFError):
    print("\nExiting predictor...")
