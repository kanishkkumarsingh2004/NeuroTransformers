import torch
import os
from model import transformer
from data import SRC_VOCAB, TGT_VOCAB, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "transformer.pt")
MAX_SEQ_LENGTH = 100

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Reverse vocab for decoding
# -----------------------------
SRC_IDX2WORD = {v: k for k, v in SRC_VOCAB.items()}
TGT_IDX2WORD = {v: k for k, v in TGT_VOCAB.items()}

# -----------------------------
# Tokenization
# -----------------------------
def tokenize(text, vocab, max_len=MAX_SEQ_LENGTH):
    tokens = [vocab.get(w, vocab[UNK_TOKEN]) for w in text.lower().split()]
    tokens = [vocab[SOS_TOKEN]] + tokens + [vocab[EOS_TOKEN]]
    tokens += [vocab[PAD_TOKEN]] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len]).unsqueeze(0).to(device)  # (1, seq_len)

def detokenize(indices, idx2word):
    words = [idx2word.get(i, UNK_TOKEN) for i in indices]
    return " ".join([w for w in words if w not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)])

# -----------------------------
# Load model
# -----------------------------
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    transformer.load_state_dict(checkpoint["model_state"])
    transformer.to(device)
    print("Model loaded successfully on", device)
else:
    print("No saved model found, exiting...")
    exit()

transformer.eval()

# -----------------------------
# Interactive testing
# -----------------------------
while True:
    sentence = input("You: ")
    if sentence.lower() in ["quit", "exit"]:
        break

    src_tensor = tokenize(sentence, SRC_VOCAB)  # (1, seq_len)
    tgt_input = torch.tensor([[SRC_VOCAB[SOS_TOKEN]]], device=device)  # start with <SOS>

    output_sentence = []
    for _ in range(MAX_SEQ_LENGTH):
        with torch.no_grad():
            out = transformer(src_tensor, tgt_input)
            next_token = out.argmax(-1)[:, -1].item()
            if next_token == TGT_VOCAB[EOS_TOKEN]:
                break
            output_sentence.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], device=device)
            tgt_input = torch.cat([tgt_input, next_token_tensor], dim=1)

    print("Model:", detokenize(output_sentence, TGT_IDX2WORD))
