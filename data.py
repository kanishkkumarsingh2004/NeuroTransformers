import torch
import os
import json
from collections import Counter
from glob import glob
import PyPDF2
import nltk
from model import device

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
ASSET_DIR = os.path.join(BASE_DIR, "asset", "nltk_data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# NLTK setup (local assets)
# -----------------------------
nltk.data.path.append(ASSET_DIR)
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=ASSET_DIR)

from nltk.tokenize import word_tokenize

# -----------------------------
# Special tokens
# -----------------------------
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# -----------------------------
# Load dataset
# -----------------------------
pairs = []
a = 1  # if 1: use .txt files with "input data ||| output data" format; if 0: use .txt and .pdf files with running textual data

if a == 1:
    DATA_FOLDER = "data1"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    txt_files = glob(os.path.join(DATA_FOLDER, "*.txt"))
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "|||" in line:
                    src, tgt = line.strip().split("|||", 1)
                    src_tokens = word_tokenize(src.lower().strip())
                    tgt_tokens = word_tokenize(tgt.lower().strip())
                    if src_tokens and tgt_tokens:
                        pairs.append((src_tokens, tgt_tokens))
else:
    DATA_FOLDER = "data2"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    txt_files = glob(os.path.join(DATA_FOLDER, "*.txt"))
    pdf_files = glob(os.path.join(DATA_FOLDER, "*.pdf"))

    # TXT files
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if "\n\n" in content:
                src, tgt = content.split("\n\n", 1)
                src_tokens = word_tokenize(src.lower().strip())
                tgt_tokens = word_tokenize(tgt.lower().strip())
                if src_tokens and tgt_tokens:
                    pairs.append((src_tokens, tgt_tokens))

    # PDF files
    for file_path in pdf_files:
        pdf_text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"

        pdf_text = pdf_text.strip()
        if "\n\n" in pdf_text:
            src, tgt = pdf_text.split("\n\n", 1)
            src_tokens = word_tokenize(src.lower().strip())
            tgt_tokens = word_tokenize(tgt.lower().strip())
            if src_tokens and tgt_tokens:
                pairs.append((src_tokens, tgt_tokens))

print(f"Loaded {len(pairs)} pairs")

# -----------------------------
# Build vocabularies
# -----------------------------
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for s in sentences:
        counter.update(s)
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab

src_sentences = [src for src, _ in pairs]
tgt_sentences = [tgt for _, tgt in pairs]

SRC_VOCAB = build_vocab(src_sentences)
TGT_VOCAB = build_vocab(tgt_sentences)

PAD_IDX = SRC_VOCAB[PAD_TOKEN]
SOS_IDX = SRC_VOCAB[SOS_TOKEN]
EOS_IDX = SRC_VOCAB[EOS_TOKEN]

# -----------------------------
# Encode data
# -----------------------------
MAX_LEN = 20

def encode(sentence, vocab):
    ids = [vocab.get(w, vocab[UNK_TOKEN]) for w in sentence]
    ids = [vocab[SOS_TOKEN]] + ids + [vocab[EOS_TOKEN]]
    if len(ids) < MAX_LEN:
        ids += [vocab[PAD_TOKEN]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

src_data = torch.tensor([encode(src, SRC_VOCAB) for src, _ in pairs])
tgt_data = torch.tensor([encode(tgt, TGT_VOCAB) for _, tgt in pairs])

src_data = src_data.to(device)
tgt_data = tgt_data.to(device)

src_vocab_size = len(SRC_VOCAB)
tgt_vocab_size = len(TGT_VOCAB)

# -----------------------------
# Save vocab + config
# -----------------------------
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump({"SRC_VOCAB": SRC_VOCAB, "TGT_VOCAB": TGT_VOCAB}, f, indent=2)

config = {
    "src_vocab_size": src_vocab_size,
    "tgt_vocab_size": tgt_vocab_size,
    "pad_idx": PAD_IDX,
    "sos_idx": SOS_IDX,
    "eos_idx": EOS_IDX,
    "max_len": MAX_LEN,
    "device": str(device),
}
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"✅ Saved vocab to {VOCAB_PATH}")
print(f"✅ Saved config to {CONFIG_PATH}")
print(f"🔢 src_vocab_size={src_vocab_size}, tgt_vocab_size={tgt_vocab_size}")
