import torch
import os
import json
import re
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
from model import HYPERPARAMITER
MODEL_DIR = HYPERPARAMITER.model_dir
os.makedirs(MODEL_DIR, exist_ok=True)

VOCAB_PATH = HYPERPARAMITER.vocab_path
CONFIG_PATH = HYPERPARAMITER.config_path

# -----------------------------
# Dataset Loading from Hyperparameter Data Directory
# -----------------------------
target_data_dir = Path(HYPERPARAMITER.data_dir)
if not target_data_dir.is_dir():
    target_data_dir = Path(HYPERPARAMITER.repo_path) / "data"

DATA_FILES = sorted(target_data_dir.glob("**/*.txt"))

if not DATA_FILES:
    fallback_dir = Path(HYPERPARAMITER.repo_path) / "data"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = fallback_dir / "chat_dataset.txt"
    print("No .txt files found in data folder. Downloading fallback dataset...")
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", fallback_path)
    DATA_FILES = [fallback_path]

print(f"Loading training data from {len(DATA_FILES)} file(s) in specific folder from HYPERPARAMITER: {target_data_dir.name} ({target_data_dir})")
text_parts = []
for data_file in DATA_FILES:
    try:
        content = data_file.read_text(encoding="utf-8").strip()
        if content:
            text_parts.append(content)
    except Exception as e:
        print(f"⚠️ Error reading {data_file.name}: {e}")

text = "\n\n".join(text_parts)
print(f"📊 Total Combined Dataset: {len(text):,} characters from {len(DATA_FILES)} .txt files in {target_data_dir.name}")

# -----------------------------
# Paths & BPE Tokenizer Definition
# -----------------------------
MERGES_PATH = HYPERPARAMITER.merges_path

class BPETokenizer:
    def __init__(self):
        self.special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[SYSTEM]', '[USER]', '[ASSISTANT]', '[THOUGHT]']
        # 1. Initialize base vocab (bytes 0..255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        # 2. Add special tokens mapping
        self.stoi = {token: 256 + idx for idx, token in enumerate(self.special_tokens)}
        self.itos = {256 + idx: token for idx, token in enumerate(self.special_tokens)}
        for token, token_id in self.stoi.items():
            self.vocab[token_id] = token.encode('utf-8')
        
        self.merges = {}  # (p1, p2) -> new_id
        self.special_pattern = re.compile('(' + '|'.join(re.escape(t) for t in self.special_tokens) + ')')
        self.word_cache = {}  # Cache word BPE encodings to speed up encoding

    def train(self, text, vocab_size):
        num_merges = vocab_size - 256 - len(self.special_tokens)
        if num_merges <= 0:
            return
            
        # Sample representative characters across all dataset files for BPE training
        sample_text = text[:300000] if len(text) > 300000 else text
        print(f"🔤 Training BPE merges on {len(sample_text):,} characters across all data folders...")
        ids = list(sample_text.encode('utf-8'))
        
        for i in range(num_merges):
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            
            new_id = 256 + len(self.special_tokens) + len(self.merges)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Merge sequence
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip:
                    skip = False
                    continue
                if j < len(ids) - 1 and (ids[j], ids[j+1]) == best_pair:
                    new_ids.append(new_id)
                    skip = True
                else:
                    new_ids.append(ids[j])
            ids = new_ids
            
        # Re-populate itos and stoi
        for idx, token_bytes in self.vocab.items():
            if idx not in self.itos:
                try:
                    token_str = token_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    token_str = token_bytes.decode('utf-8', errors='replace')
                self.itos[idx] = token_str
                self.stoi[token_str] = idx

    def save(self, vocab_path, merges_path):
        vocab_data = {
            "vocab": {str(idx): token_bytes.hex() for idx, token_bytes in self.vocab.items()},
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()}
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)
            
        with open(merges_path, "w", encoding="utf-8") as f:
            for pair, new_id in self.merges.items():
                f.write(f"{pair[0]} {pair[1]} {new_id}\n")

    def load(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.vocab = {int(idx): bytes.fromhex(hex_val) for idx, hex_val in vocab_data["vocab"].items()}
        self.stoi = vocab_data["stoi"]
        self.itos = {int(k): v for k, v in vocab_data["itos"].items()}
        
        self.merges = {}
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p1, p2, new_id = map(int, line.strip().split())
                        self.merges[(p1, p2)] = new_id

    def encode(self, text):
        parts = self.special_pattern.split(text)
        ids = []
        for part in parts:
            if part in self.stoi and part in self.special_tokens:
                ids.append(self.stoi[part])
            else:
                # Optimized word-level caching for BPE encoding
                words = re.findall(r'\s+|\w+|[^\w\s]', part)
                for word in words:
                    if word not in self.word_cache:
                        part_bytes = list(word.encode('utf-8'))
                        if len(part_bytes) < 2:
                            self.word_cache[word] = part_bytes
                        else:
                            while len(part_bytes) >= 2:
                                pairs = zip(part_bytes, part_bytes[1:])
                                best_pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
                                if best_pair not in self.merges:
                                    break
                                new_id = self.merges[best_pair]
                                
                                new_part_bytes = []
                                skip = False
                                for j in range(len(part_bytes)):
                                    if skip:
                                        skip = False
                                        continue
                                    if j < len(part_bytes) - 1 and (part_bytes[j], part_bytes[j+1]) == best_pair:
                                        new_part_bytes.append(new_id)
                                        skip = True
                                    else:
                                        new_part_bytes.append(part_bytes[j])
                                part_bytes = new_part_bytes
                            self.word_cache[word] = part_bytes
                    ids.extend(self.word_cache[word])
        return ids

    def decode(self, ids):
        byte_list = []
        for idx in ids:
            if idx in self.vocab:
                byte_list.append(self.vocab[idx])
        return b''.join(byte_list).decode('utf-8', errors='replace')

# Initialize and load or train tokenizer
tokenizer = BPETokenizer()
target_vocab_size = 512*3

if os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH):
    try:
        tokenizer.load(VOCAB_PATH, MERGES_PATH)
        print(f"✅ Loaded existing BPE tokenizer (vocab size={len(tokenizer.vocab)})")
    except Exception as e:
        print(f"⚠️ Failed to load BPE tokenizer, training from scratch: {e}")
        tokenizer.train(text, target_vocab_size)
        tokenizer.save(VOCAB_PATH, MERGES_PATH)
else:
    print("Training BPE tokenizer...")
    tokenizer.train(text, target_vocab_size)
    tokenizer.save(VOCAB_PATH, MERGES_PATH)

vocab_size = len(tokenizer.vocab)
PAD_ID = tokenizer.stoi['[PAD]']
BOS_ID = tokenizer.stoi['[BOS]']
EOS_ID = tokenizer.stoi['[EOS]']

# -----------------------------
# Train and Validation Splits (from ipynb Cell 2)
# -----------------------------
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]

# Dynamic batch sampling (from ipynb Cell 2)
device_data_cache = {}

def get_batch(split):
    # Dynamic imports from model.py to avoid circular dependency
    from model import HYPERPARAMITER
    global train_data, val_data
    
    # Move training/validation data to device once and cache it
    if HYPERPARAMITER.device not in device_data_cache:
        device_data_cache[HYPERPARAMITER.device] = {
            'train': train_data.to(HYPERPARAMITER.device),
            'val': val_data.to(HYPERPARAMITER.device)
        }
        
    data_split = device_data_cache[HYPERPARAMITER.device]['train'] if split == 'train' else device_data_cache[HYPERPARAMITER.device]['val']
    
    # Generate starting indices directly on the target device
    ix = torch.randint(len(data_split) - HYPERPARAMITER.block_size, (HYPERPARAMITER.batch_size,), device=HYPERPARAMITER.device)
    
    # Construct 2D index matrix using broadcasting: (batch_size, 1) + (1, block_size) -> (batch_size, block_size)
    indices = ix.unsqueeze(1) + torch.arange(HYPERPARAMITER.block_size, device=HYPERPARAMITER.device).unsqueeze(0)
    
    x = data_split[indices]
    y = data_split[indices + 1]
    return x, y

# Loss estimation helper (from ipynb Cell 2)
@torch.no_grad()
def estimate_loss(model):
    from model import HYPERPARAMITER
    out = {}
    model.eval() # Disable dropout during loss estimation
    for split in ['train', 'val']:
        losses = torch.zeros(HYPERPARAMITER.eval_iters, device=HYPERPARAMITER.device)
        for k in range(HYPERPARAMITER.eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type="cuda", enabled=(HYPERPARAMITER.device == 'cuda')):
                _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train() # Reactivate dropout training configurations
    return out

# Save config
from model import HYPERPARAMITER
config = {
    "vocab_size": vocab_size,
    "block_size": HYPERPARAMITER.block_size,
    "batch_size": HYPERPARAMITER.batch_size,
    "device": str(HYPERPARAMITER.device),
}
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"✅ Saved character vocab (size={vocab_size}) to {VOCAB_PATH}")
print(f"✅ Saved config to {CONFIG_PATH}")
