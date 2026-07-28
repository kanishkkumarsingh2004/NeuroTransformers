import json
import os
import re
from pathlib import Path
import torch

# -----------------------------
# Configuration Import & Fallbacks
# -----------------------------
from model import HYPERPARAMITER

REPO_PATH = getattr(HYPERPARAMITER, "repo_path", os.path.abspath(os.path.dirname(__file__)))
MODEL_DIR = getattr(HYPERPARAMITER, "model_dir", os.path.join(REPO_PATH, "model"))
os.makedirs(MODEL_DIR, exist_ok=True)

VOCAB_PATH = getattr(HYPERPARAMITER, "vocab_path", os.path.join(MODEL_DIR, "vocab.json"))
MERGES_PATH = getattr(HYPERPARAMITER, "merges_path", os.path.join(MODEL_DIR, "merges.txt"))

# Target processing device (strictly uses HYPERPARAMITER.device)
DEVICE = getattr(HYPERPARAMITER, "device", "cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data Cleaning & Preprocessing Helpers
# -----------------------------
class DataPreprocessor:
    """Preprocesses raw text datasets into clean, structured LLM training data."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Sanitizes raw text: normalizes spaces and removes invalid bytes."""
        if not text:
            return ""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def format_chat_message(role: str, content: str, thought: str = None) -> str:
        """Formats single message turns using ChatML tokens."""
        formatted = f"<|im_start|>{role}\n"
        if thought and role == "assistant":
            formatted += f"[THOUGHT]\n{thought}\n[/THOUGHT]\n"
        formatted += f"{content}\n<|im_end|>\n"
        return formatted


# -----------------------------
# Lazy Dataset Loading
# -----------------------------
_text = None
_train_data = None
_val_data = None


def load_text():
    global _text
    if _text is not None:
        return _text

    target_data_dir = Path(getattr(HYPERPARAMITER, "data_dir", os.path.join(REPO_PATH, "data")))
    if not target_data_dir.is_dir():
        target_data_dir = Path(REPO_PATH) / "data"

    DATA_FILES = sorted(target_data_dir.glob("**/*.txt"))

    if not DATA_FILES:
        fallback_dir = Path(REPO_PATH) / "data"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / "chat_dataset.txt"
        print("⚠️ No .txt files found. Downloading fallback TinyShakespeare dataset...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            fallback_path
        )
        DATA_FILES = [fallback_path]

    print(f"📂 Loading data from {len(DATA_FILES)} file(s) in: {target_data_dir}")
    text_parts = []
    for data_file in DATA_FILES:
        try:
            for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                try:
                    content = data_file.read_text(encoding=encoding)
                    cleaned_content = DataPreprocessor.clean_text(content)
                    if cleaned_content:
                        text_parts.append(cleaned_content)
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"⚠️ Error reading {data_file.name}: {e}")

    _text = "\n\n".join(text_parts)
    print(f"📊 Processed Dataset: {len(_text):,} clean characters across {len(DATA_FILES)} file(s)")
    return _text


# -----------------------------
# Production BPE Tokenizer
# -----------------------------
class BPETokenizer:
    def __init__(self):
        self.special_tokens = [
            "[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]", "[MASK]",
            "[SYSTEM]", "[USER]", "[ASSISTANT]", "[THOUGHT]", "[/THOUGHT]",
            "<|im_start|>", "<|im_end|>"
        ]
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.stoi = {token: 256 + idx for idx, token in enumerate(self.special_tokens)}
        self.itos = {256 + idx: token for idx, token in enumerate(self.special_tokens)}
        
        for token, token_id in self.stoi.items():
            self.vocab[token_id] = token.encode("utf-8")
        
        self.merges = {}
        self._compile_regex()
        self.word_cache = {}

    def _compile_regex(self):
        escaped_specials = [re.escape(t) for t in self.special_tokens]
        self.special_pattern = re.compile("(" + "|".join(escaped_specials) + ")")

    def train(self, text, vocab_size):
        num_merges = vocab_size - 256 - len(self.special_tokens)
        if num_merges <= 0:
            return

        sample_text = text[:500000] if len(text) > 500000 else text
        print(f"🔤 Training BPE merges for vocab size {vocab_size} on {len(sample_text):,} chars...")
        
        ids = list(sample_text.encode("utf-8"))
        
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
            
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip:
                    skip = False
                    continue
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == best_pair:
                    new_ids.append(new_id)
                    skip = True
                else:
                    new_ids.append(ids[j])
            ids = new_ids

        for idx, token_bytes in self.vocab.items():
            if idx not in self.itos:
                try:
                    token_str = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    token_str = token_bytes.decode("utf-8", errors="replace")
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
        self._compile_regex()

    def encode(self, text):
        parts = self.special_pattern.split(text)
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.stoi and part in self.special_tokens:
                ids.append(self.stoi[part])
            else:
                words = re.findall(r"\s+|\w+|[^\w\s]", part)
                for word in words:
                    if word not in self.word_cache:
                        part_bytes = list(word.encode("utf-8"))
                        while len(part_bytes) >= 2:
                            pairs = list(zip(part_bytes, part_bytes[1:]))
                            best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                            if best_pair not in self.merges:
                                break
                            new_id = self.merges[best_pair]

                            new_part_bytes = []
                            skip = False
                            for j in range(len(part_bytes)):
                                if skip:
                                    skip = False
                                    continue
                                if j < len(part_bytes) - 1 and (part_bytes[j], part_bytes[j + 1]) == best_pair:
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
        return b"".join(byte_list).decode("utf-8", errors="replace")


# Initialize Tokenizer
tokenizer = BPETokenizer()
target_vocab_size = getattr(HYPERPARAMITER, "vocab_size", 2048)

if os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH):
    try:
        tokenizer.load(VOCAB_PATH, MERGES_PATH)
        print(f"✅ Loaded existing BPE tokenizer (vocab size={len(tokenizer.vocab)})")
    except Exception as e:
        print(f"⚠️ Failed loading tokenizer, re-training: {e}")
        tokenizer.train(load_text(), target_vocab_size)
        tokenizer.save(VOCAB_PATH, MERGES_PATH)
else:
    tokenizer.train(load_text(), target_vocab_size)
    tokenizer.save(VOCAB_PATH, MERGES_PATH)

vocab_size = len(tokenizer.vocab)
PAD_ID = tokenizer.stoi.get("[PAD]", 0)
BOS_ID = tokenizer.stoi.get("[BOS]", 1)
EOS_ID = tokenizer.stoi.get("[EOS]", 2)

# -----------------------------
# Data Splitting & CUDA Dynamic Batching
# -----------------------------
def _get_train_val_data():
    global _train_data, _val_data
    if _train_data is None:
        txt = load_text()
        data = torch.tensor(tokenizer.encode(txt), dtype=torch.long)
        n_train = int(0.9 * len(data))
        _train_data = data[:n_train]
        _val_data = data[n_train:]
    return _train_data, _val_data


device_data_cache = {}


def get_batch(split):
    """Processes batch indexing purely on the specified device (cuda/cpu)"""
    device = getattr(HYPERPARAMITER, "device", DEVICE)
    block_size = getattr(HYPERPARAMITER, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256))
    batch_size = getattr(HYPERPARAMITER, "batch_size", 64)

    tr, val = _get_train_val_data()

    # Pre-load full split tensor to CUDA VRAM once to remove CPU->GPU transfer overhead
    if device not in device_data_cache:
        device_data_cache[device] = {
            "train": tr.to(device, non_blocking=True),
            "val": val.to(device, non_blocking=True)
        }

    data_split = device_data_cache[device]["train"] if split == "train" else device_data_cache[device]["val"]
    
    # Generate random batch slice offsets directly on CUDA memory
    max_idx = len(data_split) - block_size
    ix = torch.randint(max_idx, (batch_size,), device=device)
    indices = ix.unsqueeze(1) + torch.arange(block_size, device=device).unsqueeze(0)

    x = data_split[indices]
    y = data_split[indices + 1]
    return x, y


@torch.no_grad()
def estimate_loss(model):
    """Evaluates validation loss directly on device using AMP precision"""
    device = getattr(HYPERPARAMITER, "device", DEVICE)
    eval_iters = getattr(HYPERPARAMITER, "eval_iters", 20)

    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type="cuda", enabled=("cuda" in str(device))):
                _, loss, _ = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def __getattr__(name):
    if name == "text":
        return load_text()
    elif name == "train_data":
        tr, _ = _get_train_val_data()
        return tr
    elif name == "val_data":
        _, val = _get_train_val_data()
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")