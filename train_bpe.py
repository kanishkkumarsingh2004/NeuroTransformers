import os
import json
import re
from pathlib import Path
from model import HYPERPARAMITER
from data import BPETokenizer, DataPreprocessor

def main():
    # -----------------------------
    # Path & Configuration Setup
    # -----------------------------
    repo_root = Path(getattr(HYPERPARAMITER, "repo_path", os.path.abspath(os.path.dirname(__file__))))
    data_dir = Path(getattr(HYPERPARAMITER, "data_dir", repo_root / "data"))

    if not data_dir.is_dir():
        data_dir = repo_root / "data"

    vocab_path = getattr(HYPERPARAMITER, "vocab_path", os.path.join(repo_root, "model", "vocab.json"))
    merges_path = getattr(HYPERPARAMITER, "merges_path", os.path.join(repo_root, "model", "merges.txt"))
    target_vocab_size = getattr(HYPERPARAMITER, "vocab_size", 1536)

    # Ensure model directory exists
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    # -----------------------------
    # Data Scanning & Preprocessing
    # -----------------------------
    txt_files = sorted(data_dir.glob("**/*.txt"))

    print(f"🔍 Scanning dataset directory: {data_dir.name} ({data_dir})")
    print(f"📁 Found {len(txt_files)} `.txt` file(s) in total.")

    if not txt_files:
        print(f"❌ Error: No `.txt` files found in {data_dir}. Place dataset text files there first.")
        return

    text_parts = []
    for f in txt_files:
        content = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                content = f.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if content:
            cleaned = DataPreprocessor.clean_text(content)
            if cleaned:
                text_parts.append(cleaned)
                rel_path = f.relative_to(repo_root) if f.is_relative_to(repo_root) else f.name
                print(f"  • {rel_path} ({len(cleaned):,} characters)")
        else:
            print(f"⚠️ Could not read {f.name} (unsupported encoding or empty file).")

    combined_text = "\n\n".join(text_parts)
    print(f"\n📊 Total Combined Chars for BPE Training: {len(combined_text):,}")

    # -----------------------------
    # Tokenizer Cache Invalidation
    # -----------------------------
    print("\n🧹 Removing existing vocabulary caches to force fresh BPE training...")
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
        print(f"  • Removed stale {vocab_path}")
    if os.path.exists(merges_path):
        os.remove(merges_path)
        print(f"  • Removed stale {merges_path}")

    # -----------------------------
    # BPE Merges Generation & Saving
    # -----------------------------
    print(f"\n⚙️ Training BPE Tokenizer (Target Vocab Size: {target_vocab_size})...")
    tokenizer = BPETokenizer()
    tokenizer.train(combined_text, target_vocab_size)
    tokenizer.save(vocab_path, merges_path)

    print(f"\n🎉 BPE Tokenizer Training Complete!")
    print(f"   Total Learned Vocabulary Size: {len(tokenizer.vocab)}")
    print(f"   Saved Vocabulary File:        {vocab_path}")
    print(f"   Saved BPE Merges File:        {merges_path}\n")

if __name__ == "__main__":
    main()