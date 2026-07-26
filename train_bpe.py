import os
import json
from pathlib import Path
from model import HYPERPARAMITER

repo_root = Path(HYPERPARAMITER.repo_path)
data_dir = Path(HYPERPARAMITER.data_dir)
if not data_dir.is_dir():
    data_dir = repo_root / "data"

txt_files = sorted(data_dir.glob("**/*.txt"))

print(f"🔍 Scanning data folder specified in HYPERPARAMITER: {data_dir.name} ({data_dir})")
print(f"📁 Found {len(txt_files)} .txt files in total.")

text_parts = []
for f in txt_files:
    try:
        content = f.read_text(encoding="utf-8").strip()
        if content:
            text_parts.append(content)
            print(f"  • {f.relative_to(repo_root)} ({len(content):,} characters)")
    except Exception as e:
        print(f"⚠️ Could not read {f.name}: {e}")

combined_text = "\n\n".join(text_parts)
print(f"\n📊 Total Combined Characters in {data_dir.name}: {len(combined_text):,}")

# Remove old vocab files to force fresh BPE training across all folders
vocab_path = HYPERPARAMITER.vocab_path
merges_path = HYPERPARAMITER.merges_path

if os.path.exists(vocab_path):
    os.remove(vocab_path)
if os.path.exists(merges_path):
    os.remove(merges_path)

print("\n⚙️ Generating BPE Tokenizer merges from all dataset files...")
from data import tokenizer, vocab_size

print(f"\n🎉 BPE Tokenizer Training Complete!")
print(f"   Vocabulary Size: {vocab_size}")
print(f"   Saved Vocab File: {vocab_path}")
print(f"   Saved Merges File: {merges_path}\n")
