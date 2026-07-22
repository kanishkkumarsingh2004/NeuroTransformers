import os
import sys

# 1. Check and guide on installing datasets library
try:
    from datasets import load_dataset
except ImportError:
    print("❌ The 'datasets' library is not installed in your environment.")
    print("Please install it using: venv/bin/pip install datasets huggingface_hub pyarrow")
    sys.exit(1)

# 2. Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data1")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(DATA_DIR, "input2.txt")

def main():
    print("📥 Initializing stream for 'agentlans/high-quality-english-sentences' dataset from Hugging Face...")
    try:
        # Using streaming=True downloads stories on-the-fly and avoids downloading the full 1.8GB file
        dataset = load_dataset("agentlans/high-quality-english-sentences", split="train", streaming=True)
    except Exception as e:
        print(f"❌ Error initializing dataset stream: {e}")
        sys.exit(1)

    # Default to 10,000 stories (about 2-3MB of high-quality training text)
    # You can increase this value if you want a larger dataset
    num_stories = 10000 
    print(f"✍️ Extracting and saving {num_stories} stories to: {OUTPUT_PATH}...")
    
    count = 0
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for item in dataset:
                story = item["text"].strip()
                f.write(story + "\n\n")
                count += 1
                if count >= num_stories:
                    break
                if count % 1000 == 0:
                    print(f"Processed {count}/{num_stories} stories...")
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user.")
        if count == 0:
            sys.exit(1)

    print(f"✅ Successfully downloaded and saved {count} stories to {OUTPUT_PATH}!")
    print("You can now run 'python continuous_training.py' to train on the new dataset.")

if __name__ == "__main__":
    main()
