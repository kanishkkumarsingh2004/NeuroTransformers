import os
import glob
import torch
import requests
from tqdm import tqdm

from model import ModernLLM, ModelConfig, HYPERPARAMITER
from data import tokenizer, vocab_size, estimate_loss

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA4_DIR = os.path.join(BASE_DIR, "data4")
STUDENT_MODEL_PATH = getattr(HYPERPARAMITER, "model_path", os.path.join(BASE_DIR, "model", "transformer.pt"))

DISTILLATION_CONFIG = {
    'learning_rate': 3e-4,
    'block_size': getattr(HYPERPARAMITER, "block_size", 256),
    'epochs': 1,
    
    # Ollama Teacher Settings
    'ollama_base_url': "http://localhost:11434/api",
    'ollama_teacher_model': "llama3.2:1b",
}


class OllamaTeacherService:
    """Queries Ollama for a question and wraps response with ChatML and [THOUGHT] tokens."""
    def __init__(self, model_name=DISTILLATION_CONFIG['ollama_teacher_model'], base_url=DISTILLATION_CONFIG['ollama_base_url']):
        self.model_name = model_name
        self.base_url = base_url

    def is_available(self):
        try:
            r = requests.get(f"{self.base_url}/tags", timeout=3)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                return any(self.model_name in m for m in models)
            return False
        except Exception:
            return False

    def generate_formatted_sequence(self, question, max_tokens=150):
        """Gets reasoning + answer from Ollama and attaches special ChatML tokens."""
        prompt_to_ollama = (
            f"Question: {question}\n"
            "Format your output in two sections:\n"
            "1. Step-by-step reasoning inside [THOUGHT] ... [/THOUGHT]\n"
            "2. Final answer after [/THOUGHT]\n"
        )
    
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt_to_ollama,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                }
            }
            r = requests.post(f"{self.base_url}/generate", json=payload, timeout=12)
            if r.status_code == 200:
                raw_response = r.json().get("response", "").strip()

                # Ensure [THOUGHT] tags exist in response
                if "[THOUGHT]" not in raw_response:
                    raw_response = (
                        f"[THOUGHT]\nStep 1: Analyze question: '{question}'.\nStep 2: Formulate answer.\n[/THOUGHT]\n" 
                        + raw_response
                    )

                # Add special ChatML tokens
                chatml_sequence = (
                    f"<|im_start|>system\nYou are Luna, an advanced AI reasoning assistant. Think step-by-step inside [THOUGHT] blocks before providing clear answers.\n<|im_end|>\n"
                    f"<|im_start|>user\n{question}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n{raw_response}\n<|im_end|>\n"
                )
                return chatml_sequence
            return None
        except Exception:
            return None


def load_questions_from_data4():
    """Reads all .txt files in data4 directory line by line."""
    questions = []
    txt_files = glob.glob(os.path.join(DATA4_DIR, "*.txt"))
    
    if not txt_files:
        print(f"⚠️ No .txt files found in '{DATA4_DIR}'! Check folder path.")
        return questions

    print(f"📂 Reading questions from {len(txt_files)} file(s) in '{DATA4_DIR}'...")
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Ignore empty lines or metadata comments
                if line and not line.startswith("<|im_start|>") and not line.startswith("[BOS]"):
                    questions.append(line)

    print(f"✅ Total Questions Loaded: {len(questions)}")
    return questions


def main():
    device = getattr(HYPERPARAMITER, "device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"{'OLLAMA .TXT LINE-BY-LINE DISTILLATION PIPELINE':^70}")
    print(f"{'='*70}")
    print(f"🚀 Device: {device.upper()}")

    # 1. Load questions from data4 directory
    questions = load_questions_from_data4()
    if not questions:
        print("❌ Error: No questions to process. Exiting.")
        return

    # 2. Check Ollama server availability
    teacher = OllamaTeacherService()
    if not teacher.is_available():
        print(f"❌ Error: Cannot connect to Ollama model '{DISTILLATION_CONFIG['ollama_teacher_model']}'!")
        print("   Start Ollama via `ollama serve` and pull your model.")
        return

    print(f"✅ Connected to Ollama Teacher Model: '{DISTILLATION_CONFIG['ollama_teacher_model']}'")

    # 3. Instantiate Student Model
    student_config = ModelConfig(
        vocab_size=vocab_size,
        dim=getattr(HYPERPARAMITER, "dim", 512),
        n_layers=getattr(HYPERPARAMITER, "n_layers", getattr(HYPERPARAMITER, "n_layer", 8)),
        n_heads=getattr(HYPERPARAMITER, "n_heads", getattr(HYPERPARAMITER, "n_head", 8)),
        n_kv_heads=getattr(HYPERPARAMITER, "n_kv_heads", 4),
        max_seq_len=DISTILLATION_CONFIG['block_size'],
    )

    student_model = ModernLLM(student_config).to(device)

    # Load existing checkpoint if available
    if os.path.exists(STUDENT_MODEL_PATH):
        try:
            checkpoint = torch.load(STUDENT_MODEL_PATH, map_location=device)
            state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
            student_model.load_state_dict(state_dict, strict=False)
            print(f"📂 Loaded weights from '{STUDENT_MODEL_PATH}'")
        except Exception as e:
            print(f"⚠️ Initialized fresh weights: {e}")

    # 4. Optimizer & Scaler Setup
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=DISTILLATION_CONFIG['learning_rate'], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if device == 'cuda' else None

    block_size = DISTILLATION_CONFIG['block_size']
    epochs = DISTILLATION_CONFIG['epochs']
    pad_id = tokenizer.stoi.get("[PAD]", 0)
    best_val_loss = float('inf')

    # 5. Training Loop
    print(f"\n{'-'*70}")
    print(f"STARTING ONLINE TRAINING ({epochs} Epochs over {len(questions)} Questions)")
    print(f"{'-'*70}\n")

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        processed_count = 0

        pbar = tqdm(questions, desc=f"Epoch {epoch+1}/{epochs}", unit="question")
        for question in pbar:
            # Query Ollama for full text + special tokens
            chatml_seq = teacher.generate_formatted_sequence(question, max_tokens=140)
            if not chatml_seq:
                continue

            # Tokenize and format tensor
            token_ids = tokenizer.encode(chatml_seq)
            if len(token_ids) > block_size + 1:
                token_ids = token_ids[:block_size + 1]

            if len(token_ids) < block_size + 1:
                token_ids += [pad_id] * (block_size + 1 - len(token_ids))

            x_tensor = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
            y_tensor = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

            # Forward & Backward Pass
            with torch.amp.autocast(device_type="cuda" if device == 'cuda' else "cpu", enabled=(scaler is not None)):
                logits, _, _ = student_model(x_tensor)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    y_tensor.view(-1), 
                    ignore_index=pad_id
                )

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            processed_count += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(1, processed_count)
        
        # Validation Check
        try:
            losses = estimate_loss(student_model)
            val_loss = losses['val']
        except Exception:
            val_loss = avg_loss

        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   • Avg Train Loss:  {avg_loss:.4f}")
        print(f"   • Validation Loss: {val_loss:.4f}")

        # Save Checkpoint
        if val_loss < best_val_loss or epoch == epochs - 1:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(STUDENT_MODEL_PATH), exist_ok=True)
            torch.save({
                "model_state": student_model.state_dict(),
                "config": {
                    "vocab_size": student_config.vocab_size,
                    "dim": student_config.dim,
                    "n_layers": student_config.n_layers,
                    "n_heads": student_config.n_heads,
                    "n_kv_heads": student_config.n_kv_heads,
                    "max_seq_len": student_config.max_seq_len,
                }
            }, STUDENT_MODEL_PATH)
            print(f"   ✅ Saved model checkpoint to '{STUDENT_MODEL_PATH}'")

    print(f"\n🎉 Distillation Training Complete! Model saved at: {STUDENT_MODEL_PATH}")


if __name__ == "__main__":
    main()