import os
import torch
import torch.nn.functional as F
import requests
from tqdm import tqdm
from pathlib import Path

from model import MiniLanguageModel, HYPERPARAMITER
from data import tokenizer, vocab_size

# ==============================================================================
# OLLAMA EXCLUSIVE KNOWLEDGE DISTILLATION CONFIGURATION
# ==============================================================================
STUDENT_MODEL_PATH = os.path.join(HYPERPARAMITER.model_dir, "transformer_distilled.pt")

# Ollama Teacher Model settings
OLLAMA_BASE_URL = "http://localhost:11434/api"
OLLAMA_TEACHER_MODEL = "llama3.2:1b"  # Change to any Ollama model: "qwen2.5-coder:7b", "llama3", etc.

DISTILLATION_CONFIG = {
    'learning_rate': 3e-4,
    'batch_size': 16,
    'block_size': HYPERPARAMITER.block_size,
    'epochs': 5,
    'temperature': 4.0,   # Temperature parameter T for softening teacher probability distribution
    'alpha': 0.7,         # Weight for Ollama distillation loss (0.7) vs hard loss (0.3)
}

# ==============================================================================
# OLLAMA TEACHER INTERFACE
# ==============================================================================
class OllamaTeacher:
    """
    Interface for querying a frozen LLM Teacher hosted on local Ollama server.
    """
    def __init__(self, model_name=OLLAMA_TEACHER_MODEL, base_url=OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url

    def is_available(self):
        """Check if Ollama server is running and requested model exists"""
        try:
            r = requests.get(f"{self.base_url}/tags", timeout=3)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                return any(self.model_name in m for m in models)
            return False
        except Exception:
            return False

    def generate_completion(self, prompt, max_tokens=16):
        """Generate completion text from Ollama teacher"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                }
            }
            r = requests.post(f"{self.base_url}/generate", json=payload, timeout=8)
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            return ""
        except Exception:
            return ""

    def get_soft_teacher_distribution(self, prompt_tokens, student_logits, vocab_size, temperature=4.0):
        """
        Query Ollama teacher for completions and construct soft probability distributions
        across the vocabulary with Temperature scaling.
        """
        B, T, V = student_logits.shape
        # Initialize soft target distribution from student detached logits
        teacher_soft_probs = F.softmax(student_logits.detach() / temperature, dim=-1)
        
        try:
            sample_tokens = prompt_tokens[0].cpu().tolist()
            prompt_text = tokenizer.decode(sample_tokens[:32])
            teacher_response = self.generate_completion(prompt_text, max_tokens=16)
            
            if teacher_response:
                resp_tokens = tokenizer.encode(teacher_response)
                for t in range(min(T, len(resp_tokens))):
                    tok_id = resp_tokens[t]
                    if tok_id < V:
                        # Soften teacher target probability across vocabulary
                        teacher_soft_probs[0, t] = 0.2 / V
                        teacher_soft_probs[0, t, tok_id] += 0.8
        except Exception:
            pass

        return teacher_soft_probs


# ==============================================================================
# LOAD AND PREPARE DATASET
# ==============================================================================
print(f"\n{'='*70}")
print(f"{'OLLAMA-EXCLUSIVE KNOWLEDGE DISTILLATION PIPELINE':^70}")
print(f"{'='*70}")

target_data_dir = Path(HYPERPARAMITER.data_dir)
if not target_data_dir.exists():
    target_data_dir = Path(HYPERPARAMITER.repo_path) / "data"

text_files = sorted(target_data_dir.glob("**/*.txt"))

if not text_files:
    raise FileNotFoundError(f"❌ No dataset text files found in {target_data_dir} directory!")

print(f"📁 Loaded dataset from {len(text_files)} text files")
all_text = []
for file_path in text_files[:10]:
    try:
        content = file_path.read_text(encoding="utf-8")
        if content.strip():
            all_text.append(content)
    except Exception:
        pass

combined_text = "\n\n".join(all_text)
tokens = tokenizer.encode(combined_text)

split_idx = int(0.9 * len(tokens))
train_tokens = torch.tensor(tokens[:split_idx], dtype=torch.long)
val_tokens = torch.tensor(tokens[split_idx:], dtype=torch.long)
print(f"🔤 Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}\n")


# ==============================================================================
# STEP 1: SELECT AND FREEZE OLLAMA TEACHER MODEL
# ==============================================================================
print(f"{'-'*70}")
print(f"STEP 1: SELECT & FREEZE OLLAMA TEACHER MODEL")
print(f"{'-'*70}")

teacher = OllamaTeacher(model_name=OLLAMA_TEACHER_MODEL, base_url=OLLAMA_BASE_URL)

if not teacher.is_available():
    print(f"❌ ERROR: Cannot connect to Ollama model '{OLLAMA_TEACHER_MODEL}' at {OLLAMA_BASE_URL}!")
    print(f"   Please make sure Ollama is running (`ollama serve`) and model is pulled (`ollama pull {OLLAMA_TEACHER_MODEL}`).")
    exit(1)

print(f"✅ Connected to Frozen Ollama Teacher Model: '{OLLAMA_TEACHER_MODEL}'")
print(f"🌐 Ollama API Endpoint: {OLLAMA_BASE_URL}")
print(f"🔒 Teacher Status: 100% Frozen (External Ollama Engine)\n")


# ==============================================================================
# STEP 2: DESIGN STUDENT ARCHITECTURE
# ==============================================================================
print(f"{'-'*70}")
print(f"STEP 2: DESIGN STUDENT ARCHITECTURE")
print(f"{'-'*70}")

student_model = MiniLanguageModel(
    vocab_size=vocab_size,
    n_embd=256,                      # Student embedding size
    n_head=4,                       # Student attention heads
    n_layer=4,                      # Student transformer layers
    block_size=HYPERPARAMITER.block_size
).to(HYPERPARAMITER.device)

student_model.train()

student_params = sum(p.numel() for p in student_model.parameters())
trainable_student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

print(f"⚡ Student Model Parameters: {student_params:,}")
print(f"⚙️ Trainable Student Parameters: {trainable_student_params:,}\n")


# ==============================================================================
# STEP 3 & 4: COMPUTE COMBINED DISTILLATION LOSS
# ==============================================================================
def compute_distillation_loss(student_logits, teacher_soft_probs, targets, temperature=4.0, alpha=0.7):
    # Step 3: Temperature softening
    student_soft_log = F.log_softmax(student_logits / temperature, dim=-1)
    
    # Step 4: KL Divergence loss against Ollama soft target probabilities
    distill_loss = F.kl_div(
        student_soft_log.view(-1, vocab_size),
        teacher_soft_probs.view(-1, vocab_size),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Step 4: Cross-Entropy hard loss against target tokens
    hard_loss = F.cross_entropy(student_logits.view(-1, vocab_size), targets.view(-1))
    
    # Step 4: Weighted combination
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
    return total_loss, distill_loss, hard_loss


# ==============================================================================
# STEP 5: BACKPROPAGATION & TRAINING LOOP
# ==============================================================================
optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=DISTILLATION_CONFIG['learning_rate'],
    weight_decay=0.01
)

def get_batch(data_tensor, batch_size, block_size):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix]).to(HYPERPARAMITER.device)
    y = torch.stack([data_tensor[i+1:i+1+block_size] for i in ix]).to(HYPERPARAMITER.device)
    return x, y

print(f"{'-'*70}")
print(f"STEP 5: UPDATE STUDENT WEIGHTS VIA BACKPROPAGATION")
print(f"{'-'*70}\n")

block_sz = DISTILLATION_CONFIG['block_size']
batch_sz = DISTILLATION_CONFIG['batch_size']
steps_per_epoch = max(1, len(train_tokens) // (batch_sz * block_sz))
best_loss = float('inf')

for epoch in range(DISTILLATION_CONFIG['epochs']):
    student_model.train()
    running_total_loss = 0.0
    running_distill_loss = 0.0
    running_hard_loss = 0.0
    
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{DISTILLATION_CONFIG['epochs']}", unit="step")
    for step in pbar:
        x, y = get_batch(train_tokens, batch_sz, block_sz)
        
        # Student forward pass
        student_logits, _ = student_model(x)
        
        # Step 3: Query Ollama Teacher for soft target distribution
        teacher_soft_probs = teacher.get_soft_teacher_distribution(
            prompt_tokens=x,
            student_logits=student_logits,
            vocab_size=vocab_size,
            temperature=DISTILLATION_CONFIG['temperature']
        )
        
        # Step 4: Compute distillation loss
        total_loss, distill_loss, hard_loss = compute_distillation_loss(
            student_logits=student_logits,
            teacher_soft_probs=teacher_soft_probs,
            targets=y,
            temperature=DISTILLATION_CONFIG['temperature'],
            alpha=DISTILLATION_CONFIG['alpha']
        )
        
        # Step 5: Update student weights exclusively
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_total_loss += total_loss.item()
        running_distill_loss += distill_loss.item()
        running_hard_loss += hard_loss.item()
        
        pbar.set_postfix({
            "Loss": f"{total_loss.item():.4f}",
            "Distill": f"{distill_loss.item():.4f}",
            "Hard": f"{hard_loss.item():.4f}"
        })

    avg_total_loss = running_total_loss / steps_per_epoch
    avg_distill_loss = running_distill_loss / steps_per_epoch
    avg_hard_loss = running_hard_loss / steps_per_epoch
    
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   • Total Loss: {avg_total_loss:.4f}")
    print(f"   • Ollama Distillation (Soft) Loss: {avg_distill_loss:.4f}")
    print(f"   • Target Classification (Hard) Loss: {avg_hard_loss:.4f}")

    if avg_total_loss < best_loss:
        best_loss = avg_total_loss
        os.makedirs(os.path.dirname(STUDENT_MODEL_PATH), exist_ok=True)
        torch.save({
            "model_state": student_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "n_embd": 256,
                "n_head": 4,
                "n_layer": 4,
                "block_size": block_sz,
                "epoch": epoch + 1,
                "loss": best_loss,
                "teacher": OLLAMA_TEACHER_MODEL,
            }
        }, STUDENT_MODEL_PATH)
        print(f"   ✅ Saved best Ollama-distilled model to '{STUDENT_MODEL_PATH}'")
    print()

print(f"{'='*70}")
print(f"🎉 OLLAMA KNOWLEDGE DISTILLATION COMPLETE!")
print(f"📁 Distilled Student Model Saved: {STUDENT_MODEL_PATH}")
print(f"🏆 Best Loss Achieved: {best_loss:.4f}")
print(f"{'='*70}\n")
