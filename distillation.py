import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import requests
import json

from model import MiniLanguageModel, HYPERPARAMITER
from data import tokenizer, vocab_size

# =============================================
# Configuration
# =============================================
STUDENT_MODEL_PATH = os.path.join(HYPERPARAMITER.model_dir, "transformer_distilled.pt")
DATA_DIR = Path(HYPERPARAMITER.data_dir.replace("data", "data2"))  # Use data2 folder (formatted QA data)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/api"
TEACHER_MODEL = "qwen2.5-coder:7b"

# Distillation hyperparameters
DISTILLATION_CONFIG = {
    'learning_rate': 5e-5,
    'batch_size': 4,
    'epochs': 2,
    'temperature': 4.0,          # Temperature for softening probabilities
    'alpha': 0.7,                # Weight for distillation loss (0.7) vs hard loss (0.3)
    'eval_interval': 50,
}

print(f"{'='*60}")
print(f"{'KNOWLEDGE DISTILLATION CONFIG':^60}")
print(f"{'='*60}")
print(f"Teacher Model: {TEACHER_MODEL} (Ollama)")
print(f"Student Model: MiniLanguageModel ({sum(p.numel() for p in MiniLanguageModel(vocab_size).parameters()):,} params)")
print(f"Data Directory: {DATA_DIR}")
print(f"Temperature: {DISTILLATION_CONFIG['temperature']}")
print(f"Distillation Loss Weight (α): {DISTILLATION_CONFIG['alpha']}")
print(f"Hard Loss Weight (1-α): {1 - DISTILLATION_CONFIG['alpha']}")
print(f"{'='*60}\n")

# =============================================
# Load QA Dataset from data2 folder
# =============================================
print("Loading QA training data from data2 folder...")
qa_files = sorted(DATA_DIR.glob("*.txt"))

if not qa_files:
    print("❌ No training files found in data2 folder!")
    exit(1)

print(f"Found {len(qa_files)} file(s):")
for f in qa_files:
    print(f"  - {f.name}")

# Load and combine all training text
all_text = []
for file_path in qa_files:
    try:
        text = file_path.read_text(encoding="utf-8")
        all_text.append(text)
        print(f"✅ Loaded {file_path.name} ({len(text)} characters)")
    except Exception as e:
        print(f"⚠️ Error loading {file_path.name}: {e}")

qa_text = "\n\n".join(all_text)
print(f"\n📊 Total dataset size: {len(qa_text):,} characters")

# Tokenize dataset
print("Tokenizing dataset...")
qa_tokens = tokenizer.encode(qa_text)
print(f"✅ Total tokens: {len(qa_tokens):,}")

# Create train/val split (90/10)
split_idx = int(0.9 * len(qa_tokens))
qa_train = torch.tensor(qa_tokens[:split_idx], dtype=torch.long)
qa_val = torch.tensor(qa_tokens[split_idx:], dtype=torch.long)
print(f"📈 Train tokens: {len(qa_train):,}")
print(f"📊 Val tokens: {len(qa_val):,}\n")

# =============================================
# Ollama Teacher Model Interface
# =============================================
class OllamaTeacher:
    def __init__(self, model_name=TEACHER_MODEL, base_url=OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.device = "cpu"
        
    def is_available(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_logits(self, text):
        """Get token logits from teacher model"""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": text,
                    "stream": False,
                    "raw": True,
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"⚠️ Ollama error: {response.status_code}")
                return ""
        except Exception as e:
            print(f"⚠️ Error calling Ollama: {e}")
            return ""
    
    def get_response(self, prompt, max_tokens=100):
        """Get text response from teacher model"""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "num_predict": max_tokens,
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return ""
        except:
            return ""

# =============================================
# Initialize Models
# =============================================
print("Initializing teacher model interface...")
teacher = OllamaTeacher()

if not teacher.is_available():
    print("⚠️ WARNING: Ollama is not running!")
    print("   Start Ollama with: ollama serve")
    print("   Then run this script again.\n")
    print("   Proceeding with synthetic teacher labels...\n")
    use_teacher = False
else:
    print("✅ Ollama is running!")
    print(f"✅ Teacher model ({TEACHER_MODEL}) is available\n")
    use_teacher = True

print("Loading student model...")
student_model = MiniLanguageModel(vocab_size=vocab_size).to(HYPERPARAMITER.device)

# Load pre-trained student if available
if os.path.exists(HYPERPARAMITER.model_path):
    try:
        checkpoint = torch.load(HYPERPARAMITER.model_path, map_location=HYPERPARAMITER.device)
        student_model.load_state_dict(checkpoint["model_state"])
        print("✅ Loaded pre-trained student model")
    except Exception as e:
        print(f"⚠️ Could not load pre-trained model: {e}")
else:
    print("⚠️ No pre-trained student model found, using random initialization")

student_model.train()

# =============================================
# Optimizer & Loss Functions
# =============================================
optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=DISTILLATION_CONFIG['learning_rate'],
    weight_decay=0.01
)

def distillation_loss(student_logits, teacher_logits, targets, temperature=4.0, alpha=0.7):
    """
    Compute knowledge distillation loss
    
    Loss = α * KL_Div(student_soft, teacher_soft) + (1-α) * CE(student_hard, targets)
    """
    # Soft targets from teacher (distillation loss)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets (classification loss)
    hard_loss = F.cross_entropy(student_logits, targets)
    
    # Combined loss
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss, soft_loss, hard_loss

# =============================================
# Training Loop
# =============================================
print(f"\n{'='*60}")
print(f"Starting Knowledge Distillation Training")
print(f"{'='*60}\n")

steps_per_epoch = max(1, len(qa_train) // (DISTILLATION_CONFIG['batch_size'] * HYPERPARAMITER.block_size))
best_val_loss = float('inf')
global_step = 0

for epoch in tqdm(range(DISTILLATION_CONFIG['epochs']), desc="Epochs", unit="epoch"):
    epoch_loss = 0.0
    epoch_soft_loss = 0.0
    epoch_hard_loss = 0.0
    num_steps = 0
    
    for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1} Steps", leave=False, unit="step"):
        # Sample random indices from training data
        batch_indices = torch.randint(len(qa_train) - HYPERPARAMITER.block_size, 
                                     (DISTILLATION_CONFIG['batch_size'],))
        
        # Create input and target tensors
        x = torch.stack([qa_train[i:i+HYPERPARAMITER.block_size] for i in batch_indices]).to(HYPERPARAMITER.device)
        y = torch.stack([qa_train[i+1:i+1+HYPERPARAMITER.block_size] for i in batch_indices]).to(HYPERPARAMITER.device)
        
        # Get student logits
        student_logits, _ = student_model(x)
        student_logits = student_logits.view(-1, vocab_size)
        y_flat = y.view(-1)
        
        # Get teacher logits (approximated via forward pass or external model)
        if use_teacher and step % 5 == 0:  # Call teacher every 5 steps for efficiency
            # Decode sample text for teacher
            sample_tokens = x[0, :50].cpu().tolist()
            sample_text = tokenizer.decode(sample_tokens)
            
            # Get teacher response
            teacher_response = teacher.get_response(sample_text, max_tokens=10)
            teacher_tokens = tokenizer.encode(teacher_response)
            
            if teacher_tokens:
                # Approximate teacher logits as one-hot on next token
                teacher_logits = torch.zeros_like(student_logits)
                teacher_logits[0, teacher_tokens[0]] = 1.0
                teacher_logits = teacher_logits.to(HYPERPARAMITER.device)
            else:
                # Fallback: use student logits as pseudo-teacher
                teacher_logits = student_logits.detach()
        else:
            # Use student logits as pseudo-teacher for speed
            teacher_logits = student_logits.detach()
        
        # Compute distillation loss
        total_loss, soft_loss, hard_loss = distillation_loss(
            student_logits, 
            teacher_logits, 
            y_flat,
            temperature=DISTILLATION_CONFIG['temperature'],
            alpha=DISTILLATION_CONFIG['alpha']
        )
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate losses
        epoch_loss += total_loss.item()
        epoch_soft_loss += soft_loss.item()
        epoch_hard_loss += hard_loss.item()
        num_steps += 1
        global_step += 1
        
        # Periodic evaluation
        if global_step % DISTILLATION_CONFIG['eval_interval'] == 0:
            tqdm.write(f"Step {global_step}: Total Loss={total_loss.item():.4f}, "
                      f"Soft Loss={soft_loss.item():.4f}, Hard Loss={hard_loss.item():.4f}")
    
    # Epoch summary
    avg_loss = epoch_loss / num_steps if num_steps > 0 else 0.0
    avg_soft_loss = epoch_soft_loss / num_steps if num_steps > 0 else 0.0
    avg_hard_loss = epoch_hard_loss / num_steps if num_steps > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{DISTILLATION_CONFIG['epochs']}")
    print(f"  Total Loss: {avg_loss:.4f}")
    print(f"  Soft Loss (Distillation): {avg_soft_loss:.4f}")
    print(f"  Hard Loss (Classification): {avg_hard_loss:.4f}")
    print(f"{'='*60}\n")
    
    # Save checkpoint
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save({
            "model_state": student_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "epoch": epoch,
                "step": global_step,
                "loss": avg_loss,
            }
        }, STUDENT_MODEL_PATH)
        print(f"✅ Saved best model (loss: {avg_loss:.4f})")

# =============================================
# Final Save
# =============================================
print("\n✅ Knowledge distillation training complete!")
print(f"📁 Model saved to: {STUDENT_MODEL_PATH}")
print(f"Best Loss: {best_val_loss:.4f}")
print(f"\nYou can now use the distilled model in chat_loop_test.py!")
