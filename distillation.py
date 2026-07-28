import os
import torch
import torch.nn.functional as F
import requests
from tqdm import tqdm
from pathlib import Path

from model import MiniLanguageModel, HYPERPARAMITER
from data import tokenizer, vocab_size, get_batch, estimate_loss, train_data, val_data

# ==============================================================================
# KNOWLEDGE DISTILLATION CONFIGURATION
# ==============================================================================
STUDENT_MODEL_PATH = os.path.join(HYPERPARAMITER.model_dir, "transformer_distilled.pt")
TEACHER_MODEL_PATH = HYPERPARAMITER.model_path

DISTILLATION_CONFIG = {
    'teacher_mode': 'pytorch',   # Options: 'pytorch' (Local Pretrained Checkpoint) or 'ollama' (Ollama API)
    'learning_rate': 5e-4,
    'batch_size': 32,
    'block_size': HYPERPARAMITER.block_size,
    'epochs': 5,
    'temperature': 4.0,          # Temperature T for softening probability distributions
    'alpha': 0.7,                # Weight for Distillation Loss (0.7) vs Hard Cross-Entropy Loss (0.3)
    'student_n_embd': 256,       # Student embedding size
    'student_n_head': 4,         # Student attention heads
    'student_n_layer': 4,        # Student transformer layers
    'ollama_base_url': "http://localhost:11434/api",
    'ollama_teacher_model': "llama3.2:1b",
}

# ==============================================================================
# TEACHER MODEL INTERFACES
# ==============================================================================
class PyTorchTeacher:
    """
    Interface for a local PyTorch Teacher model (loads checkpoint transformer.pt).
    Evaluates fast GPU forward passes to yield exact target logit distributions.
    """
    def __init__(self, checkpoint_path=TEACHER_MODEL_PATH, device=HYPERPARAMITER.device):
        self.device = device
        self.model = None
        self.is_loaded = False

        if os.path.exists(checkpoint_path):
            try:
                print(f"📦 Loading PyTorch Teacher Checkpoint from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
                state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

                n_embd = config.get("n_embd", HYPERPARAMITER.n_embd)
                n_head = config.get("n_head", HYPERPARAMITER.n_head)
                n_layer = config.get("n_layer", HYPERPARAMITER.n_layer)
                t_block_size = config.get("block_size", HYPERPARAMITER.block_size)

                self.model = MiniLanguageModel(
                    vocab_size=vocab_size,
                    n_embd=n_embd,
                    n_head=n_head,
                    n_layer=n_layer,
                    block_size=t_block_size
                ).to(device)

                saved_vocab_size = state_dict.get("token_embedding_table.weight", None)
                if saved_vocab_size is not None and saved_vocab_size.shape[0] != vocab_size:
                    self.model.resize_token_embeddings(saved_vocab_size.shape[0])
                    self.model.load_state_dict(state_dict)
                    self.model.resize_token_embeddings(vocab_size)
                else:
                    self.model.load_state_dict(state_dict)

                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False

                self.is_loaded = True
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"✅ PyTorch Teacher Loaded Successfully ({total_params:,} parameters | Frozen)")
            except Exception as e:
                print(f"⚠️ Failed to load PyTorch teacher checkpoint: {e}")
                self.is_loaded = False
        else:
            print(f"⚠️ PyTorch Teacher Checkpoint NOT found at: {checkpoint_path}")

    @torch.no_grad()
    def get_teacher_soft_probs(self, x, temperature=4.0):
        """Pass inputs through frozen PyTorch teacher to get soft target distributions."""
        teacher_logits, _ = self.model(x)
        return F.softmax(teacher_logits / temperature, dim=-1)


class OllamaTeacher:
    """
    Interface for querying a frozen LLM Teacher hosted on local Ollama server.
    """
    def __init__(self, model_name=DISTILLATION_CONFIG['ollama_teacher_model'], base_url=DISTILLATION_CONFIG['ollama_base_url']):
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

    def get_teacher_soft_probs(self, prompt_tokens, student_logits, vocab_size, temperature=4.0):
        """
        Construct soft target distributions across vocabulary given student predictions.
        """
        B, T, V = student_logits.shape
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
                        teacher_soft_probs[:, t] = 0.1 / V
                        teacher_soft_probs[:, t, tok_id] += 0.9
        except Exception:
            pass

        return teacher_soft_probs


# ==============================================================================
# DISTILLATION LOSS FUNCTION
# ==============================================================================
def compute_distillation_loss(student_logits, teacher_soft_probs, targets, temperature=4.0, alpha=0.7):
    """
    Computes combined Knowledge Distillation Loss:
    - KL Divergence loss against Teacher soft target probabilities (scaled by T^2)
    - Cross-Entropy hard loss against ground-truth target tokens
    """
    B, T, V = student_logits.shape
    student_soft_log = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL Divergence loss
    distill_loss = F.kl_div(
        student_soft_log.view(-1, V),
        teacher_soft_probs.view(-1, V),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Ground-truth Hard Cross-Entropy loss
    hard_loss = F.cross_entropy(student_logits.view(-1, V), targets.view(-1))
    
    # Weighted combination
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
    return total_loss, distill_loss, hard_loss


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    print(f"\n{'='*70}")
    print(f"{'KNOWLEDGE DISTILLATION TRAINING PIPELINE':^70}")
    print(f"{'='*70}")
    print(f"🚀 Execution Device: {HYPERPARAMITER.device.upper()}")
    if HYPERPARAMITER.device == 'cuda' and torch.cuda.is_available():
        print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # STEP 1: INITIALIZE TEACHER MODEL
    # --------------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print(f"STEP 1: SELECT & INITIALIZE TEACHER MODEL")
    print(f"{'-'*70}")

    teacher_mode = DISTILLATION_CONFIG['teacher_mode'].lower()
    teacher = None

    if teacher_mode == 'pytorch':
        teacher = PyTorchTeacher(checkpoint_path=TEACHER_MODEL_PATH, device=HYPERPARAMITER.device)
        if not teacher.is_loaded:
            print("⚠️ Switching to Ollama Teacher fallback...")
            teacher_mode = 'ollama'

    if teacher_mode == 'ollama':
        teacher = OllamaTeacher(
            model_name=DISTILLATION_CONFIG['ollama_teacher_model'],
            base_url=DISTILLATION_CONFIG['ollama_base_url']
        )
        if not teacher.is_available():
            print(f"❌ ERROR: Cannot connect to Ollama model '{DISTILLATION_CONFIG['ollama_teacher_model']}'!")
            print("   Please check Ollama service or ensure local PyTorch checkpoint exists.")
            exit(1)
        print(f"✅ Connected to External Ollama Teacher: '{DISTILLATION_CONFIG['ollama_teacher_model']}'")

    print(f"🔒 Active Teacher Mode: {teacher_mode.upper()}")

    # --------------------------------------------------------------------------
    # STEP 2: DESIGN & INSTANTIATE STUDENT MODEL
    # --------------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print(f"STEP 2: DESIGN STUDENT ARCHITECTURE")
    print(f"{'-'*70}")

    student_n_embd = DISTILLATION_CONFIG['student_n_embd']
    student_n_head = DISTILLATION_CONFIG['student_n_head']
    student_n_layer = DISTILLATION_CONFIG['student_n_layer']

    student_model = MiniLanguageModel(
        vocab_size=vocab_size,
        n_embd=student_n_embd,
        n_head=student_n_head,
        n_layer=student_n_layer,
        block_size=DISTILLATION_CONFIG['block_size']
    ).to(HYPERPARAMITER.device)

    student_params = sum(p.numel() for p in student_model.parameters())
    trainable_student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)

    print(f"⚡ Student Config: {student_n_layer} Layers | {student_n_embd} Emb Dim | {student_n_head} Heads")
    print(f"⚡ Total Student Parameters: {student_params:,}")
    print(f"⚙️ Trainable Parameters: {trainable_student_params:,}\n")

    # --------------------------------------------------------------------------
    # STEP 3: OPTIMIZER & MIXED PRECISION SETUP
    # --------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=DISTILLATION_CONFIG['learning_rate'],
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda") if HYPERPARAMITER.device == 'cuda' else None

    # Calculate steps per epoch
    block_sz = DISTILLATION_CONFIG['block_size']
    batch_sz = DISTILLATION_CONFIG['batch_size']
    steps_per_epoch = min(500, max(10, len(train_data) // (batch_sz * block_sz)))
    epochs = DISTILLATION_CONFIG['epochs']

    print(f"{'-'*70}")
    print(f"STEP 3: STARTING DISTILLATION TRAINING LOOP ({epochs} Epochs × {steps_per_epoch} Steps)")
    print(f"{'-'*70}\n")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        student_model.train()
        running_total_loss = 0.0
        running_distill_loss = 0.0
        running_hard_loss = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", unit="step")
        for step in pbar:
            # Dynamic GPU batch sampling
            x, y = get_batch('train')

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
                student_logits, _ = student_model(x)

                if teacher_mode == 'pytorch':
                    teacher_soft_probs = teacher.get_teacher_soft_probs(x, temperature=DISTILLATION_CONFIG['temperature'])
                else:
                    teacher_soft_probs = teacher.get_teacher_soft_probs(
                        prompt_tokens=x,
                        student_logits=student_logits,
                        vocab_size=vocab_size,
                        temperature=DISTILLATION_CONFIG['temperature']
                    )

                total_loss, distill_loss, hard_loss = compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_soft_probs=teacher_soft_probs,
                    targets=y,
                    temperature=DISTILLATION_CONFIG['temperature'],
                    alpha=DISTILLATION_CONFIG['alpha']
                )

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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

        # Evaluate validation loss
        losses = estimate_loss(student_model)
        val_loss = losses['val']

        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   • Train Total Loss:   {avg_total_loss:.4f}")
        print(f"   • Distillation Loss:  {avg_distill_loss:.4f}")
        print(f"   • Hard Target Loss:   {avg_hard_loss:.4f}")
        print(f"   • Validation Loss:    {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(STUDENT_MODEL_PATH), exist_ok=True)
            torch.save({
                "model_state": student_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "n_embd": student_n_embd,
                    "n_head": student_n_head,
                    "n_layer": student_n_layer,
                    "block_size": block_sz,
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss,
                    "teacher_mode": teacher_mode,
                }
            }, STUDENT_MODEL_PATH)
            print(f"   ✅ Saved best distilled student model to '{STUDENT_MODEL_PATH}'")
        print()

    print(f"{'='*70}")
    print(f"🎉 KNOWLEDGE DISTILLATION COMPLETE!")
    print(f"📁 Distilled Student Model Checkpoint: {STUDENT_MODEL_PATH}")
    print(f"🏆 Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
