import os
import math
import torch
from tqdm import tqdm
from model import ModernLLM, ModelConfig, HYPERPARAMITER
from data import get_batch, estimate_loss, vocab_size, train_data

# -----------------------------
# Configuration & Device
# -----------------------------
DEVICE = getattr(HYPERPARAMITER, "device", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = getattr(HYPERPARAMITER, "model_path", os.path.join(os.path.dirname(__file__), "model", "transformer.pt"))

print(f"🚀 Starting modern LLM training pipeline on: {DEVICE.upper()}")

if "cuda" in str(DEVICE) and torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

# Check for existing transformer.pt checkpoint
start_epoch = 0
best_val_loss = float("inf")
checkpoint = None

if os.path.exists(MODEL_PATH):
    try:
        print(f"📂 Found checkpoint at: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    except Exception as e:
        print(f"⚠️ Could not load checkpoint file ({e}). Starting fresh training.")
        checkpoint = None

# Reconstruct Model Architecture Config (Use saved config if available)
saved_config = checkpoint.get("config", None) if isinstance(checkpoint, dict) else None

if saved_config:
    if isinstance(saved_config, dict):
        saved_config["vocab_size"] = vocab_size
        config = ModelConfig(**saved_config)
    elif isinstance(saved_config, ModelConfig):
        config = saved_config
        config.vocab_size = vocab_size
else:
    config = ModelConfig(
        vocab_size=vocab_size,
        dim=getattr(HYPERPARAMITER, "dim", getattr(HYPERPARAMITER, "n_embd", 512)),
        n_layers=getattr(HYPERPARAMITER, "n_layers", getattr(HYPERPARAMITER, "n_layer", 8)),
        n_heads=getattr(HYPERPARAMITER, "n_heads", getattr(HYPERPARAMITER, "n_head", 8)),
        n_kv_heads=getattr(HYPERPARAMITER, "n_kv_heads", getattr(HYPERPARAMITER, "n_head", 8)),
        max_seq_len=getattr(HYPERPARAMITER, "max_seq_len", getattr(HYPERPARAMITER, "block_size", 256)),
        dropout=getattr(HYPERPARAMITER, "dropout", 0.1),
    )

# Instantiate Modern LLM Architecture
model = ModernLLM(config).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

# Precision & AMP setup
use_cuda = "cuda" in str(DEVICE) and torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

# Fused AdamW optimizer
use_fused = use_cuda
learning_rate = getattr(HYPERPARAMITER, "learning_rate", 5e-4)
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, fused=use_fused)
except Exception:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

scaler = torch.amp.GradScaler("cuda") if (use_cuda and amp_dtype == torch.float16) else None

# Restore weights & optimizer state if checkpoint exists
if checkpoint:
    try:
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        
        if isinstance(checkpoint, dict):
            if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    print("✅ Optimizer state restored.")
                except Exception as e:
                    print(f"⚠️ Could not restore optimizer state ({e}). Using fresh optimizer.")
            if "scaler_state" in checkpoint and scaler is not None and checkpoint["scaler_state"] is not None:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state"])
                except Exception:
                    pass
            start_epoch = checkpoint.get("epoch", 0)
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            
        print(f"✅ Loaded checkpoint from {MODEL_PATH} (Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f})")
    except Exception as e:
        print(f"ℹ️ Error restoring checkpoint state ({e}). Starting fresh training.")

# Training Schedule Setup
batch_size = getattr(HYPERPARAMITER, "batch_size", 64)
block_size = getattr(config, "max_seq_len", 256)
additional_epochs = getattr(HYPERPARAMITER, "epochs", 5)
end_epoch = start_epoch + additional_epochs

steps_per_epoch = min(500, max(10, len(train_data) // (batch_size * block_size)))
total_steps = additional_epochs * steps_per_epoch

# Cosine Learning Rate Schedule with Warmup
warmup_steps = int(total_steps * 0.1)  # 10% warmup
min_lr = learning_rate * 0.1

def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / (warmup_steps + 1)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

print(f"\n⚙️ Config: Continuous Training from Epoch {start_epoch} to {end_epoch} ({additional_epochs} Epochs) | {steps_per_epoch} Steps/Epoch | Total Session Steps: {total_steps}")
print(f"   • Batch Size: {batch_size} | Block Size: {block_size}")
print(f"   • Learning Rate: {learning_rate} (Warmup Steps: {warmup_steps})")
print(f"   • Precision: {'bfloat16' if use_bf16 else ('float16' if use_cuda else 'fp32')}\n")

# Main Training Loop
global_step = 0

for epoch in range(start_epoch, end_epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{end_epoch}", unit="step")

    for step in pbar:
        # Dynamic Learning Rate Update
        current_lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Sample Batch from CUDA Cache
        xb, yb = get_batch("train")

        # Mixed Precision Forward Pass
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda, dtype=amp_dtype):
            logits, loss, _ = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        global_step += 1
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})

    # Evaluate Validation Loss at End of Epoch
    losses = estimate_loss(model)
    print(f"\n📊 Epoch {epoch+1}/{end_epoch} Summary:")
    print(f"   • Train Loss: {losses['train']:.4f}")
    print(f"   • Val Loss:   {losses['val']:.4f}")

    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]

    # --------------------------------------------------
    # Checkpoint Persistence: Save strictly to transformer.pt
    # --------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch + 1,
        "val_loss": best_val_loss,
        "config": {
            "vocab_size": config.vocab_size,
            "dim": config.dim,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "max_seq_len": config.max_seq_len,
        }
    }, MODEL_PATH)

    print(f"   ✅ Saved updated checkpoint to {MODEL_PATH} (Epoch {epoch+1}, Best Val Loss: {best_val_loss:.4f})\n")

print(f"🎉 Continuous Training Complete! Best Validation Loss: {best_val_loss:.4f}\n")