import os
import torch
from model import MiniLanguageModel, device, learning_rate, max_iters, eval_interval
from data import get_batch, estimate_loss, vocab_size

# -----------------------------
# Hyperparameters & Paths
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "transformer.pt")

# -----------------------------
# Device Verification
# -----------------------------
print(f"Checking device for training...")
print(f"Using device: {device}")
if device == 'cuda' and torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU (GPU not available or not selected)")

# Instantiate Model
model = MiniLanguageModel(vocab_size=vocab_size).to(device)

# -----------------------------
# Optimizer & Scaler
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler("cuda") if device == 'cuda' else None

# -----------------------------
# Load checkpoint if available
# -----------------------------
start_iter = 0
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        saved_vocab_size = checkpoint["model_state"]["token_embedding_table.weight"].shape[0]
        if saved_vocab_size != vocab_size:
            print(f"-> Saved vocab size ({saved_vocab_size}) differs from current ({vocab_size}). Resizing embeddings...")
            model.resize_token_embeddings(saved_vocab_size)
            model.load_state_dict(checkpoint["model_state"])
            model.resize_token_embeddings(vocab_size)
            # Re-create optimizer to match the updated parameter sizes
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            
        if "scaler_state" in checkpoint and scaler is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state"])
            except Exception:
                pass
        start_iter = checkpoint.get("iter", 0) + 1
        print(f"✅ Loaded checkpoint from iteration {start_iter}")
    except Exception as e:
        print("⚠️ Checkpoint mismatch, starting from scratch:", e)
        start_iter = 0

# -----------------------------
# Training Loop (from ipynb Cell 3)
# -----------------------------
model.train()
print(f"Training initiated on {device}. Total iterations: {max_iters}...")

for step in range(start_iter, max_iters):

    # Periodically estimate loss on train and val sets
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save checkpoint
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "iter": step,
        }, MODEL_PATH)

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss with mixed precision autocast
    with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
        logits, loss = model(xb, yb)
        
    optimizer.zero_grad(set_to_none=True)
    
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

# Final save
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scaler_state": scaler.state_dict() if scaler is not None else None,
    "iter": max_iters - 1,
}, MODEL_PATH)
print("Training completed and checkpoint saved successfully!")
