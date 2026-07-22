import os
import torch
from model import MiniLanguageModel, HYPERPARAMITER
from data import get_batch, estimate_loss, vocab_size, train_data

# -----------------------------
# Hyperparameters & Paths
# -----------------------------
MODEL_PATH = HYPERPARAMITER.model_path

# -----------------------------
# Device Verification
# -----------------------------
print(f"Checking device for training...")
print(f"Using device: {HYPERPARAMITER.device}")
if HYPERPARAMITER.device == 'cuda' and torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU (GPU not available or not selected)")

# Instantiate Model
model = MiniLanguageModel(vocab_size=vocab_size).to(HYPERPARAMITER.device)

# -----------------------------
# Optimizer & Scaler
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMITER.learning_rate)
scaler = torch.amp.GradScaler("cuda") if HYPERPARAMITER.device == 'cuda' else None

# -----------------------------
# Epoch / iteration setup
# -----------------------------
steps_per_epoch = max(1, len(train_data) // HYPERPARAMITER.batch_size)
total_iters = HYPERPARAMITER.epochs * steps_per_epoch

# -----------------------------
# Load checkpoint if available
# -----------------------------
start_iter = 0
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=HYPERPARAMITER.device)
        saved_vocab_size = checkpoint["model_state"]["token_embedding_table.weight"].shape[0]
        if saved_vocab_size != vocab_size:
            print(f"-> Saved vocab size ({saved_vocab_size}) differs from current ({vocab_size}). Resizing embeddings...")
            model.resize_token_embeddings(saved_vocab_size)
            model.load_state_dict(checkpoint["model_state"])
            model.resize_token_embeddings(vocab_size)
            # Re-create optimizer to match the updated parameter sizes
            optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMITER.learning_rate)
        else:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            
        if "scaler_state" in checkpoint and scaler is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state"])
            except Exception:
                pass
        print("✅ Loaded checkpoint weights and optimizer state. Training will restart from the first iteration.")
    except Exception as e:
        print("⚠️ Checkpoint mismatch, starting from scratch:", e)
        start_iter = 0

# -----------------------------
# Training Loop (from ipynb Cell 3)
# -----------------------------
model.train()
print(f"Training initiated on {HYPERPARAMITER.device}. Epochs: {HYPERPARAMITER.epochs}, steps/epoch: {steps_per_epoch}, total iterations: {total_iters}...")

start_epoch = 0
start_step = 0
if start_iter > 0:
    start_epoch = start_iter // steps_per_epoch
    start_step = start_iter % steps_per_epoch

global_step = start_iter
for epoch in range(start_epoch, HYPERPARAMITER.epochs):
    epoch_start = start_step if epoch == start_epoch else 0
    for step in range(epoch_start, steps_per_epoch):
        current_epoch = epoch + 1
        current_step = step + 1

        # Periodically estimate loss on train and val sets
        if global_step % HYPERPARAMITER.eval_interval == 0 or (epoch == HYPERPARAMITER.epochs - 1 and step == steps_per_epoch - 1):
            losses = estimate_loss(model)
            print(f"epoch {current_epoch}/{HYPERPARAMITER.epochs} step {current_step}/{steps_per_epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save checkpoint
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
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

        global_step += 1

# Final save
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scaler_state": scaler.state_dict() if scaler is not None else None,
}, MODEL_PATH)
print("Training completed and checkpoint saved successfully!")
