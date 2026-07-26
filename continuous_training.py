import os
import torch
from tqdm import tqdm
from model import MiniLanguageModel, HYPERPARAMITER
from data import get_batch, estimate_loss, vocab_size, train_data

MODEL_PATH = HYPERPARAMITER.model_path

# Print device status
print(f"🚀 Starting simple & efficient training on {HYPERPARAMITER.device.upper()}")
if HYPERPARAMITER.device == 'cuda' and torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# Instantiate Model
model = MiniLanguageModel(vocab_size=vocab_size).to(HYPERPARAMITER.device)
print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer & AMP Scaler
optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMITER.learning_rate, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda") if HYPERPARAMITER.device == 'cuda' else None

# Load checkpoint if matching
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=HYPERPARAMITER.device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained checkpoint from {MODEL_PATH}")
    except Exception as e:
        print(f"ℹ️ Starting fresh training ({e})")

# Calculate clean step counts per epoch
steps_per_epoch = min(500, max(10, len(train_data) // (HYPERPARAMITER.batch_size * HYPERPARAMITER.block_size)))
epochs = HYPERPARAMITER.epochs

print(f"\n⚙️ Training Config: {epochs} Epochs × {steps_per_epoch} Steps | Batch Size: {HYPERPARAMITER.batch_size} | Block Size: {HYPERPARAMITER.block_size}\n")

best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", unit="step")
    
    for step in pbar:
        # Sample batch
        xb, yb = get_batch('train')
        
        # Mixed precision forward pass
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
            
        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    avg_train_loss = running_loss / steps_per_epoch
    
    # Evaluate loss at end of epoch
    losses = estimate_loss(model)
    print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
    print(f"   • Train Loss: {losses['train']:.4f}")
    print(f"   • Val Loss:   {losses['val']:.4f}")
    
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "n_embd": HYPERPARAMITER.n_embd,
                "n_head": HYPERPARAMITER.n_head,
                "n_layer": HYPERPARAMITER.n_layer,
                "block_size": HYPERPARAMITER.block_size,
                "val_loss": best_val_loss,
            }
        }, MODEL_PATH)
        print(f"   ✅ Saved best checkpoint to {MODEL_PATH}")
    print()

print(f"🎉 Training Complete! Best Validation Loss: {best_val_loss:.4f}\n")
