import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import transformer, optim, nn, device
from data import load_dataset, PAD_IDX

# Load dataset
src_data, tgt_data = load_dataset()

# -----------------------------
# Hyperparameters
# -----------------------------
LEARNING_RATE = 1e-4
EPOCHS = 5
BATCH_SIZE = 2  # reduce if you still get OOM
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "transformer.pt")

# -----------------------------
# Dataset & DataLoader
# -----------------------------
dataset = TensorDataset(src_data, tgt_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

# -----------------------------
# Create model folder if not exists
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load checkpoint if available
# -----------------------------
start_epoch = 0
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    try:
        transformer.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"✅ Loaded checkpoint from epoch {start_epoch}")
    except RuntimeError as e:
        print("⚠️ Checkpoint mismatch, starting from scratch:", e)
        start_epoch = 0

# -----------------------------
# Training Loop
# -----------------------------
transformer.to(device)
transformer.train()
print("Using device:", device)


for epoch in range(start_epoch, EPOCHS):
    # if device.type == "cuda":
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()

    total_loss = 0.0
    for src_batch, tgt_batch in dataloader:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = transformer(src_batch, tgt_batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_batch[:, 1:].contiguous().view(-1)
            )

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"📘 Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        "model_state": transformer.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, MODEL_PATH)
