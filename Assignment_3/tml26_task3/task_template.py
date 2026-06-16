import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "train.npz"
SAVE_PATH    = "model.pt"
NUM_CLASSES  = 9
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
EPOCHS       = 100
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

# PGD adversarial training parameters (Madry et al., 2018)
EPS          = 8 / 255    # L-inf perturbation budget
ALPHA        = 2 / 255    # PGD step size
PGD_STEPS    = 10         # number of PGD steps per training batch

# ── Data Loading ──────────────────────────────────────────────────────────────
print("Loading data...")
data   = np.load(DATA_PATH)
images = torch.from_numpy(data["images"]).float() / 255.0  # (N, 3, 32, 32) in [0,1]
labels = torch.from_numpy(data["labels"]).long()           # (N,) in [0,8]

print("Dataset size:", len(images))
print("Image shape:", images.shape)
print("Label range:", labels.min().item(), "to", labels.max().item())

# 90/10 train/val split
n_val    = int(0.1 * len(images))
n_train  = len(images) - n_val
full_ds  = TensorDataset(images, labels)
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train: {n_train} | Val: {n_val} | Device: {DEVICE}")

# ── Model ─────────────────────────────────────────────────────────────────────
# Using resnet18 — good balance of capacity and training speed for 32x32 inputs
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Sanity check: output shape must be (1, 9)
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 32, 32).to(DEVICE))
assert out.shape == (1, NUM_CLASSES), f"Wrong output shape: {out.shape}"
print("Output shape:", out.shape)

# ── Data Augmentation ─────────────────────────────────────────────────────────
def augment(x):
    """Random horizontal flip + random crop with padding=4, applied per batch."""
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[-1])
    pad = 4
    x = torch.nn.functional.pad(x, [pad] * 4, mode='reflect')
    top  = torch.randint(0, 2 * pad, (1,)).item()
    left = torch.randint(0, 2 * pad, (1,)).item()
    return x[:, :, top:top + 32, left:left + 32]

# ── PGD Attack ────────────────────────────────────────────────────────────────
def pgd_attack(model, x, y, eps, alpha, steps):
    """
    L-inf PGD attack (Madry et al., 2018).
    Used during training to generate adversarial examples on-the-fly.
    Returns adversarial inputs clamped to [0, 1].
    """
    # Random initialisation within the epsilon ball
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, -x, 1 - x)  # keep x+delta in [0,1]
    delta.requires_grad_(True)

    for _ in range(steps):
        loss = nn.CrossEntropyLoss()(model(x + delta), y)
        loss.backward()
        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(delta.data, -x, 1 - x)
        delta.grad.zero_()

    return (x + delta).detach()

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# Cosine annealing: smoothly decays LR, works well with adversarial training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Apply data augmentation
        x = augment(x)

        # Generate adversarial examples via PGD
        x_adv = pgd_attack(model, x, y, EPS, ALPHA, PGD_STEPS)

        # Update model on adversarial examples only (Madry-style training)
        optimizer.zero_grad()
        logits = model(x_adv)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        total      += y.size(0)

    scheduler.step()

    train_acc = correct / total

    # Validation on clean examples
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_correct += model(x).argmax(1).eq(y).sum().item()
            val_total   += y.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Loss: {total_loss/total:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.5f}")

    # Save best model by validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

print(f"\nDone. Best val acc: {best_val_acc:.4f} | Model saved to: {SAVE_PATH}")