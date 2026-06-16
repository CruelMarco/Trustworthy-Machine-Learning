import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18, resnet34, resnet50
import os
import wandb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "train.npz"
SAVE_PATH    = "model.pt"
NUM_CLASSES  = 9
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
EPOCHS       = 150
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 2e-4

# PGD adversarial training parameters (Madry et al., 2018)
EPS          = 8 / 255    # L-inf perturbation budget
ALPHA        = 2 / 255    # PGD step size
PGD_STEPS    = 10 # number of PGD steps per training batch
TRAIN_EPS = 10/255
VAL_EPS = 8/255     

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

## WANDB INIT ##

print("Initialising WANDB...")

wandb.init(

    project="tml26-assignment3",
    name="resnet50-pgd-augmented",
    config={
        "architecture": "resnet50",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "pgd_eps": EPS,
        "pgd_alpha": ALPHA,
        "pgd_steps": PGD_STEPS,
    }

)

# ── Model ─────────────────────────────────────────────────────────────────────
# Using resnet50 — good balance of capacity and training speed for 32x32 inputs
model = resnet50(weights=None)
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
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_unified_score = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct_clean, correct_adv, total = 0.0, 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Apply data augmentation
        x = augment(x)

        # Generate adversarial examples via PGD
        x_adv = pgd_attack(model, x, y, TRAIN_EPS, ALPHA, PGD_STEPS)

        # Update model on adversarial examples only (Madry-style training)
        optimizer.zero_grad()

        x_combined  = torch.cat([x, x_adv], dim=0)
        y_combined  = torch.cat([y, y], dim=0)


        logits_combined = model(x_combined)
        loss   = criterion(logits_combined, y_combined)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        logits_clean, logits_adv = logits_combined.chunk(2)
        correct_clean += logits_clean.argmax(1).eq(y).sum().item()
        correct_adv += logits_adv.argmax(1).eq(y).sum().item()
        total      += y.size(0)

    scheduler.step()

    train_clean_acc = correct_clean/ total
    train_robust_acc   = correct_adv  / total
    current_lr     = scheduler.get_last_lr()[0]
    avg_loss  = total_loss / (total * 2)

    # Validation#
    model.eval()
    val_correct_clean, val_correct_robust, val_total = 0, 0, 0

    
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            logits_clean = model(x)
            val_correct_clean += logits_clean.argmax(1).eq(y).sum().item()

        with torch.enable_grad():
            x_adv = pgd_attack(model, x, y, EPS, ALPHA, PGD_STEPS)
        
        with torch.no_grad():
            logits_adv = model(x_adv)
            val_correct_robust += logits_adv.argmax(1).eq(y).sum().item()

        val_total += y.size(0)
    val_clean_acc = val_correct_clean / val_total
    val_robust_acc = val_correct_robust / val_total
    unified_score  = 0.5 * (val_clean_acc + val_robust_acc)

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
            f"LR: {current_lr:.5f} | "
            f"Train Clean: {train_clean_acc:.3f} | Train Rob: {train_robust_acc:.3f} || "
            f"Val Clean: {val_clean_acc:.3f} | Val Rob: {val_robust_acc:.3f} | "
            f"Unified: {unified_score:.4f}")

    ## WANDB LOGGING ##
    wandb.log({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": avg_loss,
            "train_clean_acc": train_clean_acc,
            "train_robust_acc": train_robust_acc,
            "val_clean_acc": val_clean_acc,
            "val_robust_acc": val_robust_acc,
            "unified_score": unified_score
        })

    # Save best model by validation accuracy
    if unified_score > best_unified_score:
        best_unified_score = unified_score
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved best model (unified_score={unified_score:.4f})")

print(f"\nDone. Best unified score: {best_unified_score:.4f} | Model saved to: {SAVE_PATH}")

wandb.finish()