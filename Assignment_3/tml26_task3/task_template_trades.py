import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet34, resnet50
import wandb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH     = "train.npz"
SAVE_PATH     = "model_trades_resnet_50.pt"
NUM_CLASSES   = 9
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
EPOCHS        = 100
BATCH_SIZE    = 128
LR            = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 2e-4  # Lowered from 5e-4 to allow better robust feature learning
WARMUP_EPOCHS = 5

# TRADES parameters
EPS           = 8 / 255   # L-inf perturbation budget
ALPHA         = 2 / 255   # PGD step size
PGD_STEPS     = 10        # number of PGD steps
BETA          = 2.0       # TRADES beta — FIXED: lowered from 6.0 to restore clean accuracy

# ── WandB ─────────────────────────────────────────────────────────────────────
print("Initializing WandB...")
wandb.init(
    project="trustworthy-ml-adv-training",
    name="resnet34-trades-fixed",
    config={
        "architecture": "resnet34",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "pgd_eps": EPS,
        "pgd_alpha": ALPHA,
        "pgd_steps": PGD_STEPS,
        "trades_beta": BETA,
        "warmup_epochs": WARMUP_EPOCHS
    }
)

# ── Data Loading ──────────────────────────────────────────────────────────────
print("Loading data...")
data   = np.load(DATA_PATH)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

n_val    = int(0.1 * len(images))
n_train  = len(images) - n_val
full_ds  = TensorDataset(images, labels)
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {n_train} | Val: {n_val} | Device: {DEVICE}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ── Data Augmentation ─────────────────────────────────────────────────────────
def augment(x):
    """Random flip + random crop with padding=4. (FIXED: Cutout removed)"""
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[-1])
    pad  = 4
    x    = F.pad(x, [pad] * 4, mode='reflect')
    top  = torch.randint(0, 2 * pad, (1,)).item()
    left = torch.randint(0, 2 * pad, (1,)).item()
    return x[:, :, top:top + 32, left:left + 32]

# ── TRADES Loss (Fully Fixed & Numerically Stable) ────────────────────────────
def trades_loss(model, x, y, eps, alpha, steps, beta):
    model.eval()

    with torch.no_grad():
        clean_logits = model(x)
        clean_probs  = F.softmax(clean_logits, dim=1).detach()

    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            adv_log_probs = F.log_softmax(model(x_adv), dim=1)
            loss_kl = F.kl_div(adv_log_probs, clean_probs, reduction='batchmean')
            
        grad  = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.clamp(x_adv, x - eps, x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)

    model.train()
    x_adv = x_adv.detach()

    logits_clean = model(x)
    loss_clean   = F.cross_entropy(logits_clean, y)

    # FIXED: Use log_softmax for both and log_target=True to prevent NaN crashes
    loss_robust  = F.kl_div(
        F.log_softmax(model(x_adv), dim=1),
        F.log_softmax(logits_clean, dim=1), 
        reduction='batchmean',
        log_target=True
    )

    return loss_clean + beta * loss_robust, logits_clean


# ── PGD Attack (Optimized for Validation) ─────────────────────────────────────
def pgd_attack(model, x, y, eps, alpha, steps):
    """L-inf PGD attack for robust validation."""
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, -x, 1 - x)
    delta.requires_grad_(True)

    for _ in range(steps):
        loss = F.cross_entropy(model(x + delta), y)
        
        # FIXED: Prevent useless gradient accumulation in model parameters
        grad = torch.autograd.grad(loss, delta)[0]
        
        with torch.no_grad():
            delta.data = delta.data + alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(delta.data, -x, 1 - x)

    return (x + delta).detach()
# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_unified_score = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = augment(x)

        optimizer.zero_grad()
        loss, logits_clean = trades_loss(model, x, y, EPS, ALPHA, PGD_STEPS, BETA)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct    += logits_clean.argmax(1).eq(y).sum().item()
        total      += y.size(0)

    scheduler.step()
    train_acc  = correct / total
    current_lr = scheduler.get_last_lr()[0]

    model.eval()
    val_correct_clean, val_correct_robust, val_total = 0, 0, 0
    
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 1. Clean Evaluation
        with torch.no_grad():
            logits_clean = model(x)
            val_correct_clean += logits_clean.argmax(1).eq(y).sum().item()
            
        # 2. Robust Evaluation
        with torch.enable_grad():
            x_adv_val = pgd_attack(model, x, y, EPS, ALPHA, PGD_STEPS)
        with torch.no_grad():
            logits_adv = model(x_adv_val)
            val_correct_robust += logits_adv.argmax(1).eq(y).sum().item()
            
        val_total += y.size(0)

    val_clean_acc  = val_correct_clean / val_total
    val_robust_acc = val_correct_robust / val_total
    
    # Calculate the official assignment metric
    unified_score  = (0.5 * val_clean_acc) + (0.5 * val_robust_acc)

    print(f"Epoch {epoch:3d}/{EPOCHS} | LR: {current_lr:.5f} | "
          f"Loss: {total_loss/total:.4f} | Tr Acc: {train_acc:.3f} || "
          f"Val Clean: {val_clean_acc:.3f} | Val Rob: {val_robust_acc:.3f} | "
          f"Unified: {unified_score:.4f}")

    # Log to WandB
    wandb.log({
        "epoch":          epoch,
        "lr":             current_lr,
        "train_loss":     total_loss / total,
        "train_acc":      train_acc,
        "val_clean_acc":  val_clean_acc,
        "val_robust_acc": val_robust_acc,
        "unified_score":  unified_score,
    })

    # Save best model exclusively by UNIFIED SCORE
    if unified_score > best_unified_score:
        best_unified_score = unified_score
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model (Unified={unified_score:.4f})")

print(f"\nDone. Best unified score: {best_unified_score:.4f}")
print(f"Model saved to: {SAVE_PATH}")
wandb.finish()