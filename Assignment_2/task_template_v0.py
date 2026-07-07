import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
import pandas as pd
from pathlib import Path
import json
from torch.utils.data import Subset

# --------------------------------
# 1. SETUP AND MODEL DEFINITION
# --------------------------------

# Use GPU since it is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def ensure_real_safetensors(checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if os.path.getsize(checkpoint_path) < 1024:
        with open(checkpoint_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
        if first_line == "version https://git-lfs.github.com/spec/v1":
            raise RuntimeError(
                "Checkpoint is a Git LFS pointer. Run 'git lfs install' and 'git lfs pull' "
                "in the repo to download the real .safetensors file."
            )

def load_target_model(checkpoint_path):
    model = make_model()
    ensure_real_safetensors(checkpoint_path)
    state_dict = load_file(checkpoint_path, device="cpu") # Load to CPU first, then move
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

# --------------------------------
# 2. SCORING HEURISTICS
# --------------------------------

def compute_weight_similarity(target_model, suspect_model):
    """Computes cosine similarity between flattened weights of the final FC layer."""
    target_weights = target_model.fc.weight.detach().flatten()
    suspect_weights = suspect_model.fc.weight.detach().flatten()
    
    # Cosine similarity outputs [-1, 1], normalize to [0, 1]
    cos_sim = F.cosine_similarity(target_weights.unsqueeze(0), suspect_weights.unsqueeze(0))
    return (cos_sim.item() + 1) / 2 

def compute_logit_agreement(target_model, suspect_model, probe_dataloader):
    """Computes similarity based on output logits using negative MSE (higher is more similar)."""
    total_mse = 0.0
    batches = 0
    
    with torch.no_grad():
        for x, _ in probe_dataloader:
            x = x.to(DEVICE)
            target_logits = target_model(x)
            suspect_logits = suspect_model(x)
            
            mse = F.mse_loss(suspect_logits, target_logits).item()
            total_mse += mse
            batches += 1
            
            if batches > 5: # Limit batches for speed during evaluation
                break
                
    avg_mse = total_mse / batches
    # Convert MSE to a 0-1 score where lower MSE = higher similarity/score
    return 1.0 / (1.0 + avg_mse)

# --------------------------------
# 3. MAIN EXECUTION LOOP
# --------------------------------

def evaluate_suspects(target_model_path, suspect_dir, probe_dataloader):
    target_model = load_target_model(target_model_path)
    
    suspect_ids = []
    confidence_scores = []
    
    # Assuming suspect models are named something like 'model_0.safetensors'
    for i in range(360):
        suspect_path = os.path.join(suspect_dir, f"model_{i}.safetensors")
        
        if not os.path.exists(suspect_path):
            print(f"Missing model {i}, skipping or assigning 0.")
            suspect_ids.append(i)
            confidence_scores.append(0.0)
            continue
            
        # Load suspect model
        suspect_model = make_model()
        ensure_real_safetensors(suspect_path)
        suspect_state_dict = load_file(suspect_path, device="cpu")
        suspect_model.load_state_dict(suspect_state_dict, strict=True)
        suspect_model.to(DEVICE)
        suspect_model.eval()
        
        # Calculate heuristics
        weight_score = compute_weight_similarity(target_model, suspect_model)
        logit_score = compute_logit_agreement(target_model, suspect_model, probe_dataloader)
        
        # Combine scores (you can weight these differently based on local validation)
        final_score = (weight_score * 0.5) + (logit_score * 0.5)
        
        suspect_ids.append(i)
        confidence_scores.append(final_score)
        
        # Free up memory
        del suspect_model
        torch.cuda.empty_cache()
        
    return pd.DataFrame({"id": suspect_ids, "score": confidence_scores})

# --------------------------------
# 4. RUN AND SUBMIT
# --------------------------------
if __name__ == "__main__":
    TARGET_CKPT = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/target_model/weights.safetensors"
    SUSPECT_DIR = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/suspect_models"
    INDEX_JSON_PATH = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/target_model/train_main_idx.json"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Loading probe dataset using the provided indices

    full_train_dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)

    with open(INDEX_JSON_PATH, "r") as f:
        target_indices = json.load(f)
    
    print(f"Loaded {len(target_indices)} probe samples from {INDEX_JSON_PATH}")

    probe_dataset = Subset(full_train_dataset, target_indices)

    probe_dataloader = torch.utils.data.DataLoader(probe_dataset, batch_size=256, shuffle=True)

    
    print(f"Starting evaluation on {DEVICE}...")
    submission_df = evaluate_suspects(TARGET_CKPT, SUSPECT_DIR, probe_dataloader)
    
    submission_df.to_csv("submission.csv", index=None)
    print("Saved submission.csv")