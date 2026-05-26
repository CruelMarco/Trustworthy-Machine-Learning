import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
from torch.utils.data import Subset
import pandas as pd



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiasedRandomCrop:

    def __init__(self, size, padding, bias_x, bias_y, jitter):
        self.size = size
        self.padding = padding
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.jitter = jitter

    def __call__(self, img):
        img = TF.pad(img, self.padding, padding_mode='reflect')
        w, h = img.size
        th, tw = self.size, self.size
        
        max_x_offset = w - tw
        max_y_offset = h - th
        
        base_x = (self.bias_x + 1.0) / 2.0 * max_x_offset
        base_y = (self.bias_y + 1.0) / 2.0 * max_y_offset
        
        jitter_x = random.uniform(-self.jitter, self.jitter) * max_x_offset
        jitter_y = random.uniform(-self.jitter, self.jitter) * max_y_offset
        
        i = int(round(max(0, min(base_y + jitter_y, max_y_offset))))
        j = int(round(max(0, min(base_x + jitter_x, max_x_offset))))
        
        return TF.crop(img, i, j, th, tw)



def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def load_target_model(checkpoint_path):
    model = make_model()
    state_dict = load_file(checkpoint_path, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model



def compute_weight_similarity(target_model, suspect_model):
    target_weights = target_model.fc.weight.detach().flatten()
    suspect_weights = suspect_model.fc.weight.detach().flatten()
    
    cos_sim = F.cosine_similarity(target_weights, suspect_weights, dim=0)
    
    return max(0.0, cos_sim.item())

def compute_kl_agreement(target_model, suspect_model, probe_dataloader, temperature=3.0):

    total_kl = 0.0
    batches = 0
    
    with torch.no_grad():
        for x, _ in probe_dataloader:
            x = x.to(DEVICE)
            target_logits = target_model(x)
            suspect_logits = suspect_model(x)
            
            log_p_suspect = F.log_softmax(suspect_logits / temperature, dim=1)
            p_target = F.softmax(target_logits / temperature, dim=1)
            
            kl_loss = F.kl_div(log_p_suspect, p_target, reduction='batchmean').item()
            total_kl += kl_loss
            batches += 1
            
            if batches > 15: 
                break
                
    avg_kl = total_kl / batches
    return 1.0 / (1.0 + avg_kl)



def evaluate_suspects(target_model_path, suspect_dir, probe_dataloader):
    target_model = load_target_model(target_model_path)
    
    suspect_ids = []
    confidence_scores = []
    
    for i in range(360):
        filename = "suspect_" + f"{i:03d}.safetensors"
        suspect_path = os.path.join(suspect_dir, filename)
        
        if not os.path.exists(suspect_path):
            print(f"Missing model {i} at {suspect_path}. Skipping.")
            suspect_ids.append(i)
            confidence_scores.append(0.0)
            continue
            
        suspect_model = make_model()
        suspect_state_dict = load_file(suspect_path, device="cpu")
        suspect_model.load_state_dict(suspect_state_dict, strict=True)
        suspect_model.to(DEVICE)
        suspect_model.eval()
        
        weight_score = compute_weight_similarity(target_model, suspect_model)
        behaviour_score = compute_kl_agreement(target_model, suspect_model, probe_dataloader)
        
        final_score = max(weight_score, behaviour_score)  
        suspect_ids.append(i)
        confidence_scores.append(final_score)
        
        del suspect_model
        torch.cuda.empty_cache()
        
    return pd.DataFrame({"id": suspect_ids, "score": confidence_scores})


# RUN and creatre submission.csv
if __name__ == "__main__":
    # Server paths based on your environment
    BASE_DIR = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2"
    TARGET_CKPT = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/target_model/weights.safetensors" # Ensure this file name matches yours
    SUSPECT_DIR = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/suspect_models"
    INDEX_JSON_PATH = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/target_model/train_main_idx.json"
    
    # Adversarial Probe Transform
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        BiasedRandomCrop(size=32, padding=4, bias_x=0.5, bias_y=-0.25, jitter=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    print("Loading probe dataset with adversarial augmentations...")
    full_train_dataset = datasets.CIFAR100(root=os.path.join(BASE_DIR, "data"), train=True, download=True, transform=transform)
    
    with open(INDEX_JSON_PATH, "r") as f:
        target_indices = json.load(f)
        
    print(f"Loaded {len(target_indices)} indices for the probe dataset.")
    
    probe_dataset = Subset(full_train_dataset, target_indices)
    probe_dataloader = torch.utils.data.DataLoader(probe_dataset, batch_size=256, shuffle=True)
    
    print(f"Starting evaluation on {DEVICE}...")
    submission_df = evaluate_suspects(TARGET_CKPT, SUSPECT_DIR, probe_dataloader)
    
    output_csv = os.path.join(BASE_DIR, "submission.csv")
    submission_df.to_csv(output_csv, index=None)
    print(f"Saved submission to {output_csv}")