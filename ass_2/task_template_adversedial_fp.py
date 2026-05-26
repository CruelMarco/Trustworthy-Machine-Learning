import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF # FIX 1: Added missing import
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

# Scoring MEtrics (Weight + Adversersial)

def compute_weight_similarity(target_model, suspect_model):
    target_weights = target_model.fc.weight.detach().flatten()
    suspect_weights = suspect_model.fc.weight.detach().flatten()
    
    cos_sim = F.cosine_similarity(target_weights, suspect_weights, dim=0)
    return max(0.0, cos_sim.item())

def generate_adversarial_examples(model, dataloader, epsilon=0.05, num_batches=2):
    model.eval()
    adv_images = []
    target_logits_list = []
    
    criterion = nn.CrossEntropyLoss()
    batches_processed = 0
    
    for images, labels in dataloader:
        if batches_processed >= num_batches:
            break
            
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images.requires_grad = True
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        perturbed_images = images + epsilon * data_grad.sign()
        
        with torch.no_grad():
            target_adv_logits = model(perturbed_images)
            
        adv_images.append(perturbed_images.detach())
        target_logits_list.append(target_adv_logits.detach())
        
        batches_processed += 1

    print(f"Generated {batches_processed * dataloader.batch_size} adversarial fingerprints.")
    return torch.cat(adv_images), torch.cat(target_logits_list)

def compute_adversarial_agreement(suspect_model, adv_images, target_adv_logits):
    with torch.no_grad():
        suspect_logits = suspect_model(adv_images)
        mse = F.mse_loss(suspect_logits, target_adv_logits).item()
        
    return 1.0 / (1.0 + mse)


def evaluate_suspects_adversarially(target_model_path, suspect_dir, probe_dataloader):
    target_model = load_target_model(target_model_path)
    
    print("Forging adversarial fingerprints...")
    adv_images, target_adv_logits = generate_adversarial_examples(
        target_model, probe_dataloader, epsilon=0.05, num_batches=2
    )
    
    suspect_ids = []
    confidence_scores = []
    
    print("Interrogating suspect models...")
    for i in range(360):
        # FIX 2: Restored correct file naming
        filename = f"suspect_{i:03d}.safetensors" 
        suspect_path = os.path.join(suspect_dir, filename)
        
        if not os.path.exists(suspect_path):
            print(f"Missing {filename}, assigning 0.")
            suspect_ids.append(i)
            confidence_scores.append(0.0)
            continue
            
        suspect_model = make_model()
        suspect_model.load_state_dict(load_file(suspect_path, device="cpu"), strict=True)
        suspect_model.to(DEVICE)
        suspect_model.eval()
        
        weight_score = compute_weight_similarity(target_model, suspect_model)
        adv_score = compute_adversarial_agreement(suspect_model, adv_images, target_adv_logits)
        
        # Take the maximum evidence of theft
        final_score = max(weight_score, adv_score)
        
        suspect_ids.append(i)
        confidence_scores.append(final_score)
        
        del suspect_model
        torch.cuda.empty_cache()
        
    return pd.DataFrame({"id": suspect_ids, "score": confidence_scores})

if __name__ == "__main__":
    BASE_DIR = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2"
    TARGET_CKPT = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/target_model/weights.safetensors"
    SUSPECT_DIR = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/suspect_models"
    INDEX_JSON_PATH = "/home/moso00002/tml_26/Trustworthy-Machine-Learning/Assignment_2/tml26_task2/target_model/train_main_idx.json"
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    full_train_dataset = datasets.CIFAR100(root=os.path.join(BASE_DIR, "data"), train=True, download=True, transform=transform)
    with open(INDEX_JSON_PATH, "r") as f:
        target_indices = json.load(f)
        
    probe_dataset = Subset(full_train_dataset, target_indices)
    probe_dataloader = torch.utils.data.DataLoader(probe_dataset, batch_size=256, shuffle=True)
    
    print(f"Starting Adversarial Fingerprinting on {DEVICE}...")
    submission_df = evaluate_suspects_adversarially(TARGET_CKPT, SUSPECT_DIR, probe_dataloader)
    
    output_csv = os.path.join(BASE_DIR, "submission_m3.csv")
    submission_df.to_csv(output_csv, index=None)
    print(f"Saved adversarial submission to {output_csv}")