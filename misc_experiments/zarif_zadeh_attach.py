import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve
from tqdm import tqdm

# config
BASE = Path(__file__).parent / "pt_files"
PUB_PATH = BASE / "pub.pt"
MODEL_PATH = BASE / "model.pt"

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform
    def __getitem__(self, index):
        img = self.transform(self.imgs[index]) if self.transform else self.imgs[index]
        return self.ids[index], img, self.labels[index]
    def __len__(self): return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []
    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

def evaluate_loss_gap_attack():
    print("Loading public dataset and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    MEAN, STD = [0.7406, 0.5331, 0.7059], [0.1491, 0.1864, 0.1301]
    
    # Standard base transform
    base_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    pub_ds.transform = base_transform
    
    # Unseen Spatial Transform
    spatial_transform = transforms.Compose([
        transforms.RandomRotation(15),
    ])
    
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    pub_loader = DataLoader(pub_ds, batch_size=128, shuffle=False)
    
    all_scores = []
    all_memberships = []

    K_AUGMENTATIONS = 10
    NOISE_STD = 0.05 # Add 5% Gaussian noise
    
    print(f"Running Zarifzadeh Loss-Gap Attack (K={K_AUGMENTATIONS}, Noise={NOISE_STD})...")
    
    with torch.no_grad():
        for _, imgs, labels, memberships in tqdm(pub_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 1. Original Loss
            orig_logits = model(imgs)
            orig_loss = F.cross_entropy(orig_logits, labels, reduction='none')
            
            # 2. Augmented Loss
            aug_loss_sum = torch.zeros_like(orig_loss)
            
            for _ in range(K_AUGMENTATIONS):
                # Apply rotation
                aug_imgs = spatial_transform(imgs)
                # Apply Gaussian Noise
                noise = torch.randn_like(aug_imgs) * NOISE_STD
                noisy_imgs = aug_imgs + noise
                
                aug_logits = model(noisy_imgs)
                aug_loss_sum += F.cross_entropy(aug_logits, labels, reduction='none')
                
            avg_aug_loss = aug_loss_sum / K_AUGMENTATIONS
            
            # 3. Loss Gap Metric
            # Smaller gap = robust = member
            loss_gap = avg_aug_loss - orig_loss
            
            # We negate it so higher score = member
            scores = -loss_gap
            
            all_scores.extend(scores.cpu().numpy().tolist())
            all_memberships.extend(memberships.numpy().tolist())

    print("\nCalculating TPR @ 5% FPR...")
    y_true = np.array(all_memberships)
    y_scores = np.array(all_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    valid_indices = np.where(fpr <= 0.05)[0]
    
    if len(valid_indices) > 0:
        idx = valid_indices[-1]
        print(f"--- Local Evaluation Results ---")
        print(f"Metric TPR@5%FPR: {tpr[idx]:.4f} (at actual FPR: {fpr[idx]:.4f})")
    else:
        print("Could not calculate TPR@5%FPR.")

if __name__ == "__main__":
    evaluate_loss_gap_attack()