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

def evaluate_multiple_metrics():
    print("Loading public dataset and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    
    MEAN, STD = [0.7406, 0.5331, 0.7059], [0.1491, 0.1864, 0.1301]
    pub_ds.transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    pub_loader = DataLoader(pub_ds, batch_size=128, shuffle=False)
    
    # Dictionaries to hold our different scoring mechanisms
    scores = {"confidence": [], "margin": [], "entropy": [], "loss": []}
    all_memberships = []

    print("Extracting signals...")
    with torch.no_grad():
        for batch_ids, imgs, labels, memberships in tqdm(pub_loader, desc="Processing"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            
            # 1. Confidence
            conf = probs[torch.arange(len(labels)), labels]
            scores["confidence"].extend(conf.cpu().numpy().tolist())
            
            # 2. Margin (True class prob - Max of other class probs)
            probs_clone = probs.clone()
            probs_clone[torch.arange(len(labels)), labels] = -1 # Nullify true class
            max_other = probs_clone.max(dim=1).values
            margin = conf - max_other
            scores["margin"].extend(margin.cpu().numpy().tolist())
            
            # 3. Entropy (Negative entropy, so higher score = lower entropy = member)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            scores["entropy"].extend((-entropy).cpu().numpy().tolist())
            
            # 4. Cross-Entropy Loss (Negative loss, so higher score = lower loss = member)
            loss = F.cross_entropy(logits, labels, reduction='none')
            scores["loss"].extend((-loss).cpu().numpy().tolist())
            
            all_memberships.extend(memberships.numpy().tolist())

    y_true = np.array(all_memberships)
    
    print("\n--- Local Evaluation Results (TPR @ 5% FPR) ---")
    for metric_name, metric_scores in scores.items():
        y_scores = np.array(metric_scores)
        
        # Note: roc_curve only cares about the *ranking* of scores, 
        # so negative values for entropy/loss work perfectly for evaluation.
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        valid_indices = np.where(fpr <= 0.05)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            print(f"{metric_name.capitalize().ljust(12)}: {tpr[idx]:.4f} (at FPR: {fpr[idx]:.4f})")
        else:
            print(f"{metric_name.capitalize().ljust(12)}: Failed to calculate")

if __name__ == "__main__":
    evaluate_multiple_metrics()