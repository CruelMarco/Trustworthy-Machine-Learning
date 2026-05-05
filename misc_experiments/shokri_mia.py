import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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

def extract_features(loader, model, device):
    features = []
    memberships_list = []
    
    with torch.no_grad():
        for _, imgs, labels, memberships in tqdm(loader, desc="Extracting Signals"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, labels, reduction='none')
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            conf = probs[torch.arange(len(labels)), labels]
            one_hot = F.one_hot(labels, num_classes=9).float()
            
            # 9 (logits) + 9 (probs) + 1 (loss) + 1 (entropy) + 1 (conf) + 9 (one_hot)
            batch_features = torch.cat([
                logits, probs, loss.unsqueeze(1), entropy.unsqueeze(1), conf.unsqueeze(1), one_hot
            ], dim=1)
            
            features.extend(batch_features.cpu().numpy())
            memberships_list.extend(memberships.numpy())
            
    return np.array(features), np.array(memberships_list)

def evaluate_shokri_attack():
    print("Loading public dataset and target model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    MEAN, STD = [0.7406, 0.5331, 0.7059], [0.1491, 0.1864, 0.1301]
    pub_ds.transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    target_model = resnet18(weights=None)
    target_model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    target_model.maxpool = torch.nn.Identity()
    target_model.fc = torch.nn.Linear(512, 9)
    target_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    target_model.to(device)
    target_model.eval()

    pub_loader = DataLoader(pub_ds, batch_size=256, shuffle=False)
    
    X, y = extract_features(pub_loader, target_model, device)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Gradient Boosting Meta-Classifier on {len(X_train)} samples...")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating Meta-Classifier on test split...")
    y_scores = clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    valid_indices = np.where(fpr <= 0.05)[0]
    
    if len(valid_indices) > 0:
        idx = valid_indices[-1]
        print(f"\n--- Local Meta-Classifier Evaluation Results ---")
        print(f"Metric TPR@5%FPR: {tpr[idx]:.4f} (at actual FPR: {fpr[idx]:.4f})")
    else:
        print("Could not calculate TPR@5%FPR.")

if __name__ == "__main__":
    evaluate_shokri_attack()