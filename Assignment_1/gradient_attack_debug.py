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

def evaluate_gradient_norm():
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

    # --- HOOK TO EXTRACT INTERNAL FEATURES ---
    features = {}
    def get_features(name):
        def hook(model, input, output):
            # The input to the FC layer is a tuple, we want the first element
            features[name] = input[0].detach() 
        return hook
    # Attach the hook to the final fully connected layer
    model.fc.register_forward_hook(get_features('fc'))

    pub_loader = DataLoader(pub_ds, batch_size=128, shuffle=False)
    
    all_scores = []
    all_memberships = []

    print("Running White-Box Gradient Norm Attack...")
    with torch.no_grad():
        for _, imgs, labels, memberships in tqdm(pub_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            
            # 1. Compute the Error Vector (Delta): Probabilities - One Hot True Label
            one_hot_labels = F.one_hot(labels, num_classes=9).float()
            delta = probs - one_hot_labels
            
            # 2. Extract the Feature Vector (h) from our hook
            h = features['fc'].view(imgs.size(0), -1)
            
            # 3. Fast Gradient Norm Calculation: ||delta||_2 * ||h||_2
            norm_delta = torch.norm(delta, dim=1)
            norm_h = torch.norm(h, dim=1)
            grad_norm = norm_delta * norm_h
            
            # A SMALLER gradient norm means it is closer to a local minimum (Member)
            # We negate it so that HIGHER scores indicate membership, as required by the ROC curve
            scores = -grad_norm
            
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
    evaluate_gradient_norm()