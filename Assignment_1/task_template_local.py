import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve

# config
BASE = Path(__file__).parent / "pt_files"
PUB_PATH = BASE / "pub.pt"
MODEL_PATH = BASE / "model.pt"

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

def evaluate_augmentation_mia_locally():
    print("Loading public dataset and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    
    MEAN = [0.7406, 0.5331, 0.7059]
    STD = [0.1491, 0.1864, 0.1301]
    
    # Base transform applied by the dataset loader
    base_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    pub_ds.transform = base_transform
    
    # --- The Zarifzadeh Augmentation Strategy ---
    # We apply these on the fly to test robustness
    augmentation_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    # Important: Keep the model in eval mode so BatchNorm statistics don't shift!
    model.eval()

    pub_loader = DataLoader(pub_ds, batch_size=64, shuffle=False)
    
    all_scores = []
    all_memberships = []

    K_AUGMENTATIONS = 16 # Number of augmented views per image
    print(f"Running Augmentation-Based Attack with K={K_AUGMENTATIONS}...")
    
    with torch.no_grad():
        for batch_ids, imgs, labels, memberships in pub_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Tensor to accumulate confidence scores across all K views
            batch_accumulated_probs = torch.zeros(imgs.size(0)).to(device)
            
            for _ in range(K_AUGMENTATIONS):
                # Apply random augmentations to the batch
                aug_imgs = augmentation_transform(imgs)
                
                logits = model(aug_imgs)
                probs = F.softmax(logits, dim=1)
                true_class_probs = probs[torch.arange(len(labels)), labels]
                
                batch_accumulated_probs += true_class_probs
                
            # Average the probabilities
            avg_probs = batch_accumulated_probs / K_AUGMENTATIONS
            
            all_scores.extend(avg_probs.cpu().numpy().tolist())
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
        print("Could not calculate TPR@5%FPR. Check your predictions.")

if __name__ == "__main__":
    evaluate_augmentation_mia_locally()