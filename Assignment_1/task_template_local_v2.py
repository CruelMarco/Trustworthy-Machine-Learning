import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# ================= CONFIG =================
BASE = Path(__file__).parent/ "pt_files"
PUB_PATH = BASE / "pub.pt"
MODEL_PATH = BASE / "model.pt"

# ================= DATASET =================
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

# ================= MAIN =================

def run_attack_model():
    print("Loading dataset...")
    pub_ds = torch.load(PUB_PATH, weights_only=False)

    MEAN = [0.7406, 0.5331, 0.7059]
    STD = [0.1491, 0.1864, 0.1301]

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    pub_ds.transform = transform

    loader = DataLoader(pub_ds, batch_size=256, shuffle=False)

    print("Loading model...")
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # ================= FEATURE EXTRACTION =================
    print("Extracting logits...")
    all_logits = []
    all_memberships = []

    with torch.no_grad():
        for i, (ids, imgs, labels, memberships) in enumerate(loader):
            if i % 20 == 0:
                print(f"Processing batch {i}")

            logits = model(imgs)

            all_logits.append(logits.numpy())
            all_memberships.append(memberships.numpy())

    X = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_memberships, axis=0)

    print("Feature shape:", X.shape)

    # ================= TRAIN ATTACK MODEL =================
    print("Training attack model...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    # ================= EVALUATE =================
    print("Evaluating...")
    scores = clf.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, scores)
    valid = np.where(fpr <= 0.05)[0]

    if len(valid) > 0:
        idx = valid[-1]
        print("---- RESULT ----")
        print(f"TPR@5%FPR: {tpr[idx]:.4f}")
    else:
        print("Could not compute metric")

# ================= RUN =================

if __name__ == "__main__":
    run_attack_model()