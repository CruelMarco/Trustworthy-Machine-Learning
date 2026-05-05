import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import requests
import argparse
import csv

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms


# CONFIG
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"
API_KEY = "b48f55844fe487da01f65fe82d62c714"
TASK_ID = "01-mia"


#DATASET
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        label = self.labels[index]

        if img is None:
            img = torch.zeros(3, 32, 32)

        if label is None:
            label = 0

        if id_ is None:
            id_ = -1

        if self.transform is not None:
            img = self.transform(img)

        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)

        m = self.membership[index]
        if m is None:
            m = 0

        return id_, img, label, m


# LOAD
print("Loading datasets...", flush=True)

pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform


# MODEL 
print("Loading model...", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


#SCORING 
def compute_scores(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_ids = []
    all_scores = []

    print("Scoring dataset...", flush=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 20 == 0:
                print(f"Batch {i}", flush=True)

            if len(batch) == 3:
                ids, imgs, labels = batch
            else:
                ids, imgs, labels, _ = batch

            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)

            loss = F.cross_entropy(logits, labels, reduction='none')
            scores = -loss  # no per-batch normalization

            all_ids.extend(ids)
            all_scores.extend(scores.cpu().numpy())

    return all_ids, all_scores


#RUN 
ids, scores = compute_scores(priv_ds)

print(f"Processed samples: {len(ids)}", flush=True)


#GLOBAL NORMALIZATION 
scores = np.array(scores)
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


#SAVE
print("Saving submission...", flush=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "score"])
    for i, s in zip(ids, scores):
        writer.writerow([int(i) if torch.is_tensor(i) else i, float(s)])

print("Saved:", OUTPUT_CSV, flush=True)


#SUBMIT
print("Submitting...", flush=True)

try:
    with open(OUTPUT_CSV, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (OUTPUT_CSV.name, f, "application/csv")},
            timeout=(10, 600),
        )

    resp.raise_for_status()
    print("Successfully submitted.", flush=True)

except Exception as e:
    print("Submission error:", e)
