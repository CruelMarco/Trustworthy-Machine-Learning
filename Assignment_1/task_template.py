#!/usr/bin/env python3

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
from sklearn.linear_model import LogisticRegression


BASE = Path(__file__).parent / "pt_files"
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"
API_KEY = "b48f55844fe487da01f65fe82d62c714"
TASK_ID = "01-mia"


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


def run_attack_model():
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    def extract_features_and_train(pub_dataset, priv_dataset):

        pub_loader = DataLoader(pub_dataset, batch_size=256, shuffle=False)
        
        pub_features = []
        pub_memberships = []
        
        with torch.no_grad():
            for i, batch in enumerate(pub_loader):
                if i % 20 == 0:
                    print(f"Processing public batch {i}", flush=True)

                if len(batch) == 3:
                    ids, imgs, labels = batch
                    memberships = torch.ones(ids.shape[0])
                else:
                    ids, imgs, labels, memberships = batch

                imgs, labels = imgs.to(device), labels.to(device)
                
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                
                true_class_probs = probs[torch.arange(len(labels)), labels].unsqueeze(1)
                loss = F.cross_entropy(logits, labels, reduction='none').unsqueeze(1)
                
                features = torch.cat([logits, true_class_probs, loss], dim=1)
                
                pub_features.append(features.cpu().numpy())
                pub_memberships.append(memberships.numpy())

        X_train = np.concatenate(pub_features, axis=0)
        y_train = np.concatenate(pub_memberships, axis=0)
        
        print(f"Training set shape: {X_train.shape} (Should be N x 11)")

        print("\nTraining Logistic Regression Attack Model...", flush=True)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)


        print("\nExtracting contextual features from private dataset...", flush=True)
        priv_loader = DataLoader(priv_dataset, batch_size=256, shuffle=False)
        
        all_ids = []
        priv_features = []
        
        with torch.no_grad():
            for i, batch in enumerate(priv_loader):
                if i % 20 == 0:
                    print(f"Processing private batch {i}", flush=True)

                if len(batch) == 3:
                    ids, imgs, labels = batch
                else:
                    ids, imgs, labels, _ = batch

                imgs, labels = imgs.to(device), labels.to(device)
                
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                
                true_class_probs = probs[torch.arange(len(labels)), labels].unsqueeze(1)
                loss = F.cross_entropy(logits, labels, reduction='none').unsqueeze(1)
                
                features = torch.cat([logits, true_class_probs, loss], dim=1)
                
                all_ids.extend(ids)
                priv_features.append(features.cpu().numpy())

        X_test = np.concatenate(priv_features, axis=0)
        print(f"Test set shape: {X_test.shape} (Should be 14000 x 11)")

        print("Generating final probability scores...", flush=True)
        scores = clf.predict_proba(X_test)[:, 1]
        
        return all_ids, scores

    # Execute extraction and training
    ids, scores = extract_features_and_train(pub_ds, priv_ds)

    print(f"\nProcessed samples: {len(ids)}", flush=True)

    # Scale strictly to [0, 1] constraints
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Save
    print(f"Saving submission to {OUTPUT_CSV}...", flush=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "score"])
        for i, s in zip(ids, scores):
            writer.writerow([int(i) if torch.is_tensor(i) else i, float(s)])

    print("Saved.", flush=True)

    # Submit
    print("\nSubmitting to server...", flush=True)

    try:
        with open(OUTPUT_CSV, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files={"file": (OUTPUT_CSV.name, f, "application/csv")},
                timeout=(10, 600),
            )

        resp.raise_for_status()
        
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}
            
        print("Successfully submitted!", flush=True)
        print("Server Response:", body)

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print("Submission error:", e)
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)


if __name__ == "__main__":
    run_attack_model()