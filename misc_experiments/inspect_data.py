import torch
import numpy as np
from pathlib import Path
from collections import Counter
import hashlib

# config
BASE = Path(__file__).parent / "pt_files"
PUB_PATH = BASE / "pub.pt"

# Minimal class definition just to load the file
class TaskDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.ids, self.imgs, self.labels = [], [], []
    def __len__(self): return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self):
        super().__init__()
        self.membership = []

def get_image_hash(img_tensor):
    # Convert tensor to numpy, make it contiguous, and hash it to find exact duplicates
    return hashlib.md5(img_tensor.numpy().tobytes()).hexdigest()

def inspect_dataset():
    print(f"Loading {PUB_PATH}...")
    try:
        pub_ds = torch.load(PUB_PATH, weights_only=False)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    total_samples = len(pub_ds.ids)
    print(f"\n--- BASIC STATISTICS ---")
    print(f"Total Samples: {total_samples}")
    
    # Membership Balance
    members = sum(pub_ds.membership)
    non_members = total_samples - members
    print(f"Members (1): {members} ({members/total_samples*100:.1f}%)")
    print(f"Non-Members (0): {non_members} ({non_members/total_samples*100:.1f}%)")

    # Class Balance
    print("\n--- CLASS DISTRIBUTION ---")
    member_labels = [pub_ds.labels[i] for i in range(total_samples) if pub_ds.membership[i] == 1]
    non_member_labels = [pub_ds.labels[i] for i in range(total_samples) if pub_ds.membership[i] == 0]
    
    mem_counts = Counter(member_labels)
    non_mem_counts = Counter(non_member_labels)
    
    print("Class | Member Count | Non-Member Count")
    print("-" * 40)
    for c in range(9): # Assuming 9 classes based on our FC layer
        print(f"{c:5d} | {mem_counts.get(c, 0):12d} | {non_mem_counts.get(c, 0):16d}")

    # Pixel Statistics Check (Are members brighter/darker?)
    print("\n--- RAW TENSOR STATISTICS ---")
    print(f"Image Shape: {pub_ds.imgs[0].shape}")
    print(f"Data Type: {pub_ds.imgs[0].dtype}")
    
    # Calculate average pixel values
    mem_imgs = torch.stack([pub_ds.imgs[i] for i in range(total_samples) if pub_ds.membership[i] == 1]).float()
    non_mem_imgs = torch.stack([pub_ds.imgs[i] for i in range(total_samples) if pub_ds.membership[i] == 0]).float()
    
    print(f"Members Mean Pixel: {mem_imgs.mean().item():.4f}, Std: {mem_imgs.std().item():.4f}")
    print(f"Non-Mem Mean Pixel: {non_mem_imgs.mean().item():.4f}, Std: {non_mem_imgs.std().item():.4f}")

    # Duplicate Image Check (The "Gotcha" Test)
    print("\n--- EXACT DUPLICATE CHECK ---")
    hashes = {}
    duplicates = 0
    cross_over_duplicates = 0 # Image is in both Member and Non-Member sets
    
    print("Hashing images... (this might take a few seconds)")
    for i in range(total_samples):
        img_hash = get_image_hash(pub_ds.imgs[i])
        mem_status = pub_ds.membership[i]
        
        if img_hash in hashes:
            duplicates += 1
            if hashes[img_hash] != mem_status:
                cross_over_duplicates += 1
        else:
            hashes[img_hash] = mem_status

    print(f"Total Exact Duplicate Images: {duplicates}")
    if duplicates > 0:
        print(f"Duplicates crossing the Member/Non-Member boundary: {cross_over_duplicates}")
        if cross_over_duplicates > 0:
            print("WARNING: You have images that appear in BOTH the training set and the test set.")
            print("This explains why the model cannot distinguish them—they are literally the same images.")
    else:
        print("No exact duplicates found. The images are unique.")

if __name__ == "__main__":
    inspect_dataset()