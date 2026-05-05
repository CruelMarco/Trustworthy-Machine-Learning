import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
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

def run_sanity_check():
    print("Loading public dataset and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    
    sample_img = pub_ds.imgs[0]
    print(f"Image Tensor Type: {sample_img.dtype}, Min: {sample_img.min().item()}, Max: {sample_img.max().item()}")
    
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

    pub_loader = DataLoader(pub_ds, batch_size=256, shuffle=False)
    
    member_correct, member_total, member_loss = 0, 0, 0.0
    non_member_correct, non_member_total, non_member_loss = 0, 0, 0.0

    print("Evaluating Model Accuracy...")
    with torch.no_grad():
        for _, imgs, labels, memberships in tqdm(pub_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, reduction='none')
            preds = logits.argmax(dim=1)
            
            is_correct = (preds == labels)
            
            # Split stats by membership
            for i in range(len(memberships)):
                if memberships[i] == 1:
                    member_total += 1
                    member_correct += is_correct[i].item()
                    member_loss += loss[i].item()
                else:
                    non_member_total += 1
                    non_member_correct += is_correct[i].item()
                    non_member_loss += loss[i].item()

    print("\n--- SANITY CHECK RESULTS ---")
    if member_total > 0:
        print(f"MEMBERS     | Accuracy: {member_correct/member_total*100:.2f}% | Avg Loss: {member_loss/member_total:.4f}")
    if non_member_total > 0:
        print(f"NON-MEMBERS | Accuracy: {non_member_correct/non_member_total*100:.2f}% | Avg Loss: {non_member_loss/non_member_total:.4f}")

if __name__ == "__main__":
    run_sanity_check()