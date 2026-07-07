import os
import zipfile
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import lpips
import wandb
import torch.nn.functional as F


# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("Dataset")
TEMP_OUT_DIR = Path("submission_temp")
FILE_PATH = "submission_jnd_v2.zip"

# Hyperparameters for the Deep Image Prior
DIP_ITERATIONS = 1000  # Number of optimization steps per image
LR = 0.001             # Learning rate for the DIP network

# >>> START OF CHANGE: Method-Specific Hyperparameters
# Instead of a global alpha of 3.0, we use a dictionary. 
# Bumping the default to 10.0 to trade your massive LPIPS budget for Bit Accuracy.
ALPHA_CONFIG = {
    "WM_1": 5, "WM_2": 5, "WM_3": 5, "WM_4": 5,
    "WM_5": 5, "WM_6": 5, "WM_7": 5, "WM_8": 5
}
# <<< END OF CHANGE

# Initialize Weights & Biases
wandb.init(
    project="watermark-forgery-task",
    name=f"DIP_Iter{DIP_ITERATIONS}_A10_JNDFloor",
    config={
        "iterations": DIP_ITERATIONS,
        "learning_rate": LR,
        "alpha_config": ALPHA_CONFIG,
        "architecture": "Mini-DIP-CNN"
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading LPIPS model for local Sqlt evaluation...")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

if not DATASET_DIR.exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset.")
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")

TEMP_OUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. DEEP IMAGE PRIOR ARCHITECTURE
# ==========================================
class MiniDIP(nn.Module):
    """
    A lightweight, untrained Encoder-Decoder CNN. 
    It naturally learns low-frequency image structures (content) faster than 
    high-frequency noise (watermarks).
    """
    def __init__(self):
        super().__init__()
        # Encoder: Downsamples the image to capture broad structures
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.LeakyReLU(0.2)
        )
        # Decoder: Upsamples back to original resolution
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 2. CORE EXTRACTION & SCORING FUNCTIONS
# ==========================================
def extract_watermark_dip(image_path, iterations, lr, device):
    """Extracts watermark by fitting an untrained CNN to the image."""
    original_pil = Image.open(image_path).convert("RGB")
    target_tensor = transforms.ToTensor()(original_pil).unsqueeze(0).to(device)
    
    # >>> START OF CHANGE: DIP Stagnation Restart
    # Helper to create fresh weights if we get stuck
    def get_fresh_model():
        m = MiniDIP().to(device)
        o = optim.Adam(m.parameters(), lr=lr)
        return m, o
        
    model, optimizer = get_fresh_model()
    # <<< END OF CHANGE
    
    loss_fn = nn.MSELoss()
    z = (torch.rand_like(target_tensor).to(device) * 0.1).requires_grad_(False)
    
    prev_100_loss = float('inf') # Added to track stagnation
    
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(z)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()

        # >>> START OF CHANGE: Stagnation Check
        if i > 0 and i % 100 == 0:
            current_loss = loss.item()
            # If the loss hasn't changed by at least 1e-5 over 100 steps, we are stuck.
            if abs(prev_100_loss - current_loss) < 1e-5:
                print(f"      [Step {i:04d}/{iterations}] Stagnated at {current_loss:.6f}. Restarting model...")
                model, optimizer = get_fresh_model()
                z = (torch.rand_like(target_tensor).to(device) * 0.1).requires_grad_(False)
                prev_100_loss = float('inf')
                continue # Skip the logging for this step and start over
                
            prev_100_loss = current_loss
            
            # Stream to W&B
            wandb.log({"dip_extraction_loss": current_loss, "dip_step": i})
            print(f"      [Step {i:04d}/{iterations}] MSE Loss: {current_loss:.6f}")
        # <<< END OF CHANGE

    with torch.no_grad():
        clean_estimate = model(z)
        watermark_residual = target_tensor - clean_estimate
        
    return watermark_residual.squeeze(0).cpu().numpy()

def calculate_sqlt(clean_img_tensor, forged_img_tensor):
    clean_scaled = (clean_img_tensor * 2.0) - 1.0
    forged_scaled = (forged_img_tensor * 2.0) - 1.0
    
    with torch.no_grad():
        lpips_distance = loss_fn_alex(clean_scaled.to(device), forged_scaled.to(device)).item()
        
    sqlt = math.exp(-8 * lpips_distance)
    return lpips_distance, sqlt


def get_jnd_mask(img_tensor, device):
    """
    Creates a perceptual mask based on image textures/edges using a Sobel filter.
    Returns a tensor where smooth areas approach 0.0 and textured areas approach 1.0.
    """
    # 1. Convert to grayscale for edge detection
    gray = 0.2989 * img_tensor[:, 0:1, :, :] + 0.5870 * img_tensor[:, 1:2, :, :] + 0.1140 * img_tensor[:, 2:3, :, :]
    
    # 2. Define Sobel kernels for horizontal and vertical edges
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    
    # 3. Apply convolutions to find edges
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    
    # 4. Calculate gradient magnitude
    magnitude = torch.sqrt(edge_x**2 + edge_y**2)
    
    # 5. Normalize to [0, 1] range to create the mask multiplier
    mask = magnitude / (magnitude.max() + 1e-8)
    
    # 6. Smooth the mask slightly so watermark injection blends naturally
    mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
    
    # >>> START OF CHANGE: JND Floor
    # Ensure smooth areas (like skies) still receive at least 10% of the watermark 
    # to protect frequency-domain / global payloads from being completely erased.
    mask = 0.1 + (0.9 * mask)
    # <<< END OF CHANGE
    
    # Expand back to 3 channels to match the RGB image
    return mask.repeat(1, 3, 1, 1)

# ==========================================
# 3. THE ATTACK LOOP
# ==========================================
CATEGORIES = [
    ("WM_1", 1, 25), ("WM_2", 26, 50), ("WM_3", 51, 75), ("WM_4", 76, 100),
    ("WM_5", 101, 125), ("WM_6", 126, 150), ("WM_7", 151, 175), ("WM_8", 176, 200)
]

total_processed = 0
global_sqlt_scores = []

print("\nExecuting Deep Image Prior Forgery Attack...")

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"\nProcessing {source_wm} ...")
    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_images = list(source_dir.glob("*.png"))

    if not source_images:
        continue
        
    # >>> START OF CHANGE: Apply specific Alpha
    current_alpha = ALPHA_CONFIG[source_wm]
    print(f"  > Using injection multiplier (ALPHA): {current_alpha}")
    # <<< END OF CHANGE

    # --- PHASE 1: EXTRACT & AVERAGE ---
    extracted_watermarks = []
    
    for idx, source_path in enumerate(source_images):
        print(f"  > Optimizing DIP for image {idx+1}/{len(source_images)}...")
        residual = extract_watermark_dip(source_path, DIP_ITERATIONS, LR, device)
        extracted_watermarks.append(residual)
        
    w_batch = np.mean(extracted_watermarks, axis=0)
    w_batch_tensor = torch.from_numpy(w_batch).float()

# --- PHASE 2: INJECT & VALIDATE ---
    target_dir = DATASET_DIR / "clean_targets"
    batch_sqlt = []
    
    # >>> START OF CHANGE: Define maximum acceptable LPIPS distance
    LPIPS_THRESHOLD = 0.015  # Tweak this if you need more/less visual quality
    # <<< END OF CHANGE
    
    print(f"  > Injecting extracted signature into target images...")
    for number in range(target_start, target_stop + 1):
        target_path = target_dir / f"{number}.png"
        
        target_pil = Image.open(target_path).convert("RGB")
        target_tensor = transforms.ToTensor()(target_pil).to(device)

        jnd_mask = get_jnd_mask(target_tensor.unsqueeze(0), device).squeeze(0)

        w_batch_gpu = w_batch_tensor.to(device)

        # >>> START OF CHANGE: Adaptive LPIPS Guardrail Loop
        dynamic_alpha = current_alpha
        
        for attempt in range(10): # Try up to 10 times to get under the threshold
            # 1. Inject noise using the dynamic alpha
            forged_tensor = target_tensor + (dynamic_alpha * w_batch_gpu * jnd_mask)
            
            # 2. Clamp to valid image bounds (this causes the clipping distortion if alpha is too high)
            forged_tensor = torch.clamp(forged_tensor, 0, 1)
            
            # 3. Check the LPIPS penalty
            dist, sqlt = calculate_sqlt(target_tensor.unsqueeze(0), forged_tensor.unsqueeze(0))
            
            # 4. Guardrail check
            if dist <= LPIPS_THRESHOLD:
                if attempt > 0:
                     print(f"      [Image {number}] Alpha reduced to {dynamic_alpha:.4f} to pass guardrail.")
                break # We are within safe visual limits!
            else:
                # Distorted too much! Dial back the strength by 20% and try again
                dynamic_alpha *= 0.8
        # <<< END OF CHANGE
        
        # Move back to CPU for packaging
        forged_tensor = forged_tensor.cpu()
        batch_sqlt.append(sqlt)
        
        forged_img_np = (forged_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        out_path = TEMP_OUT_DIR / target_path.name
        Image.fromarray(forged_img_np).save(out_path)
        total_processed += 1
        
        wandb.log({
            "watermark_method": source_wm,
            "target_image_id": number,
            "lpips_distance": dist,
            "sqlt_score": sqlt,
            "final_alpha_used": dynamic_alpha, # >>> Added so you can track the nerfed alpha in W&B
            "forged_image": wandb.Image(forged_img_np, caption=f"Sqlt: {sqlt:.4f}, Alpha: {dynamic_alpha:.4f}")
        })
        
    avg_batch_sqlt = np.mean(batch_sqlt)
    global_sqlt_scores.extend(batch_sqlt)
    
    wandb.log({f"avg_sqlt_{source_wm}": avg_batch_sqlt})
    print(f"  > Forgery complete. Estimated Sqlt for {source_wm}: {avg_batch_sqlt:.4f}")

# ==========================================
# 4. FINAL SUMMARY & PACKAGING
# ==========================================
final_global_sqlt = np.mean(global_sqlt_scores)
print(f"\nSuccessfully forged {total_processed} images.")
print(f"GLOBAL ESTIMATED S_qlt: {final_global_sqlt:.4f}")

wandb.log({"final_global_sqlt": final_global_sqlt})
wandb.finish()

print(f"Packaging images into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in TEMP_OUT_DIR.glob("*.png"):
        zipf.write(img_path, arcname=img_path.name)

print("Done. Check your W&B dashboard!")