import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

# CONFIG
ZIP_FILE = "Dataset.zip"          # Path to the downloaded dataset zip
DATASET_DIR = Path("Dataset")      # Unzipped folder
CLEAN_DIR = DATASET_DIR / "clean_targets"
SOURCES_DIR = DATASET_DIR / "watermarked_sources"

LOCAL_OUT_DIR = Path("submission_temp")
LOCAL_OUT_DIR.mkdir(exist_ok=True)

FILE_PATH = Path("submission.zip")

DENOISER = "nlm"          # "nlm" | "median" | "gauss"
ALPHA_FINAL = 1.5

CATEGORIES = [
    ("WM_1", 1, 25), ("WM_2", 26, 50), ("WM_3", 51, 75), ("WM_4", 76, 100),
    ("WM_5", 101, 125), ("WM_6", 126, 150), ("WM_7", 151, 175), ("WM_8", 176, 200),
]

# 1. UNZIP DATASET (if not already extracted)
if not DATASET_DIR.exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"Could not find {ZIP_FILE}. Please download the dataset first.")
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Dataset already extracted.")


def denoise(img_arr_uint8: np.ndarray) -> np.ndarray:
    if DENOISER == "nlm":
        return cv2.fastNlMeansDenoisingColored(img_arr_uint8, None, h=7, hColor=7,
                                                templateWindowSize=7, searchWindowSize=21)
    elif DENOISER == "median":
        return cv2.medianBlur(img_arr_uint8, 3)
    elif DENOISER == "gauss":
        return cv2.GaussianBlur(img_arr_uint8, (3, 3), 0)
    raise ValueError(DENOISER)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def estimate_watermark_pattern(source_dir: Path) -> np.ndarray:
    source_images = sorted(source_dir.glob("*.png"))
    if not source_images:
        raise RuntimeError(f"No source images found in {source_dir}")
    residual_sum = None
    for p in source_images:
        wm_arr = load_rgb(p).astype(np.float32)
        clean_est = denoise(wm_arr.astype(np.uint8)).astype(np.float32)
        residual = wm_arr - clean_est
        residual_sum = residual if residual_sum is None else residual_sum + residual
    return residual_sum / len(source_images)


def forge(target_arr, watermark_pattern, alpha):
    forged = target_arr.astype(np.float32) + alpha * watermark_pattern
    return np.clip(forged, 0, 255).astype(np.uint8)


print(f"Building forgery submission with alpha={ALPHA_FINAL} ...")
total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Processing {source_wm} -> forging onto {target_start}.png .. {target_stop}.png")
    source_dir = SOURCES_DIR / source_wm
    watermark_pattern = estimate_watermark_pattern(source_dir)

    for number in range(target_start, target_stop + 1):
        target_path = CLEAN_DIR / f"{number}.png"
        target_arr = load_rgb(target_path)
        forged_arr = forge(target_arr, watermark_pattern, ALPHA_FINAL)
        Image.fromarray(forged_arr).save(LOCAL_OUT_DIR / target_path.name)
        total_processed += 1

print(f"\nForged {total_processed} images.")
if total_processed != 200:
    print(f"[WARNING] Expected 200, got {total_processed}.")

print(f"Packaging into {FILE_PATH} ...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(LOCAL_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)

print(f"Saved to {FILE_PATH}")