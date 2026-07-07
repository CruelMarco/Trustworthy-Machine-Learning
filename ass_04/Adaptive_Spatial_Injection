# Black-Box Watermark Forgery via Adaptive Spatial Injection

## Overview
This repository contains a robust, automated pipeline designed to execute a black-box watermark forgery attack. The goal of this task is to extract hidden watermarks from an unidentified source dataset (8 distinct, unknown watermarking models) and inject them into 200 clean target images. 

To succeed, the forged images must achieve a high **False Detection Rate ($S_{det}$)** while maintaining near-imperceptible visual distortion, measured by **LPIPS ($S_{qlt}$)**. 

Our pipeline achieved a highly stable score by bridging the gap between raw spatial signal extraction and human perceptual limits using **Deep Image Prior (DIP)**, **Signal Averaging**, **Just Noticeable Difference (JND) Masking**, and an **Adaptive LPIPS Guardrail**.

---

## Core Strategy & Rationale

Because the 8 source watermarking methods are completely unknown (black-box), we cannot rely on model-specific extraction techniques (like reversing a known DCT or DWT embedding). We must treat the watermark as a universal additive signal. 

**The Strategy:**
1. Extract the high-frequency noise from the source images without needing a pre-trained network.
2. Isolate the true watermark signal by canceling out the underlying image content.
3. Inject the signal into target images as aggressively as possible to trigger the detector.
4. Mathematically guarantee that the injection never crosses the threshold of visible human distortion.

---

## Pipeline Architecture

### Phase 1: Extraction via Deep Image Prior (DIP)
**What it does:** We utilize a lightweight, untrained Convolutional Neural Network (CNN) called `MiniDIP` to extract the watermark. 
**Why we chose it:** Deep Image Prior operates on the principle that the architecture of a CNN naturally resonates with the low-level statistics of natural images. When we try to fit this untrained network to a watermarked image, it learns to reconstruct the smooth, natural shapes of the image *much faster* than it learns the chaotic, high-frequency noise of the watermark. 
* By stopping the optimization early (1000 iterations), the network outputs a "clean" estimate of the image.
* Subtracting this clean estimate from the original watermarked image isolates the high-frequency residual (the watermark proxy).

### Phase 2: Signal Isolation via Simple Averaging
**What it does:** We extract the residual noise from all 25 images in a specific source category (e.g., `WM_1`) and compute the spatial mean across the batch.
**Why we chose it:** As demonstrated in *Yang et al., "Can Simple Averaging Defeat Modern Watermarks?"*, a watermark is a consistent payload, while the underlying image content is highly variable. By averaging the residuals of 25 different images containing the same watermark, the random image content cancels itself out, leaving behind a highly concentrated, isolated watermark signature.

### Phase 3: Perceptual Hiding (JND Masking)
**What it does:** Before injecting the payload, the script calculates a "Just Noticeable Difference" (JND) mask using a Sobel edge-detection filter. This creates a topographical map of the target image where flat areas approach `0.0` and highly textured areas approach `1.0`.
**Why we chose it:** The scoring metric heavily penalizes noise in smooth areas (like a clear blue sky) because the human eye easily detects grain there. By multiplying our watermark payload by the JND mask, we hide the majority of the injected noise inside the natural textures of the image (like grass or fabric) where it is perceptually invisible.
* **The 10% Floor:** We apply a mathematical floor to the mask (`mask = 0.1 + 0.9 * original_mask`). If a watermarking method operates globally in the frequency domain (like DCT), completely zeroing out the noise in smooth areas would destroy the payload. The 10% floor ensures a faint signal survives everywhere.

### Phase 4: The Adaptive LPIPS Guardrail
**What it does:** Instead of blindly injecting the watermark at a static strength (`ALPHA = 5.0`) and hoping for the best, the injection runs inside an adaptive validation loop. 
**Why we chose it:** Brute-force spatial injection often pushes pixel values beyond the valid `[0, 1]` RGB bounds. When PyTorch clamps these out-of-bound pixels, it creates severe, blocky color artifacts that instantly destroy the visual quality score ($S_{qlt}$) and the watermark payload. 

**How it works:**
1. The script attempts to inject the payload at the maximum configured `ALPHA`.
2. It immediately evaluates the `LPIPS` distance locally.
3. If the distortion exceeds our strict safety threshold (`0.015`), it dynamically reduces the `ALPHA` by 20% and tries again.
4. It repeats this until the payload safely slides under the perceptual radar.

This guardrail acts as a fail-safe, mathematically ensuring that the pipeline squeezes the absolute maximum bit-accuracy out of the image without ever crossing the threshold into destructive clipping.

---

## Hyperparameters & Configuration

* **`DIP_ITERATIONS` (1000):** The sweet spot for our `MiniDIP` network. Enough steps to learn the broad image structures, but not enough to begin memorizing the high-frequency watermark noise.
* **`LPIPS_THRESHOLD` (0.015):** The maximum allowable visual distortion. This maps to an estimated $S_{qlt}$ of `~0.88`, ensuring our baseline quality score remains elite.
* **`ALPHA_CONFIG` (Base 5.0):** A method-specific dictionary allowing us to tune the baseline injection strength for each of the 8 unknown watermarking models independently based on Weights & Biases telemetry.

