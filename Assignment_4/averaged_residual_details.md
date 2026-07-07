# Black-Box Watermark Forgery via Averaged-Residual Estimation

## Overview
This repository contains our pipeline for a black-box watermark forgery attack. The task is to extract hidden watermarks from an unidentified source dataset (8 distinct, unknown watermarking methods) and inject them into 200 clean target images.

The code pertaining to this markdown is `averaged_residual.py`. 

The result zip file pertaining to this experiment is `result_averaged_residual.zip`

To succeed, the forged images must achieve a high **Detection Strength ($S_{det}$)**, the bit accuracy of the recovered watermark message, while maintaining near-imperceptible visual distortion, measured by **LPIPS ($S_{qlt}$)**.

Our final pipeline achieves its best recorded leaderboard score (**0.4260**) using a deliberately simple two-stage design: **classical denoising** to isolate each image's watermark residual, **signal averaging** across a watermark group to cancel out image content, and a **single globally-tuned injection strength** chosen via an offline LPIPS sweep.

---

## Core Strategy & Rationale

Because all 8 source watermarking methods are unknown, we cannot rely on model-specific extraction (e.g. reversing a known DCT or DWT embedding). We treat the watermark as an additive signal that must be isolated statistically rather than algorithmically reversed.

**The strategy:**
1. Approximate each watermarked image's "clean" version using a fixed, well-understood classical denoiser, no training required.
2. Isolate the true watermark signal by averaging residuals across the 25 images sharing that watermark, canceling out the image-specific content.
3. Inject the isolated signal into target images at a single, empirically-tuned global strength.
4. Validate strength choices offline via an LPIPS sweep before committing a leaderboard submission, rather than tuning against the live scoring endpoint.

We initially explored a more elaborate pipeline (Deep Image Prior extraction, JND-masked injection, a per-image adaptive LPIPS guardrail) but found it underperformed this simpler design (0.40 vs. 0.4260), see "Why Simple Won" below.

---

## Pipeline Architecture

### Phase 1: Extraction via Classical Denoising
**What it does:** For each watermarked source image, we compute `residual = image - denoise(image)` using Non-Local Means (NLM), OpenCV's `fastNlMeansDenoisingColored`.
**Why we chose it:** NLM estimates a locally-adaptive, edge-aware "clean" image rather than uniformly blurring content the way a Gaussian or median filter does. This means less genuine image detail leaks into the residual compared to simpler filters, giving a cleaner starting estimate of the watermark component before averaging. We also evaluated median and Gaussian denoising as alternatives; NLM performed best as the default across most groups in our diagnostics.

### Phase 2: Signal Isolation via Averaging
**What it does:** We compute the per-image residual for all 25 images in a source group (e.g. `WM_1`) and take the pixelwise mean.
**Why we chose it:** As in *Yang et al., "Can Simple Averaging Defeat Modern Watermarks?"*, the watermark is a consistent payload while the underlying image content is highly variable across the 25 samples. Averaging the residuals cancels the random, image-specific component while reinforcing the shared watermark component, yielding a much cleaner estimate than any single residual would.

### Phase 3: Global Injection
**What it does:** Each of the 200 clean targets is forged as `forged = clip(target + alpha * watermark_pattern, 0, 255)`, using the pattern estimated from its mapped source group.
**Why we chose it:** Rather than a per-image adaptive strength, we use one fixed `alpha` across all 8 groups and all 200 images. This was a deliberate simplicity choice: a global constant is far less prone to overfitting to the public 30% leaderboard split than a heavily-tuned per-image or per-group configuration, and our diagnostics (see below) showed that more elaborate tuning did not outperform this baseline in practice.

### Phase 4: Offline LPIPS Validation
**What it does:** Before any leaderboard submission, we sweep `alpha \in {0.5, 1, 1.5, 2, 2.5, 3}` and compute LPIPS locally against the original clean targets to estimate the $S_{qlt}$ each strength would achieve.
**Why we chose it:** Since the leaderboard imposes a 60-minute cooldown between submissions, tuning strength live against the scoring endpoint is slow and wasteful. Sweeping locally lets us pick a strength that stays in a visually safe LPIPS regime across most groups before spending a submission, reserving live submissions for validating genuinely different methods rather than incremental strength tuning.

---

## Why Simple Won

We compared this pipeline against a more architecturally sophisticated initial attempt: Deep Image Prior extraction (a small untrained CNN fit per image via 1000 gradient steps) combined with Sobel-edge JND masking and a per-image adaptive LPIPS guardrail (strength cut by 20% per attempt, up to 10 tries, until LPIPS fell under a strict 0.015 threshold). That pipeline scored **0.40**, identical to a naive alpha-blend baseline, despite being far more computationally expensive (200 independent CNN optimizations vs. one filter call per image).

Our working theory: the strict 0.015 LPIPS guardrail likely suppressed the effective injection strength well below what was needed for strong detection. Since $S_{qlt} = e^{-8 \cdot \text{LPIPS}}$ already yields high quality scores (0.85–0.90) at LPIPS around 0.02, only marginally above the 0.015 cutoff, the guardrail was trading away real detection strength to chase a quality margin with diminishing returns. This is why our final method uses a single, moderately-tuned global strength instead of an aggressive per-image guardrail.

---

## Hyperparameters & Configuration

* **`DENOISER` (`"nlm"`):** Non-Local Means denoising via OpenCV, chosen after comparing against median and Gaussian filtering across all 8 groups.
* **`ALPHA_FINAL` (`1.5`):** A single global injection strength, chosen from an offline LPIPS sweep as a value that stays visually safe across most groups while remaining simple enough to generalize to the held-out 70% private leaderboard split.
* **Per-group tuning (explored, not adopted):** We validated per-group denoiser and strength calibration against a negative control (identical extraction applied to 25 clean, unwatermarked images) and found real extractable signal in only 4 of 8 groups (`WM_1`, `WM_3`, `WM_5`, `WM_6`). A per-group configuration built on these findings did not outperform the simpler global setting on the public leaderboard, so we kept the latter for the final submission.
