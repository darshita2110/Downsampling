"""
STEP 3 - How good are the bands each ranking method chose?

The downlink story:
  - the satellite sends ONLY the K chosen bands
  - on the ground we REBUILD all 200 bands from those K
  - the closer the rebuilt image is to the original, the better the choice

How we rebuild: linear regression. Each missing band is predicted as a weighted sum of the
K sent bands. The weights are learned on 10% of the pixels (a "calibration" set the ground
station has) and the result is scored on the other 90%.

Scores:
  PSNR (dB)     - overall pixel accuracy. Higher is better. +3 dB = half the error.
  SAM (degrees) - how much each pixel's spectrum curve is bent out of shape. Lower is better.

Run (after step 2):
    python darshita/step3_evaluate_ranking.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from ranking import entropy_scores, mvpca_scores, top_k, variance_scores   # our own file

IMAGE_FILE = "data/darshita/Indian_pines_corrected.mat"
OUT_DIR = "darshita/outputs"

image = loadmat(IMAGE_FILE)["indian_pines_corrected"].astype(np.float32)
height, width, n_bands = image.shape
raw_pixels = image.reshape(-1, n_bands)

# Scale every band to 0..1 so that every band counts equally in the error.
low, high = raw_pixels.min(axis=0), raw_pixels.max(axis=0)
pixels = (raw_pixels - low) / (high - low)

# Split pixels: 10% to learn the rebuild weights, 90% to test.
rng = np.random.default_rng(0)                 # fixed seed -> same split every run
order = rng.permutation(len(pixels))
calib, test = order[: len(pixels) // 10], order[len(pixels) // 10:]


def rebuild(bands):
    """Predict all bands from the chosen ones with least-squares linear regression."""
    X = np.c_[pixels[:, bands], np.ones(len(pixels))]       # chosen bands + a constant column
    weights, *_ = np.linalg.lstsq(X[calib], pixels[calib], rcond=None)
    return np.clip(X @ weights, 0, 1)                       # rebuilt pixel table


def psnr(original, rebuilt):
    mse = np.mean((original - rebuilt) ** 2)
    return 10 * np.log10(1.0 / mse)                         # peak value is 1 after scaling


def sam_degrees(original, rebuilt):
    dot = np.sum(original * rebuilt, axis=1)
    norms = np.linalg.norm(original, axis=1) * np.linalg.norm(rebuilt, axis=1)
    angles = np.arccos(np.clip(dot / np.maximum(norms, 1e-12), -1, 1))
    return np.degrees(angles).mean()


# ------------------------------------------------ part A: the K bands from step 2
with open(f"{OUT_DIR}/step2_selected_bands.json") as f:
    saved = json.load(f)
K, selected = saved["K"], saved["selected"]

print(f"Sending {K} of {n_bands} bands = {100 * K / n_bands:.0f}% of the data\n")
print(f"  {'Method':9s} {'PSNR (dB)':>10s} {'SAM (deg)':>10s}")
rebuilt_images = {}
for name, bands in selected.items():
    rebuilt = rebuild(bands)
    rebuilt_images[name] = rebuilt
    print(f"  {name:9s} {psnr(pixels[test], rebuilt[test]):10.2f} {sam_degrees(pixels[test], rebuilt[test]):10.2f}")

# Picture: one band that was NOT sent, original vs rebuilt by each method, plus error
band = 99
fig, axes = plt.subplots(2, len(selected) + 1, figsize=(3 * (len(selected) + 1), 6))
axes[0, 0].imshow(pixels[:, band].reshape(height, width), cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title(f"Original band {band}")
axes[1, 0].axis("off")
for i, (name, rebuilt) in enumerate(rebuilt_images.items(), start=1):
    axes[0, i].imshow(rebuilt[:, band].reshape(height, width), cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"Rebuilt: {name}")
    error = np.abs(pixels[:, band] - rebuilt[:, band]).reshape(height, width)
    axes[1, i].imshow(error, cmap="magma", vmin=0, vmax=0.3)
    axes[1, i].set_title("Error (brighter = worse)")
for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/step3_rebuilt_band.png", dpi=120)
print(f"\nSaved {OUT_DIR}/step3_rebuilt_band.png")

# ------------------------------------------------ part B: what if we send more bands?
k_values = [5, 10, 15, 20, 25, 30]
results = {name: [] for name in ["Variance", "Entropy", "MVPCA", "Uniform"]}
var_s, ent_s = variance_scores(raw_pixels), entropy_scores(raw_pixels)  # these don't depend on k
for k in k_values:
    picks = {
        "Variance": top_k(var_s, k),
        "Entropy": top_k(ent_s, k),
        "MVPCA": top_k(mvpca_scores(raw_pixels, k), k),
        "Uniform": np.linspace(0, n_bands - 1, k).round().astype(int).tolist(),
    }
    for name, bands in picks.items():
        results[name].append(psnr(pixels[test], rebuild(bands)[test]))

plt.figure(figsize=(7, 4.5))
for name, values in results.items():
    plt.plot(k_values, values, marker="o", lw=2, ls="--" if name == "Uniform" else "-", label=name)
plt.xlabel("Number of bands sent (K)")
plt.ylabel("PSNR of rebuilt image (dB)  - higher is better")
plt.title("Ranking methods vs simply taking evenly spaced bands")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/step3_psnr_vs_k.png", dpi=120)
print(f"Saved {OUT_DIR}/step3_psnr_vs_k.png")

print("\nPSNR (dB) for each K:")
print(f"  {'K':>3s} " + " ".join(f"{n:>9s}" for n in results))
for i, k in enumerate(k_values):
    print(f"  {k:>3d} " + " ".join(f"{results[n][i]:9.2f}" for n in results))
