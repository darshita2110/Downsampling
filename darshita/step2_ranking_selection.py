"""
STEP 2 - Ranking-based band selection.

The idea in one line:
    give every band a SCORE on its own, sort the bands by score, keep the top K.

Three ways to score a band (three ranking methods):
  1. Variance - how much the band's values change across the image.
                A band where everything looks the same carries little detail.
  2. Entropy  - how much information the band holds (how spread out its histogram is).
                Measured in bits; a band with only a few distinct values has low entropy.
  3. MVPCA    - Maximum-Variance PCA. Run PCA, then score each band by how strongly it
                contributes to the first K principal components (the main patterns).

Run (after step 1):
    python darshita/step2_ranking_selection.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from ranking import entropy_scores, mvpca_scores, top_k, variance_scores   # our own file

# ---------------------------------------------------------------- settings
K = 10                                     # how many bands we are allowed to send
IMAGE_FILE = "data/darshita/Indian_pines_corrected.mat"
OUT_DIR = "darshita/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

image = loadmat(IMAGE_FILE)["indian_pines_corrected"].astype(np.float32)
height, width, n_bands = image.shape

# Turn the cube into a table: one ROW per pixel, one COLUMN per band.
# (145 x 145 x 200)  ->  (21025 pixels x 200 bands). Much easier to work with.
pixels = image.reshape(-1, n_bands)
print(f"Pixel table: {pixels.shape[0]} pixels x {pixels.shape[1]} bands")


all_scores = {
    "Variance": variance_scores(pixels),
    "Entropy": entropy_scores(pixels),
    "MVPCA": mvpca_scores(pixels, K),
}
selected = {name: top_k(scores, K) for name, scores in all_scores.items()}

# A baseline to compare against: just take evenly spaced bands (no thinking at all)
selected["Uniform"] = np.linspace(0, n_bands - 1, K).round().astype(int).tolist()

print(f"\nTop {K} bands chosen by each method:")
for name, bands in selected.items():
    gaps = np.diff(bands)
    print(f"  {name:9s} {bands}   average gap between chosen bands: {gaps.mean():.1f}")

# ---------------------------------------------------- save for step 3
with open(f"{OUT_DIR}/step2_selected_bands.json", "w") as f:
    json.dump({"K": K, "selected": selected}, f, indent=2)
print(f"\nSaved {OUT_DIR}/step2_selected_bands.json")

# ---------------------------------------------------- picture: scores and picks
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, (name, scores) in zip(axes, all_scores.items()):
    ax.plot(scores, color="gray", lw=1.2)
    ax.scatter(selected[name], scores[selected[name]], color="red", zorder=3, label=f"top {K}")
    ax.set_ylabel(f"{name}\nscore")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Band number")
fig.suptitle(f"Ranking methods: score of every band, red = the {K} bands kept")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/step2_scores.png", dpi=120)
print(f"Saved {OUT_DIR}/step2_scores.png")

# ---------------------------------------------------- the weakness of ranking
# Bands are scored ALONE, so if band 30 scores high, bands 29 and 31 (nearly identical)
# also score high, and all of them get picked. Check how alike the chosen bands are:
print("\nHow alike are the chosen bands? (mean |correlation| between them, 1 = copies)")
for name, bands in selected.items():
    c = np.abs(np.corrcoef(pixels[:, bands], rowvar=False))
    redundancy = (c.sum() - K) / (K * (K - 1))               # average of the off-diagonal values
    print(f"  {name:9s} {redundancy:.3f}")
