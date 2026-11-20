"""
Sanity check (NOT our method yet): run plain PCA on the synthetic cube and see
that it fails to recover the Swiss-roll structure - this is the concrete evidence
for the "why PCA is not enough" discussion in documentation/01_concepts.md.

Run (after 00_make_synthetic_dataset.py):
    python experiments/01_pca_sanity_check.py
Output:
    visualisations/01_pca_vs_truth.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic_v1"
OUT_PATH = ROOT / "visualisations" / "01_pca_vs_truth.png"


def main():
    cube = np.load(DATA_DIR / "cube.npy")             # (H, W, 7)
    true_color = np.load(DATA_DIR / "swiss_roll_color.npy")  # (H, W) - the roll's true unrolled position
    height, width, n_channels = cube.shape

    # Flatten spatial dims: each pixel becomes one 7-D sample. This "pixel as a
    # point in 7-D feature space" view is what every technique in this project
    # (Isomap, Kernel PCA, LLE, etc.) operates on - see documentation/01_concepts.md
    # for why this does NOT by itself use spatial neighbourhoods.
    pixels = cube.reshape(-1, n_channels)

    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(pixels)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(projected[:, 0], projected[:, 1], c=true_color.ravel(), cmap="viridis", s=8)
    axes[0].set_title(f"PCA 2-D projection\n(explained variance: {explained[0]:.2f} + {explained[1]:.2f} = {explained.sum():.2f})")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    axes[1].imshow(true_color, cmap="viridis")
    axes[1].set_title("Ground truth: position along\nthe Swiss roll (per pixel)")
    axes[1].axis("off")

    fig.suptitle("If PCA had recovered the manifold, colour (true position) would vary smoothly\n"
                  "across the PCA scatter. Instead, the roll's turns overlap in the 2-D projection.")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")
    print(f"PCA explained variance ratio (first 2 components): {explained}")
    print("Look at the saved figure: colours (true roll position) should mix/overlap in the PCA "
          "scatter if PCA is failing to unroll the manifold, as expected.")


if __name__ == "__main__":
    main()
