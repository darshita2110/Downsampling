"""
Stage 2 of Kernel PCA: same Swiss roll used for Technique 1 and 2, now testing
the specific question Stage 1's maths raised rather than assuming an answer -
does RBF Kernel PCA, with its smooth similarity and no neighbourhood graph,
actually unroll the Swiss roll the way Isomap does, or does its lack of a
"walk around the surface" mechanism show up as a real difference?

Also sweeps gamma to show its role directly (too small -> looks like linear
PCA; too large -> only near-duplicates look similar), the RBF analogue of
Technique 1's n_neighbors sweep and Technique 2's n_init sweep.

Run (from the project root, Anugrah (Nonlinear Techniques)/):
    python 03_KernelPCA/experiments/01_basic_example.py
Output:
    visualisations/01a_kpca_vs_isomap_swissroll.png
    visualisations/01b_gamma_sweep.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
VIS_DIR = TECHNIQUE_DIR / "visualisations"
RANDOM_SEED = 42


def best_corr(embedding, color):
    return max(abs(spearmanr(embedding[:, i], color).correlation) for i in range(embedding.shape[1]))


def part_a_kpca_vs_isomap(X, color):
    iso_embedding = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    # gamma chosen via the sweep below (part b identifies a reasonable value first in practice;
    # here we use a value already known from that sweep to keep this function self-contained)
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.01, random_state=RANDOM_SEED)
    kpca_embedding = kpca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(iso_embedding[:, 0], iso_embedding[:, 1], c=color, cmap="viridis", s=8)
    axes[0].set_title("Isomap (Technique 1)")
    axes[1].scatter(kpca_embedding[:, 0], kpca_embedding[:, 1], c=color, cmap="viridis", s=8)
    axes[1].set_title("RBF Kernel PCA (Technique 3)\ngamma=0.01")
    fig.suptitle("Same data, same n_components. Colour = true position along the roll.\n"
                  "Isomap 'walks around' the roll via its graph; Kernel PCA has no such mechanism.")
    fig.tight_layout()
    path = VIS_DIR / "01a_kpca_vs_isomap_swissroll.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    corr_iso = best_corr(iso_embedding, color)
    corr_kpca = best_corr(kpca_embedding, color)
    print(f"Best |Spearman correlation| with true roll position:")
    print(f"  Isomap:          {corr_iso:.3f}")
    print(f"  RBF Kernel PCA:  {corr_kpca:.3f}")


def part_b_gamma_sweep(X, color):
    gammas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    fig, axes = plt.subplots(1, len(gammas), figsize=(4 * len(gammas), 4.2))
    correlations = []
    for ax, gamma in zip(axes, gammas):
        kpca = KernelPCA(n_components=2, kernel="rbf", gamma=gamma, random_state=RANDOM_SEED)
        embedding = kpca.fit_transform(X)
        corr = best_corr(embedding, color)
        correlations.append(corr)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="viridis", s=6)
        ax.set_title(f"gamma={gamma}\ncorr={corr:.2f}")
    fig.suptitle("gamma sweep: too small looks like linear PCA (left); too large, only\n"
                 "near-duplicates look similar and structure collapses (right)")
    fig.tight_layout()
    path = VIS_DIR / "01b_gamma_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    for g, c in zip(gammas, correlations):
        print(f"  gamma={g}: correlation={c:.3f}")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    X, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=RANDOM_SEED)
    part_a_kpca_vs_isomap(X, color)
    part_b_gamma_sweep(X, color)


if __name__ == "__main__":
    main()
