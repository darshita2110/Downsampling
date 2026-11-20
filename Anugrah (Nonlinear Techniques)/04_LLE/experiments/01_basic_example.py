"""
Stage 2 of LLE: same Swiss roll as Technique 1-3. Tests Stage 1's specific
claim directly rather than assuming it: LLE has no chained-distance graph, so
does it actually behave differently from Isomap as n_neighbors grows, or does
it just fail in some other way at large k?

Also demonstrates the `reg` hyperparameter's role: with k > n_features (true
for our actual 7-channel data later, and deliberately forced here too by
testing k values well above the roll's 3 ambient dimensions), LLE's local
least-squares step becomes ill-conditioned without regularisation.

Run (from the project root, Anugrah (Nonlinear Techniques)/):
    python 04_LLE/experiments/01_basic_example.py
Output:
    visualisations/01a_lle_vs_isomap_swissroll.png
    visualisations/01b_lle_k_sweep.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, LocallyLinearEmbedding

TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
VIS_DIR = TECHNIQUE_DIR / "visualisations"
RANDOM_SEED = 42


def best_corr(embedding, color):
    return max(abs(spearmanr(embedding[:, i], color).correlation) for i in range(embedding.shape[1]))


def part_a_lle_vs_isomap(X, color):
    iso_embedding = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method="standard", random_state=RANDOM_SEED)
    lle_embedding = lle.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(iso_embedding[:, 0], iso_embedding[:, 1], c=color, cmap="viridis", s=8)
    axes[0].set_title("Isomap (Technique 1)")
    axes[1].scatter(lle_embedding[:, 0], lle_embedding[:, 1], c=color, cmap="viridis", s=8)
    axes[1].set_title(f"LLE (Technique 4)\nn_neighbors=10, reconstruction_error_={lle.reconstruction_error_:.2e}")
    fig.suptitle("Same data, n_neighbors=10. LLE never chains distances into a graph -\n"
                 "check whether it still unrolls the roll, and how.")
    fig.tight_layout()
    path = VIS_DIR / "01a_lle_vs_isomap_swissroll.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    corr_iso = best_corr(iso_embedding, color)
    corr_lle = best_corr(lle_embedding, color)
    print(f"Best |Spearman correlation| with true roll position:")
    print(f"  Isomap: {corr_iso:.3f}")
    print(f"  LLE:    {corr_lle:.3f}")


def part_b_k_sweep(X, color):
    """Stage 1's claim under direct test: does LLE show Isomap's short-circuit
    curling-into-a-loop failure as k grows, or something qualitatively
    different (e.g. just noisier/less structured, without curling back)?"""
    neighbor_values = [6, 10, 30, 80]
    fig, axes = plt.subplots(1, len(neighbor_values), figsize=(4 * len(neighbor_values), 4.2))
    correlations = []
    for ax, k in zip(axes, neighbor_values):
        lle = LocallyLinearEmbedding(n_neighbors=k, n_components=2, method="standard",
                                      random_state=RANDOM_SEED, reg=1e-3)
        embedding = lle.fit_transform(X)
        corr = best_corr(embedding, color)
        correlations.append(corr)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="viridis", s=6)
        ax.set_title(f"n_neighbors={k}\ncorr={corr:.2f}")
    fig.suptitle("LLE as n_neighbors grows: compare the SHAPE of any failure here against\n"
                 "Isomap's curling-into-a-loop failure (01_Isomap/visualisations/02c)")
    fig.tight_layout()
    path = VIS_DIR / "01b_lle_k_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    for k, c in zip(neighbor_values, correlations):
        print(f"  n_neighbors={k}: correlation={c:.3f}")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    X, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=RANDOM_SEED)
    part_a_lle_vs_isomap(X, color)
    part_b_k_sweep(X, color)


if __name__ == "__main__":
    main()
