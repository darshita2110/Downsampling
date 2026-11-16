"""
Stage 2 of Isomap: a basic, understandable example before touching our 7-channel
cube. We use scikit-learn's own Swiss-roll generator (not our satellite-flavoured
synthetic cube) because it is the exact toy problem Tenenbaum et al. (2000) used,
so there is nothing project-specific to second-guess while we are just learning
how Isomap behaves.

Three things are demonstrated:
  (a) Isomap recovers the 2-D "unrolled" layout; PCA does not (same style of
      check as experiments/01_pca_sanity_check.py, now with the textbook object).
  (b) What the k-NN graph Isomap builds actually looks like, in 3-D, before any
      dimensionality reduction happens - this is Step 1 of the algorithm made visible.
  (c) The "short-circuit" failure mode: with n_neighbors too large, a single bad
      edge jumps between two windings of the roll and visibly folds the embedding -
      concrete evidence for the limitation described in documentation/02_isomap.md,
      not just an assertion.

Run:
    python experiments/02_isomap_basic_example.py
Output:
    visualisations/02a_isomap_vs_pca_swissroll.png
    visualisations/02b_neighbor_graph.png
    visualisations/02c_short_circuit_failure.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualisations"
RANDOM_SEED = 42


def part_a_isomap_vs_pca(X, color):
    """The headline comparison: does the method recover the true 1-D position
    along the roll (encoded here as `color`) in a 2-D embedding?"""
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)
    # n_neighbors=10 is a reasonable middle ground for ~1500 points on this roll;
    # we deliberately revisit this choice in part_c below to show what goes wrong.
    iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={})
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="viridis", s=8)
    ax3d.set_title("Original data (3-D)\nthe Swiss roll")

    axp = fig.add_subplot(1, 3, 2)
    axp.scatter(pca[:, 0], pca[:, 1], c=color, cmap="viridis", s=8)
    axp.set_title("PCA (2-D)\nwindings overlap")

    axi = fig.add_subplot(1, 3, 3)
    axi.scatter(iso[:, 0], iso[:, 1], c=color, cmap="viridis", s=8)
    axi.set_title("Isomap (2-D)\nroll correctly 'unrolled'")

    fig.suptitle("Colour = true position along the roll. Isomap should show colour varying\n"
                  "smoothly and monotonically; PCA should show colour looping back on itself.")
    fig.tight_layout()
    path = VIS_DIR / "02a_isomap_vs_pca_swissroll.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return pca, iso


def part_b_neighbor_graph(X, color, n_neighbors=10):
    """Visualise Step 1 of the algorithm directly: the k-NN graph that geodesic
    distances will later be computed over. This is what 'locally flat, globally
    curved' looks like as actual edges, not just a sentence."""
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity")
    graph = graph.tocoo()
    segments = [[X[i], X[j]] for i, j in zip(graph.row, graph.col) if i < j]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Line3DCollection(segments, colors="grey", linewidths=0.3, alpha=0.5))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="viridis", s=10)
    ax.set_title(f"The k-NN graph Isomap builds (k={n_neighbors})\n"
                 f"Geodesic distance = shortest path along these edges")
    fig.tight_layout()
    path = VIS_DIR / "02b_neighbor_graph.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def part_c_short_circuit(X, color):
    """Show the short-circuit failure mode concretely: as n_neighbors grows too
    large, edges start jumping between windings of the roll, and the Isomap
    embedding visibly folds/tears instead of unrolling cleanly."""
    neighbor_values = [6, 10, 30, 80]
    fig, axes = plt.subplots(1, len(neighbor_values), figsize=(4 * len(neighbor_values), 4.2))
    for ax, k in zip(axes, neighbor_values):
        embedding = Isomap(n_neighbors=k, n_components=2).fit_transform(X)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="viridis", s=6)
        ax.set_title(f"n_neighbors = {k}")
    fig.suptitle("Short-circuit failure: too few neighbours can disconnect the graph; too many\n"
                 "let edges jump across windings, folding the embedding (watch colour mix as k grows)")
    fig.tight_layout()
    path = VIS_DIR / "02c_short_circuit_failure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    # n_samples chosen to be large enough to show the effect but fast to run (<5s total).
    X, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=RANDOM_SEED)

    pca, iso = part_a_isomap_vs_pca(X, color)
    part_b_neighbor_graph(X, color, n_neighbors=10)
    part_c_short_circuit(X, color)

    # A simple, numeric companion to the visual check: correlation between the
    # true roll position and each method's first embedding coordinate.  We use
    # Spearman (rank) correlation because Isomap/PCA coordinates can be an
    # arbitrary monotone reparameterisation (e.g. flipped sign) of the true
    # position - we care whether the ORDER along the roll is preserved, not the
    # exact numeric values.
    from scipy.stats import spearmanr
    pca_corr = max(abs(spearmanr(pca[:, i], color).correlation) for i in range(2))
    iso_corr = max(abs(spearmanr(iso[:, i], color).correlation) for i in range(2))
    print(f"\nBest |Spearman correlation| with true roll position:")
    print(f"  PCA (2-D):    {pca_corr:.3f}")
    print(f"  Isomap (2-D): {iso_corr:.3f}")
    print("Isomap should score clearly higher - its coordinates should track the true\n"
          "unrolled position much more closely than PCA's.")


if __name__ == "__main__":
    main()
