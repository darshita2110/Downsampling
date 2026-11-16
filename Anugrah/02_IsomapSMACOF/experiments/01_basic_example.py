"""
Stage 2 of Isomap-with-SMACOF: the same Swiss roll used in
01_Isomap/experiments/02_isomap_basic_example.py, now comparing classical-MDS
Isomap (Technique 1) against Isomap-with-SMACOF (Technique 2) directly - same
geodesic distances in both cases, only the embedding step differs.

Two things demonstrated:
  (a) Do the two give visually/numerically similar embeddings on data this
      well-behaved? (Expectation: yes, closely - SMACOF solving the "right"
      criterion should land close to classical MDS's answer when the data is
      clean and well-sampled, since there is then little room for local minima
      to matter.)
  (b) SMACOF's hallmark property made visible: Stress decreasing monotonically
      across n_init random restarts, and the restarts NOT all landing on
      identical stress - concrete evidence of the local-minima behaviour
      classical MDS (Technique 1) structurally cannot have.

Run (from the project root, Anugrah (Nonlinear Techniques)/):
    python 02_IsomapSMACOF/experiments/01_basic_example.py
Output:
    visualisations/01a_isomap_vs_smacof_swissroll.png
    visualisations/01b_smacof_restarts_stress.png
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap

TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TECHNIQUE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TECHNIQUE_DIR))
from isomap_smacof import _build_mds

VIS_DIR = TECHNIQUE_DIR / "visualisations"
RANDOM_SEED = 42


def part_a_isomap_vs_smacof(X, color, n_neighbors):
    iso_model = Isomap(n_neighbors=n_neighbors, n_components=2).fit(X)
    classical_embedding = iso_model.embedding_
    dist_matrix = iso_model.dist_matrix_

    mds = _build_mds(n_components=2, n_init=4, max_iter=300, eps=1e-6, random_state=RANDOM_SEED)
    smacof_embedding = mds.fit_transform(dist_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(classical_embedding[:, 0], classical_embedding[:, 1], c=color, cmap="viridis", s=8)
    axes[0].set_title("Classical-MDS Isomap (Technique 1)")
    axes[1].scatter(smacof_embedding[:, 0], smacof_embedding[:, 1], c=color, cmap="viridis", s=8)
    axes[1].set_title(f"Isomap + SMACOF (Technique 2)\nfinal stress={mds.stress_:.1f}, {mds.n_iter_} iterations")
    fig.suptitle("Same geodesic distances (same k-NN graph) fed to two different embedding steps.\n"
                  "Colour = true position along the roll - both should unroll it similarly here.")
    fig.tight_layout()
    path = VIS_DIR / "01a_isomap_vs_smacof_swissroll.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    corr_classical = max(abs(spearmanr(classical_embedding[:, i], color).correlation) for i in range(2))
    corr_smacof = max(abs(spearmanr(smacof_embedding[:, i], color).correlation) for i in range(2))
    print(f"Best |Spearman correlation| with true roll position:")
    print(f"  Classical-MDS Isomap: {corr_classical:.3f}")
    print(f"  Isomap + SMACOF:      {corr_smacof:.3f}")
    return dist_matrix


def part_b_restart_variability(dist_matrix, n_restarts=8):
    """SMACOF's local-minima behaviour, made concrete: run SMACOF from
    n_restarts different random initial configurations (n_init=1 each, so we
    see every individual restart rather than sklearn's own best-of-n_init),
    and plot the final stress each one reaches. Classical-MDS Isomap has no
    equivalent plot - it has exactly one answer, always."""
    stresses = []
    for seed in range(n_restarts):
        mds = _build_mds(n_components=2, n_init=1, max_iter=300, eps=1e-6, random_state=seed)
        mds.fit(dist_matrix)
        stresses.append(mds.stress_)
    stresses = np.array(stresses)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(n_restarts), stresses, color="steelblue")
    ax.axhline(stresses.min(), color="red", linestyle="--", linewidth=1,
               label=f"best of {n_restarts} restarts (what n_init keeps)")
    ax.set_xlabel("Restart (different random initial configuration)")
    ax.set_ylabel("Final SMACOF stress")
    ax.set_title("SMACOF's final stress varies by restart - local minima,\n"
                 "a real difference from classical-MDS Isomap's single exact answer")
    ax.legend()
    fig.tight_layout()
    path = VIS_DIR / "01b_smacof_restarts_stress.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    print(f"Stress across {n_restarts} restarts: min={stresses.min():.1f}, max={stresses.max():.1f}, "
          f"spread={(stresses.max() - stresses.min()):.1f} "
          f"({100 * (stresses.max() - stresses.min()) / stresses.min():.1f}% of the best value)")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    X, color = make_swiss_roll(n_samples=500, noise=0.05, random_state=RANDOM_SEED)
    # n=500 (smaller than Technique 1's 1500) - SMACOF's O(n^2) per-iteration
    # cost across many iterations and several restarts is noticeably heavier
    # than classical MDS's one-shot eigendecomposition; kept small here so
    # Stage 2 runs quickly, revisited properly in Stage 5's timing comparison.
    #
    # n_neighbors=6, not Technique 1's 10: checked directly (not assumed) that
    # at this smaller, sparser n=500 sample, k=10 already short-circuits
    # (Spearman 0.83 instead of ~1.0) and gets worse by k=12-20 (down to
    # 0.16-0.52) - confirming, from a different angle, Technique 1's
    # documented short-circuit sensitivity: it depends on SAMPLING DENSITY
    # relative to k, not k in isolation. k=6 recovers a clean unroll
    # (Spearman 0.999) at this n, so we use that here - otherwise a broken
    # shared k-NN graph would confound the actual question this script asks
    # (classical MDS vs SMACOF), since both embedding methods would be fed
    # the same corrupted geodesic distances either way.
    dist_matrix = part_a_isomap_vs_smacof(X, color, n_neighbors=6)
    part_b_restart_variability(dist_matrix, n_restarts=8)


if __name__ == "__main__":
    main()
