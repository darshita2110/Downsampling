"""
Isomap Stage 4 (apply to a spatial-spectral cube) and Stage 5 (evaluation) on
our synthetic 7-channel dataset (data/synthetic_v1/, see
experiments/00_make_synthetic_dataset.py).

What this script does, and why, in order:
  1. Load the (40, 40, 7) cube and the KNOWN ground-truth latents (t, s) that
     generated it (only used for evaluation here - never fed to Isomap).
  2. Standardise channels (see src/preprocessing.standardize) - required because
     our 7 synthetic channels, like real B02 vs NDVI, could otherwise be on
     different numeric scales.
  3. Run Isomap pixel-wise, 7 -> 3 components (src/isomap.fit_isomap_cube).
  4. Reshape the (n_pixels, 3) embedding back into a (40, 40, 3) cube and save
     it as a false-colour composite image.
  5. Check recovery of the TRUE 2-D latent structure (Spearman correlation
     between each embedding dimension and the true t, s latents) - this is
     only possible because we built this dataset ourselves and know the answer;
     real data in Stage/Section 5 will not allow this check.
  6. Run the shared evaluation suite (trustworthiness, continuity, neighbourhood
     preservation, reconstruction error, timing, memory) from src/evaluation.py.
  7. Demonstrate, concretely, that pixel-wise Isomap does NOT use spatial
     position: for one example pixel, compare its spectral k-NN (what Isomap's
     graph actually uses) against its spatial k-NN (its literal neighbours in
     the image) and show these are different sets scattered across the image.
  8. Save a results row to results/isomap_synthetic_v1.json for the eventual
     Section-5 comparison table.

Run:
    python experiments/03_isomap_on_synthetic_cube.py
Output:
    visualisations/03a_isomap_embedding_composite.png
    visualisations/03b_latent_recovery.png
    visualisations/03c_spectral_vs_spatial_neighbors.png
    results/isomap_synthetic_v1.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `from src...` works when run as a script

from src.evaluation import summarize
from src.isomap import fit_isomap_cube
from src.preprocessing import cube_to_pixels, standardize

DATA_DIR = ROOT / "data" / "synthetic_v1"
VIS_DIR = ROOT / "visualisations"
RESULTS_DIR = ROOT / "results"
N_NEIGHBORS = 12
N_COMPONENTS = 3
RANDOM_SEED = 42


def false_color(embedding_cube):
    """Min-max-normalise each of the 3 embedding channels into [0, 1] so they
    can be shown as an RGB image. Purely for visualisation - the actual numeric
    embedding values (not this rescaled version) are what get evaluated."""
    out = np.empty_like(embedding_cube)
    for c in range(embedding_cube.shape[-1]):
        channel = embedding_cube[..., c]
        out[..., c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    return out


def plot_embedding_composite(embedding_cube, original_cube):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_cube[..., [2, 1, 0]])  # B04,B03,B02 as a rough "true colour"
    axes[0].set_title("Original cube\n(B04, B03, B02 shown as RGB)")
    axes[0].axis("off")
    axes[1].imshow(false_color(embedding_cube))
    axes[1].set_title(f"Isomap embedding (7 -> {N_COMPONENTS} components)\nshown as false colour")
    axes[1].axis("off")
    fig.tight_layout()
    path = VIS_DIR / "03a_isomap_embedding_composite.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_latent_recovery(embedding_pixels, latents_pixels):
    """For each embedding dimension, find its best-correlated true latent and
    show the two side by side. This tells us whether Isomap recovered
    coordinates that track the TRUE drivers, not just low reconstruction
    error - a stronger check than metrics computed from the data alone."""
    t, s = latents_pixels[:, 0], latents_pixels[:, 1]
    correlations = []
    for c in range(embedding_pixels.shape[1]):
        corr_t = spearmanr(embedding_pixels[:, c], t).correlation
        corr_s = spearmanr(embedding_pixels[:, c], s).correlation
        correlations.append((c, "t", corr_t) if abs(corr_t) > abs(corr_s) else (c, "s", corr_s))

    fig, axes = plt.subplots(2, max(2, embedding_pixels.shape[1]), figsize=(4 * embedding_pixels.shape[1], 7))
    side = int(np.sqrt(len(t)))
    axes[0, 0].imshow(t.reshape(side, side), cmap="viridis"); axes[0, 0].set_title("True latent t"); axes[0, 0].axis("off")
    axes[0, 1].imshow(s.reshape(side, side), cmap="viridis"); axes[0, 1].set_title("True latent s"); axes[0, 1].axis("off")
    for extra in range(2, axes.shape[1]):
        axes[0, extra].axis("off")
    for c in range(embedding_pixels.shape[1]):
        axes[1, c].imshow(embedding_pixels[:, c].reshape(side, side), cmap="viridis")
        _, matched_name, corr = correlations[c]
        axes[1, c].set_title(f"Embedding dim {c}\nbest match: {matched_name} (Spearman={corr:.2f})")
        axes[1, c].axis("off")
    fig.suptitle("Top row: ground truth (known only because this is synthetic). "
                 "Bottom row: Isomap's 3 output dimensions.")
    fig.tight_layout()
    path = VIS_DIR / "03b_latent_recovery.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return correlations


def plot_spectral_vs_spatial_neighbors(pixels_std, spatial_shape, pixel_index, k=10):
    """Concrete demonstration (not just an assertion) that the Isomap graph's
    'neighbours' are spectral, not spatial. Picks one example pixel and plots
    (a) its spatial neighbourhood (a small box around it in the image) against
    (b) its actual k spectral nearest neighbours (which, if spatial structure
    were being used, would also cluster tightly around it - they generally
    will not)."""
    from sklearn.neighbors import NearestNeighbors

    height, width = spatial_shape
    row0, col0 = divmod(pixel_index, width)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(pixels_std)
    spectral_neighbor_idx = nn.kneighbors([pixels_std[pixel_index]], return_distance=False)[0][1:]
    spectral_rc = [divmod(idx, width) for idx in spectral_neighbor_idx]

    grid = np.zeros((height, width))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, alpha=0.15)
    ax.scatter([col0], [row0], c="red", marker="*", s=200, label="query pixel", zorder=5)
    ax.scatter([c for _, c in spectral_rc], [r for r, _ in spectral_rc], c="blue", s=60,
               label=f"{k} spectral nearest neighbours\n(what Isomap's graph uses)", zorder=4)
    spatial_box = plt.Rectangle((col0 - 1.5, row0 - 1.5), 3, 3, fill=False,
                                 edgecolor="orange", linewidth=2,
                                 label="immediate spatial neighbourhood\n(3x3 window, NOT used by Isomap)")
    ax.add_patch(spatial_box)
    ax.set_xlim(-1, width); ax.set_ylim(height, -1)
    ax.set_title("Spectral neighbours (used by Isomap)\nvs. spatial neighbours (not used)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    path = VIS_DIR / "03c_spectral_vs_spatial_neighbors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    n_overlap = len(set(spectral_neighbor_idx) & {
        r * width + c for r in range(max(0, row0 - 1), min(height, row0 + 2))
        for c in range(max(0, col0 - 1), min(width, col0 + 2)) if (r, c) != (row0, col0)
    })
    print(f"Of the {k} spectral nearest neighbours Isomap's graph uses for this pixel, "
          f"only {n_overlap} are also immediate (3x3) spatial neighbours.")
    return n_overlap


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cube = np.load(DATA_DIR / "cube.npy")
    latents_cube = np.load(DATA_DIR / "latents.npy")  # ground truth, evaluation only
    pixels, spatial_shape = cube_to_pixels(cube)
    latents_pixels, _ = cube_to_pixels(latents_cube)
    pixels_std, _, _ = standardize(pixels)

    embedding_cube, model, diagnostics = fit_isomap_cube(
        cube, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, random_state=RANDOM_SEED
    )
    print(f"Isomap fit: {diagnostics['fit_time_seconds']:.3f}s, "
          f"{diagnostics['peak_memory_mb']:.2f} MB peak, "
          f"n_neighbors={N_NEIGHBORS}")
    if diagnostics["disconnected_graph_warning"]:
        print(f"WARNING: {diagnostics['disconnected_graph_warning']}")

    embedding_pixels, _ = cube_to_pixels(embedding_cube)

    plot_embedding_composite(embedding_cube, cube)
    correlations = plot_latent_recovery(embedding_pixels, latents_pixels)
    overlap = plot_spectral_vs_spatial_neighbors(
        pixels_std, spatial_shape, pixel_index=(spatial_shape[0] // 2) * spatial_shape[1] + spatial_shape[1] // 2,
        k=10,
    )

    metrics = summarize(pixels_std, embedding_pixels, model=model, diagnostics=diagnostics, n_neighbors=N_NEIGHBORS)
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    results = {
        "technique": "Isomap",
        "dataset": "synthetic_v1",
        "n_neighbors": N_NEIGHBORS,
        "n_components": N_COMPONENTS,
        "random_seed": RANDOM_SEED,
        "metrics": metrics,
        "latent_recovery_spearman": [
            {"embedding_dim": c, "best_matched_latent": name, "spearman": corr}
            for c, name, corr in correlations
        ],
        "spectral_vs_spatial_neighbor_overlap_example": overlap,
        "spatial_awareness_note": (
            "Pixel-wise Isomap's neighbourhood graph is built on spectral "
            "distance only; see 03c_spectral_vs_spatial_neighbors.png for a "
            "concrete per-pixel example of how little this overlaps with "
            "actual spatial adjacency."
        ),
    }
    out_path = RESULTS_DIR / "isomap_synthetic_v1.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
