"""
Kernel PCA Stage 4 (synthetic cube) and Stage 5 (evaluation), run as close to
identically to Technique 1/2's synthetic runs as this technique allows, so
all three are comparable.

Run (from the project root, Anugrah (Nonlinear Techniques)/):
    python 03_KernelPCA/experiments/02_on_synthetic_cube.py
Output:
    visualisations/02a_kpca_embedding_composite.png
    visualisations/02b_latent_recovery.png
    results/kernel_pca_synthetic_v1.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TECHNIQUE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TECHNIQUE_DIR))

from src.evaluation import summarize
from src.preprocessing import cube_to_pixels, standardize
from kernel_pca import explained_variance_ratio, fit_kernel_pca_cube

DATA_DIR = PROJECT_ROOT / "data" / "synthetic_v1"
VIS_DIR = TECHNIQUE_DIR / "visualisations"
RESULTS_DIR = TECHNIQUE_DIR / "results"
N_COMPONENTS = 3
GAMMA = None  # scikit-learn default (1/n_channels); Stage 2's sweep showed no
# gamma unrolled the pure Swiss roll, but that doesn't mean the default is
# wrong for THIS, differently-shaped dataset - checked directly here instead
# of assumed.
RANDOM_SEED = 42


def false_color(embedding_cube):
    out = np.empty_like(embedding_cube)
    for c in range(embedding_cube.shape[-1]):
        channel = embedding_cube[..., c]
        out[..., c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    return out


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cube = np.load(DATA_DIR / "cube.npy")
    latents_cube = np.load(DATA_DIR / "latents.npy")
    pixels, spatial_shape = cube_to_pixels(cube)
    latents_pixels, _ = cube_to_pixels(latents_cube)
    pixels_std, _, _ = standardize(pixels)

    embedding_cube, model, diagnostics = fit_kernel_pca_cube(
        cube, n_components=N_COMPONENTS, gamma=GAMMA, random_state=RANDOM_SEED
    )
    print(f"Fit: {diagnostics['fit_time_seconds']:.3f}s, {diagnostics['peak_memory_mb']:.2f} MB peak, "
          f"gamma_used={diagnostics['gamma_used']:.4f}")

    embedding_pixels, _ = cube_to_pixels(embedding_cube)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cube[..., [2, 1, 0]])
    axes[0].set_title("Original cube\n(B04, B03, B02 shown as RGB)")
    axes[0].axis("off")
    axes[1].imshow(false_color(embedding_cube))
    axes[1].set_title(f"Kernel PCA embedding (7 -> {N_COMPONENTS})\nshown as false colour")
    axes[1].axis("off")
    fig.tight_layout()
    path = VIS_DIR / "02a_kpca_embedding_composite.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    t, s = latents_pixels[:, 0], latents_pixels[:, 1]
    correlations = []
    for c in range(N_COMPONENTS):
        corr_t = spearmanr(embedding_pixels[:, c], t).correlation
        corr_s = spearmanr(embedding_pixels[:, c], s).correlation
        correlations.append((c, "t", corr_t) if abs(corr_t) > abs(corr_s) else (c, "s", corr_s))

    fig, axes = plt.subplots(2, max(2, N_COMPONENTS), figsize=(4 * N_COMPONENTS, 7))
    side = int(np.sqrt(len(t)))
    axes[0, 0].imshow(t.reshape(side, side), cmap="viridis"); axes[0, 0].set_title("True latent t"); axes[0, 0].axis("off")
    axes[0, 1].imshow(s.reshape(side, side), cmap="viridis"); axes[0, 1].set_title("True latent s"); axes[0, 1].axis("off")
    for extra in range(2, axes.shape[1]):
        axes[0, extra].axis("off")
    for c in range(N_COMPONENTS):
        axes[1, c].imshow(embedding_pixels[:, c].reshape(side, side), cmap="viridis")
        _, matched_name, corr = correlations[c]
        axes[1, c].set_title(f"Embedding dim {c}\nbest match: {matched_name} (Spearman={corr:.2f})")
        axes[1, c].axis("off")
    fig.suptitle("Top: ground truth. Bottom: Kernel PCA's 3 output dimensions.")
    fig.tight_layout()
    path = VIS_DIR / "02b_latent_recovery.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    metrics = summarize(pixels_std, embedding_pixels, model=None, diagnostics=diagnostics, n_neighbors=12)
    # model=None here deliberately: src/evaluation.reconstruction_error and
    # explained_variance_ratio are written around Isomap's specific attributes
    # (model.reconstruction_error(), model.kernel_pca_); Kernel PCA's own
    # explained-variance figure is computed separately below via the
    # dedicated, full-spectrum-aware function in kernel_pca.py instead.
    metrics["explained_variance_ratio"] = explained_variance_ratio(
        pixels_std, n_components=N_COMPONENTS, gamma=GAMMA, random_state=RANDOM_SEED
    )
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    isomap_result_path = PROJECT_ROOT / "01_Isomap" / "results" / "isomap_synthetic_v1.json"
    comparison = None
    if isomap_result_path.exists():
        isomap_result = json.loads(isomap_result_path.read_text())
        comparison = {
            "trustworthiness_isomap_vs_kpca": [isomap_result["metrics"]["trustworthiness"], metrics["trustworthiness"]],
            "continuity_isomap_vs_kpca": [isomap_result["metrics"]["continuity"], metrics["continuity"]],
            "neighborhood_preservation_isomap_vs_kpca": [isomap_result["metrics"]["neighborhood_preservation"], metrics["neighborhood_preservation"]],
            "fit_time_seconds_isomap_vs_kpca": [isomap_result["metrics"]["fit_time_seconds"], metrics["fit_time_seconds"]],
        }
        print("\nDirect comparison vs. Technique 1 (Isomap), same dataset/settings:")
        for name, (v1, v2) in comparison.items():
            print(f"  {name}: Isomap={v1:.4f}  KernelPCA={v2:.4f}")

    results = {
        "technique": "Kernel PCA",
        "dataset": "synthetic_v1",
        "n_components": N_COMPONENTS,
        "kernel": "rbf",
        "gamma_used": diagnostics["gamma_used"],
        "random_seed": RANDOM_SEED,
        "metrics": metrics,
        "latent_recovery_spearman": [
            {"embedding_dim": c, "best_matched_latent": name, "spearman": corr}
            for c, name, corr in correlations
        ],
        "comparison_vs_technique1_isomap": comparison,
    }
    out_path = RESULTS_DIR / "kernel_pca_synthetic_v1.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
