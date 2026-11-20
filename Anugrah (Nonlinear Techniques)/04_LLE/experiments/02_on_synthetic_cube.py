"""
LLE Stage 4 (synthetic cube) and Stage 5 (evaluation), matching Technique 1-3's
synthetic runs for direct comparability.

Run (from the project root, Anugrah (Nonlinear Techniques)/):
    python 04_LLE/experiments/02_on_synthetic_cube.py
Output:
    visualisations/02a_lle_embedding_composite.png
    visualisations/02b_latent_recovery.png
    results/lle_synthetic_v1.json
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
from lle import fit_lle_cube

DATA_DIR = PROJECT_ROOT / "data" / "synthetic_v1"
VIS_DIR = TECHNIQUE_DIR / "visualisations"
RESULTS_DIR = TECHNIQUE_DIR / "results"
N_NEIGHBORS = 12  # same as Technique 1/2's synthetic runs; also > 7 channels, so reg matters (see Stage 1)
N_COMPONENTS = 3
REG = 1e-3
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

    embedding_cube, model, diagnostics = fit_lle_cube(
        cube, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, reg=REG, random_state=RANDOM_SEED
    )
    print(f"Fit: {diagnostics['fit_time_seconds']:.3f}s, {diagnostics['peak_memory_mb']:.2f} MB peak, "
          f"reconstruction_error={diagnostics['reconstruction_error']:.2e}")

    embedding_pixels, _ = cube_to_pixels(embedding_cube)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cube[..., [2, 1, 0]])
    axes[0].set_title("Original cube\n(B04, B03, B02 shown as RGB)")
    axes[0].axis("off")
    axes[1].imshow(false_color(embedding_cube))
    axes[1].set_title(f"LLE embedding (7 -> {N_COMPONENTS})\nshown as false colour")
    axes[1].axis("off")
    fig.tight_layout()
    path = VIS_DIR / "02a_lle_embedding_composite.png"
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
    fig.suptitle("Top: ground truth. Bottom: LLE's 3 output dimensions.")
    fig.tight_layout()
    path = VIS_DIR / "02b_latent_recovery.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Check whether a weak single-dimension correlation (if any) is a rotation
    # artifact (as found for Isomap/SMACOF) or a real gap (as found for Kernel
    # PCA) - do this unconditionally, not just when a number looks weak, so
    # the check is applied consistently rather than only when convenient.
    from sklearn.linear_model import LinearRegression
    joint_recovery = {}
    for name, true_latent in [("t", t), ("s", s)]:
        reg_model = LinearRegression().fit(embedding_pixels, true_latent)
        pred = reg_model.predict(embedding_pixels)
        joint_recovery[name] = float(spearmanr(pred, true_latent).correlation)
    print(f"Joint (all 3 dims combined) recovery: t={joint_recovery['t']:.3f}, s={joint_recovery['s']:.3f}")

    metrics = summarize(pixels_std, embedding_pixels, model=model, diagnostics=diagnostics, n_neighbors=N_NEIGHBORS)
    # model=model here (unlike Kernel PCA/SMACOF's model=None): src/evaluation.py's
    # reconstruction_error() now explicitly recognises LLE's `.reconstruction_error_`
    # attribute (fixed this session - see src/evaluation.py's updated docstring).
    # explained_variance_ratio remains None here: LLE's eigenproblem has no
    # variance interpretation (same reasoning as Laplacian Eigenmaps will have).
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    isomap_result_path = PROJECT_ROOT / "01_Isomap" / "results" / "isomap_synthetic_v1.json"
    comparison = None
    if isomap_result_path.exists():
        isomap_result = json.loads(isomap_result_path.read_text())
        comparison = {
            "trustworthiness_isomap_vs_lle": [isomap_result["metrics"]["trustworthiness"], metrics["trustworthiness"]],
            "continuity_isomap_vs_lle": [isomap_result["metrics"]["continuity"], metrics["continuity"]],
            "neighborhood_preservation_isomap_vs_lle": [isomap_result["metrics"]["neighborhood_preservation"], metrics["neighborhood_preservation"]],
            "fit_time_seconds_isomap_vs_lle": [isomap_result["metrics"]["fit_time_seconds"], metrics["fit_time_seconds"]],
        }
        print("\nDirect comparison vs. Technique 1 (Isomap), same dataset/settings:")
        for name, (v1, v2) in comparison.items():
            print(f"  {name}: Isomap={v1:.4f}  LLE={v2:.4f}")

    results = {
        "technique": "LLE",
        "dataset": "synthetic_v1",
        "n_neighbors": N_NEIGHBORS,
        "n_components": N_COMPONENTS,
        "reg": REG,
        "random_seed": RANDOM_SEED,
        "reconstruction_error_lle_native": diagnostics["reconstruction_error"],
        "metrics": metrics,
        "latent_recovery_spearman": [
            {"embedding_dim": c, "best_matched_latent": name, "spearman": corr}
            for c, name, corr in correlations
        ],
        "latent_recovery_joint_all_dims": joint_recovery,
        "comparison_vs_technique1_isomap": comparison,
    }
    out_path = RESULTS_DIR / "lle_synthetic_v1.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
