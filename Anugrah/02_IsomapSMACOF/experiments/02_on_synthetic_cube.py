"""
Isomap-with-SMACOF Stage 4 (apply to our synthetic 7-channel cube) and Stage 5
(evaluation), run identically to 01_Isomap/experiments/03_isomap_on_synthetic_cube.py
so the two techniques are directly comparable.

What is NOT re-demonstrated here, and why: the spectral-vs-spatial-neighbour
check (01_Isomap/visualisations/03c_spectral_vs_spatial_neighbors.png) tests
Steps 1-2 of Isomap (the k-NN graph) - which are UNCHANGED by SMACOF (only
Step 3, the embedding, differs; see documentation/01_isomap_smacof.md). Running
the identical check again here would reproduce the identical result, not test
anything new about this technique - so we skip it and simply reference
Technique 1's figure instead, and focus this script on what SMACOF-specific
Stage 4/5 behaviour actually looks like on this dataset.

Run (from the project root, Anugrah (Nonlinear Techniques)/):
    python 02_IsomapSMACOF/experiments/02_on_synthetic_cube.py
Output:
    visualisations/02a_smacof_embedding_composite.png
    visualisations/02b_latent_recovery.png
    results/isomap_smacof_synthetic_v1.json
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
from isomap_smacof import fit_isomap_smacof_cube

DATA_DIR = PROJECT_ROOT / "data" / "synthetic_v1"
VIS_DIR = TECHNIQUE_DIR / "visualisations"
RESULTS_DIR = TECHNIQUE_DIR / "results"
N_NEIGHBORS = 12  # identical to Technique 1's synthetic-cube run, for comparability
N_COMPONENTS = 3
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

    embedding_cube, info, diagnostics = fit_isomap_smacof_cube(
        cube, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, random_state=RANDOM_SEED
    )
    print(f"Fit: geodesic {diagnostics['geodesic_distance_time_seconds']:.3f}s + "
          f"SMACOF {diagnostics['smacof_time_seconds']:.3f}s "
          f"({diagnostics['smacof_n_iter']} iterations) = "
          f"{diagnostics['fit_time_seconds']:.3f}s total, "
          f"{diagnostics['peak_memory_mb']:.2f} MB peak, SMACOF stress={diagnostics['smacof_stress']:.3f}")
    if diagnostics["disconnected_graph_warning"]:
        print(f"WARNING: {diagnostics['disconnected_graph_warning']}")

    embedding_pixels, _ = cube_to_pixels(embedding_cube)

    # Composite visual
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cube[..., [2, 1, 0]])
    axes[0].set_title("Original cube\n(B04, B03, B02 shown as RGB)")
    axes[0].axis("off")
    axes[1].imshow(false_color(embedding_cube))
    axes[1].set_title(f"Isomap+SMACOF embedding (7 -> {N_COMPONENTS})\nshown as false colour")
    axes[1].axis("off")
    fig.tight_layout()
    path = VIS_DIR / "02a_smacof_embedding_composite.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Latent recovery (same cyclic-latent caveat as Technique 1 - see
    # 01_Isomap/documentation/02_isomap_stage4_5.md; checked the same way here)
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
    fig.suptitle("Top: ground truth. Bottom: Isomap+SMACOF's 3 output dimensions.")
    fig.tight_layout()
    path = VIS_DIR / "02b_latent_recovery.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    metrics = summarize(pixels_std, embedding_pixels, model=None, diagnostics=diagnostics, n_neighbors=N_NEIGHBORS)
    # model=None: reconstruction_error()/explained_variance_ratio() are defined
    # on sklearn.manifold.Isomap specifically (see src/evaluation.py), not on
    # the MDS/SMACOF model - neither is mathematically defined the same way
    # for a SMACOF solution (its own stress, diagnostics['smacof_stress'], is
    # the appropriate analogue and is reported separately, not forced into
    # this shared summary's Isomap-shaped fields).
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    # Load Technique 1's result for a direct, same-dataset comparison
    isomap_result_path = PROJECT_ROOT / "01_Isomap" / "results" / "isomap_synthetic_v1.json"
    comparison = None
    if isomap_result_path.exists():
        isomap_result = json.loads(isomap_result_path.read_text())
        comparison = {
            "trustworthiness_isomap_vs_smacof": [isomap_result["metrics"]["trustworthiness"], metrics["trustworthiness"]],
            "continuity_isomap_vs_smacof": [isomap_result["metrics"]["continuity"], metrics["continuity"]],
            "neighborhood_preservation_isomap_vs_smacof": [isomap_result["metrics"]["neighborhood_preservation"], metrics["neighborhood_preservation"]],
            "fit_time_seconds_isomap_vs_smacof": [isomap_result["metrics"]["fit_time_seconds"], metrics["fit_time_seconds"]],
        }
        print("\nDirect comparison vs. Technique 1 (classical-MDS Isomap), same dataset/settings:")
        for name, (v1, v2) in comparison.items():
            print(f"  {name}: Isomap={v1:.4f}  SMACOF={v2:.4f}")

    results = {
        "technique": "Isomap with SMACOF",
        "dataset": "synthetic_v1",
        "n_neighbors": N_NEIGHBORS,
        "n_components": N_COMPONENTS,
        "random_seed": RANDOM_SEED,
        "smacof_n_init": diagnostics["n_init"],
        "smacof_max_iter": diagnostics["max_iter"],
        "smacof_n_iter": diagnostics["smacof_n_iter"],
        "smacof_stress": diagnostics["smacof_stress"],
        "metrics": metrics,
        "latent_recovery_spearman": [
            {"embedding_dim": c, "best_matched_latent": name, "spearman": corr}
            for c, name, corr in correlations
        ],
        "comparison_vs_technique1_isomap": comparison,
        "spatial_awareness_note": (
            "Steps 1-2 (k-NN graph, geodesic distances) are IDENTICAL to "
            "Technique 1 - see 01_Isomap/visualisations/03c_spectral_vs_spatial_neighbors.png "
            "for the (not re-run here) concrete demonstration that this graph "
            "is spectral, not spatial."
        ),
    }
    out_path = RESULTS_DIR / "isomap_smacof_synthetic_v1.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
