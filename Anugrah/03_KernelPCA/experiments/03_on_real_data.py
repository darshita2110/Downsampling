"""
Kernel PCA on REAL Raichur Sentinel-2 data. Same pattern as
01_Isomap/experiments/05_isomap_on_real_data.py and
02_IsomapSMACOF/experiments/03_on_real_data.py. Cannot be run in the sandbox
this project was developed in (see src/data_loader.py docstring).

Sample size: Kernel PCA was the fastest and lightest of the three techniques
tested so far on synthetic data (5x faster, less memory than Isomap - see
documentation/03_stage4_5_synthetic.md), so we use Technique 1's FULL 8,000
pixel sample size here rather than a reduced one - no cost-driven reason to
shrink it the way Technique 2 (SMACOF) needed to.

Run:
    python 03_KernelPCA/experiments/03_on_real_data.py
Output (if the data file is present):
    visualisations/03a_kpca_real_data_composite.png
    results/kernel_pca_real_raichur.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TECHNIQUE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TECHNIQUE_DIR))

from src.data_loader import get_grid_reference, load_raichur_tile, reconstruct_grid
from src.evaluation import summarize
from src.preprocessing import standardize
from kernel_pca import explained_variance_ratio, fit_kernel_pca_pixels

PARQUET_PATH = Path(
    REPO_ROOT / "DIP_project" / "data" / "processed" / "ML_Dataset" / "combined" / "raichur_features.parquet"  # confirmed path on this project
)
DATE = "2026-09-27"
SAMPLE_SIZE = 8000  # same as Technique 1 - see docstring
N_COMPONENTS = 3
RANDOM_SEED = 42
VIS_DIR = TECHNIQUE_DIR / "visualisations"
RESULTS_DIR = TECHNIQUE_DIR / "results"


def main():
    if not PARQUET_PATH.exists():
        print(f"Real data file not found at:\n  {PARQUET_PATH}\n")
        print("Edit PARQUET_PATH near the top of this script if your checkout "
              "has it elsewhere. See docs/DATA_ACCESS_GUIDE.md for how to get the file.")
        return

    print(f"Loading date={DATE} from {PARQUET_PATH} ...")
    reference = get_grid_reference(PARQUET_PATH, reference_date=DATE)
    tile = load_raichur_tile(PARQUET_PATH, date=DATE, lst_mode="drop",
                              reflectance_scale=True, grid_reference=reference)
    n_valid = tile["pixels"].shape[0]
    print(f"Loaded {n_valid} valid pixels, grid_shape={tile['grid_shape']}, "
          f"channels={tile['channel_names']}, lst_available={tile['lst_available']}")

    rng = np.random.default_rng(RANDOM_SEED)
    if n_valid > SAMPLE_SIZE:
        sample_idx = rng.choice(n_valid, size=SAMPLE_SIZE, replace=False)
        print(f"Subsampling {SAMPLE_SIZE} of {n_valid} pixels.")
    else:
        sample_idx = np.arange(n_valid)

    pixels = tile["pixels"][sample_idx]
    row = tile["row"][sample_idx]
    col = tile["col"][sample_idx]

    pixels_std, _, _ = standardize(pixels)
    embedding, model, diagnostics = fit_kernel_pca_pixels(
        pixels_std, n_components=N_COMPONENTS, random_state=RANDOM_SEED
    )
    print(f"Fit: {diagnostics['fit_time_seconds']:.2f}s, {diagnostics['peak_memory_mb']:.2f} MB peak, "
          f"gamma_used={diagnostics['gamma_used']:.4f}")

    metrics = summarize(pixels_std, embedding, model=None, diagnostics=diagnostics, n_neighbors=12)
    metrics["explained_variance_ratio"] = explained_variance_ratio(
        pixels_std, n_components=N_COMPONENTS, random_state=RANDOM_SEED
    )
    print("\nEvaluation metrics (on the subsample):")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    grid = reconstruct_grid(embedding, row, col, tile["grid_shape"])
    composite = np.empty_like(grid)
    for c in range(grid.shape[-1]):
        channel = grid[..., c]
        finite = channel[np.isfinite(channel)]
        if finite.size:
            composite[..., c] = (channel - finite.min()) / (finite.max() - finite.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.nan_to_num(composite, nan=1.0))
    ax.set_title(f"Kernel PCA on real Raichur data, {DATE}\n"
                 f"({SAMPLE_SIZE} of {n_valid} valid pixels, 7->3 components)")
    ax.axis("off")
    fig.tight_layout()
    path = VIS_DIR / "03a_kpca_real_data_composite.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "technique": "Kernel PCA",
        "dataset": "real_raichur",
        "date": DATE,
        "n_valid_pixels_available": int(n_valid),
        "n_pixels_used": int(len(sample_idx)),
        "lst_available": tile["lst_available"],
        "channels_used": tile["channel_names"],
        "n_components": N_COMPONENTS,
        "kernel": "rbf",
        "gamma_used": diagnostics["gamma_used"],
        "random_seed": RANDOM_SEED,
        "metrics": metrics,
        "known_limitations": [
            "Clouds/shadows are not masked in this dataset.",
            "LST is not available upstream; 6 channels used, not 7.",
            "Pixels were randomly subsampled (8,000 of the available valid pixels).",
            "Stage 4 on synthetic data found Kernel PCA did not recover one of two "
            "true latent variables at all (not just rotated, as with Isomap/SMACOF) - "
            "worth checking whether the real-data embedding similarly concentrates on "
            "one dominant mode.",
        ],
    }
    out_path = RESULTS_DIR / "kernel_pca_real_raichur.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
