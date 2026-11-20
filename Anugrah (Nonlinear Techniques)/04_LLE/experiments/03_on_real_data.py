"""
LLE on REAL Raichur Sentinel-2 data. Same pattern as the prior three
techniques' real-data scripts. Cannot be run in the sandbox this project was
developed in (see src/data_loader.py docstring).

Sample size: LLE was the lightest-memory technique on synthetic data (2.47 MB
vs Isomap's 63 MB - no n x n matrix ever built), and comparably fast, so we
use Technique 1's full 8,000-pixel sample size here, same as Kernel PCA - no
cost-driven reason to shrink it the way Technique 2 (SMACOF) needed to.

Run:
    python 04_LLE/experiments/03_on_real_data.py
Output (if the data file is present):
    visualisations/03a_lle_real_data_composite.png
    results/lle_real_raichur.json
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
from lle import fit_lle_pixels

# Confirmed path on this project (see 01_Isomap's and 02_IsomapSMACOF's
# real-data scripts for the same default - edit if your checkout differs).
PARQUET_PATH = Path(
    REPO_ROOT / "DIP_project" / "data" / "processed" / "ML_Dataset" / "combined" / "raichur_features.parquet"
)
DATE = "2026-09-27"
SAMPLE_SIZE = 8000  # same as Technique 1 and 3 - see docstring
N_NEIGHBORS = 12
N_COMPONENTS = 3
REG = 1e-3
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
    embedding, model, diagnostics = fit_lle_pixels(
        pixels_std, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, reg=REG, random_state=RANDOM_SEED
    )
    print(f"Fit: {diagnostics['fit_time_seconds']:.2f}s, {diagnostics['peak_memory_mb']:.2f} MB peak, "
          f"reconstruction_error={diagnostics['reconstruction_error']:.2e}")

    metrics = summarize(pixels_std, embedding, model=model, diagnostics=diagnostics, n_neighbors=N_NEIGHBORS)
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
    ax.set_title(f"LLE on real Raichur data, {DATE}\n"
                 f"({SAMPLE_SIZE} of {n_valid} valid pixels, 7->3 components)")
    ax.axis("off")
    fig.tight_layout()
    path = VIS_DIR / "03a_lle_real_data_composite.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "technique": "LLE",
        "dataset": "real_raichur",
        "date": DATE,
        "n_valid_pixels_available": int(n_valid),
        "n_pixels_used": int(len(sample_idx)),
        "lst_available": tile["lst_available"],
        "channels_used": tile["channel_names"],
        "n_neighbors": N_NEIGHBORS,
        "n_components": N_COMPONENTS,
        "reg": REG,
        "random_seed": RANDOM_SEED,
        "metrics": metrics,
        "known_limitations": [
            "Clouds/shadows are not masked in this dataset.",
            "LST is not available upstream; 6 channels used, not 7.",
            "Pixels were randomly subsampled (8,000 of the available valid pixels).",
            "Stage 4 on synthetic data found LLE's own (very low) reconstruction "
            "error did NOT correspond to good global neighbourhood preservation - "
            "worth checking whether the same disconnect appears on real data.",
        ],
    }
    out_path = RESULTS_DIR / "lle_real_raichur.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
