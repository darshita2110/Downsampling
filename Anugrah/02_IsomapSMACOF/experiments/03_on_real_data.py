"""
Isomap-with-SMACOF on REAL Raichur Sentinel-2 data. Same pattern as
01_Isomap/experiments/05_isomap_on_real_data.py - cannot be run in the sandbox
this project was developed in (see src/data_loader.py docstring); run on a
machine with the Parquet file in place per docs/DATA_ACCESS_GUIDE.md.

Why a SMALLER subsample than Technique 1 (2,000 vs. 8,000 pixels): Stage 4/5
on synthetic data (documentation/02_stage4_5_synthetic.md) measured SMACOF at
~25x Technique 1's runtime and ~2.5x its memory at the SAME pixel count.
Running SMACOF at Technique 1's full 8,000-pixel subsample would be expected
to take on the order of 8000/1600 (ratio used in Technique 1's own scaling
note) compounded with this ~25x SMACOF multiplier - likely several minutes at
best, possibly much worse given SMACOF's iteration count also scaling with n.
Rather than discover this empirically with a long, uncertain wait, we use a
smaller subsample here and say so explicitly - this is a stated, deliberate
choice, not a silent shortcut, and the resulting numbers describe Isomap+SMACOF
on 2,000 real pixels, not a fair pixel-count match to Technique 1's real-data
run (which used 8,000). Both pixel counts are recorded in the results JSON so
this difference is never lost when building the Section-5 table.

Run:
    python 02_IsomapSMACOF/experiments/03_on_real_data.py
Output (if the data file is present):
    visualisations/03a_smacof_real_data_composite.png
    results/isomap_smacof_real_raichur.json
"""
import json
import sys
import time
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
from isomap_smacof import fit_isomap_smacof_pixels

# Same default path caveat as Technique 1's real-data script - edit this if
# your checkout has the Parquet file somewhere else (see that script's
# comments and documentation/03_real_data_loader.md in 01_Isomap/'s sibling
# documentation folder for why this path differs across machines in practice).
PARQUET_PATH = Path(
    REPO_ROOT / "DIP_project" / "data" / "processed" / "ML_Dataset" / "combined" / "raichur_features.parquet"  # confirmed path on this project
)
DATE = "2026-09-27"
SAMPLE_SIZE = 2000  # smaller than Technique 1's 8,000 - see docstring
N_NEIGHBORS = 12
N_COMPONENTS = 3
RANDOM_SEED = 42
VIS_DIR = TECHNIQUE_DIR / "visualisations"
RESULTS_DIR = TECHNIQUE_DIR / "results"


def main():
    if not PARQUET_PATH.exists():
        print(f"Real data file not found at:\n  {PARQUET_PATH}\n")
        print("Edit PARQUET_PATH near the top of this script if your checkout "
              "has it elsewhere (see 01_Isomap/experiments/05_isomap_on_real_data.py "
              "for the same situation). See docs/DATA_ACCESS_GUIDE.md for how "
              "to get the file.")
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
        print(f"Subsampling {SAMPLE_SIZE} of {n_valid} pixels (smaller than "
              f"Technique 1's 8,000 - see this script's docstring for why).")
    else:
        sample_idx = np.arange(n_valid)

    pixels = tile["pixels"][sample_idx]
    row = tile["row"][sample_idx]
    col = tile["col"][sample_idx]

    pixels_std, _, _ = standardize(pixels)
    start = time.perf_counter()
    embedding, info, diagnostics = fit_isomap_smacof_pixels(
        pixels_std, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, random_state=RANDOM_SEED
    )
    total_time = time.perf_counter() - start
    print(f"Fit: geodesic {diagnostics['geodesic_distance_time_seconds']:.2f}s + "
          f"SMACOF {diagnostics['smacof_time_seconds']:.2f}s "
          f"({diagnostics['smacof_n_iter']} iterations) = {total_time:.2f}s total, "
          f"{diagnostics['peak_memory_mb']:.2f} MB peak, stress={diagnostics['smacof_stress']:.1f}")
    if diagnostics["disconnected_graph_warning"]:
        print(f"WARNING: {diagnostics['disconnected_graph_warning']}")

    metrics = summarize(pixels_std, embedding, model=None, diagnostics=diagnostics, n_neighbors=N_NEIGHBORS)
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
    ax.set_title(f"Isomap+SMACOF on real Raichur data, {DATE}\n"
                 f"({SAMPLE_SIZE} of {n_valid} valid pixels, 7->3 components)")
    ax.axis("off")
    fig.tight_layout()
    path = VIS_DIR / "03a_smacof_real_data_composite.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "technique": "Isomap with SMACOF",
        "dataset": "real_raichur",
        "date": DATE,
        "n_valid_pixels_available": int(n_valid),
        "n_pixels_used": int(len(sample_idx)),
        "n_pixels_used_technique1_isomap_for_comparison": 8000,
        "subsampled": bool(n_valid > SAMPLE_SIZE),
        "lst_available": tile["lst_available"],
        "channels_used": tile["channel_names"],
        "n_neighbors": N_NEIGHBORS,
        "n_components": N_COMPONENTS,
        "random_seed": RANDOM_SEED,
        "smacof_n_init": diagnostics["n_init"],
        "smacof_max_iter": diagnostics["max_iter"],
        "smacof_n_iter": diagnostics["smacof_n_iter"],
        "smacof_stress": diagnostics["smacof_stress"],
        "metrics": metrics,
        "known_limitations": [
            "Clouds/shadows are not masked in this dataset.",
            "LST is not available upstream; 6 channels used, not 7.",
            "Pixels were randomly subsampled, and at a SMALLER sample size "
            "(2,000) than Technique 1's real-data run (8,000) due to SMACOF's "
            "much higher per-pixel cost (see this script's docstring) - "
            "metrics are not directly pixel-count-comparable to Technique 1's "
            "real-data results without accounting for this.",
        ],
    }
    out_path = RESULTS_DIR / "isomap_smacof_real_raichur.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
