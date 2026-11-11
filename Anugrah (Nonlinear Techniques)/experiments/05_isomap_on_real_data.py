"""
Isomap on REAL Raichur Sentinel-2 data. Cannot be run in the sandbox this
project was developed in (the Parquet file is not available there - see
src/data_loader.py docstring); run this on a machine that has
data/processed/ML_Dataset/combined/raichur_features.parquet in place per
docs/DATA_ACCESS_GUIDE.md. If the file is missing, this script says so clearly
and exits rather than failing confusingly.

Why we subsample pixels (documented, not hidden):
Exact Isomap needs all-pairs shortest paths, which documentation/02_isomap_stage4_5.md
already flagged as likely infeasible on a full tile. A single Raichur date can
have up to ~10.9 million valid pixels (see docs/DATA_ACCESS_GUIDE.md "Dates"
table). Even just the pairwise distance matrix at that size would need on the
order of 10^14 entries - far beyond memory. We therefore take a random subsample
of SAMPLE_SIZE pixels. This is a real, acknowledged limitation (see "Research
gap" note in documentation/02_isomap_stage4_5.md: patch-based or landmark-Isomap
variants would be a better long-term answer) - the comparison-table numbers this
produces describe Isomap run on a subsample, not the full tile, and the paper
must say so.

Run:
    python experiments/05_isomap_on_real_data.py
Output (if the data file is present):
    visualisations/05a_isomap_real_data_composite.png
    results/isomap_real_raichur.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ROOT = ROOT.parent  # the Downsampling repo root, where data/ lives

from src.data_loader import get_grid_reference, load_raichur_tile, reconstruct_grid
from src.evaluation import summarize
from src.isomap import fit_isomap_pixels
from src.preprocessing import standardize

PARQUET_PATH = REPO_ROOT / "data" / "processed" / "ML_Dataset" / "combined" / "raichur_features.parquet"
# 2026-09-27 is documented as "Mostly clear, some small clouds" at 100% coverage
# (docs/DATA_ACCESS_GUIDE.md) - a reasonable first real-data date. Clouds/shadows
# are NOT masked in this dataset (also documented) - a real limitation we are
# knowingly accepting for this first pass, not one we are hiding.
DATE = "2026-09-27"
SAMPLE_SIZE = 8000
N_NEIGHBORS = 12
N_COMPONENTS = 3
RANDOM_SEED = 42
VIS_DIR = ROOT / "visualisations"
RESULTS_DIR = ROOT / "results"


def main():
    if not PARQUET_PATH.exists():
        print(f"Real data file not found at:\n  {PARQUET_PATH}\n")
        print("This is expected if you haven't downloaded it yet - it is not stored in "
              "the GitHub repo (808 MB, gitignored). See docs/DATA_ACCESS_GUIDE.md for "
              "how to get it, then re-run this script.")
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
        print(f"Subsampling {SAMPLE_SIZE} of {n_valid} pixels (see this script's "
              f"docstring for why exact Isomap cannot run on the full tile).")
    else:
        sample_idx = np.arange(n_valid)

    pixels = tile["pixels"][sample_idx]
    row = tile["row"][sample_idx]
    col = tile["col"][sample_idx]

    pixels_std, _, _ = standardize(pixels)
    embedding, model, diagnostics = fit_isomap_pixels(
        pixels_std, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, random_state=RANDOM_SEED
    )
    print(f"Isomap fit: {diagnostics['fit_time_seconds']:.2f}s, "
          f"{diagnostics['peak_memory_mb']:.2f} MB peak")
    if diagnostics["disconnected_graph_warning"]:
        print(f"WARNING: {diagnostics['disconnected_graph_warning']}")

    metrics = summarize(pixels_std, embedding, model=model, diagnostics=diagnostics, n_neighbors=N_NEIGHBORS)
    print("\nEvaluation metrics (on the subsample):")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    # Visual: false-colour composite of the subsampled embedding, scattered
    # back into the real (gappy) tile grid - gaps include BOTH genuinely
    # uncovered pixels (clouds etc, documented upstream) AND pixels we simply
    # didn't sample this run; this script does not distinguish the two in the
    # plot, which is a known simplification worth noting in the paper.
    grid = reconstruct_grid(embedding, row, col, tile["grid_shape"])
    composite = np.empty_like(grid)
    for c in range(grid.shape[-1]):
        channel = grid[..., c]
        finite = channel[np.isfinite(channel)]
        if finite.size:
            composite[..., c] = (channel - finite.min()) / (finite.max() - finite.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.nan_to_num(composite, nan=1.0))  # NaN (uncovered/unsampled) shown as white
    ax.set_title(f"Isomap on real Raichur data, {DATE}\n"
                 f"({SAMPLE_SIZE} of {n_valid} valid pixels, 7->3 components)")
    ax.axis("off")
    fig.tight_layout()
    path = VIS_DIR / "05a_isomap_real_data_composite.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "technique": "Isomap",
        "dataset": "real_raichur",
        "date": DATE,
        "n_valid_pixels_available": int(n_valid),
        "n_pixels_used": int(len(sample_idx)),
        "subsampled": bool(n_valid > SAMPLE_SIZE),
        "lst_available": tile["lst_available"],
        "channels_used": tile["channel_names"],
        "n_neighbors": N_NEIGHBORS,
        "n_components": N_COMPONENTS,
        "random_seed": RANDOM_SEED,
        "metrics": metrics,
        "known_limitations": [
            "Clouds/shadows are not masked in this dataset (see docs/DATA_ACCESS_GUIDE.md); "
            "this run does not mask them either.",
            "LST is not available upstream; this run uses 6 channels (B02,B03,B04,B08,NDVI,NDWI), "
            "not the full assigned 7.",
            "Pixels were randomly subsampled due to Isomap's quadratic-or-worse scaling; "
            "metrics describe the subsample, not the full tile.",
        ],
    }
    out_path = RESULTS_DIR / "isomap_real_raichur.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
