"""
Independent Component Analysis (FastICA) - runs on ANY supported dataset.

    python algorithms/ica/main.py                          # asks for the dataset path
    python algorithms/ica/main.py --dataset PATH           # file or folder (PNG/JPG, TIFF/GeoTIFF, ENVI, .npy/.npz, .mat, .nc, .h5)
    python algorithms/ica/main.py --dataset PATH --no-standardize
Or set DATASET_PATH below and simply run the file.
Results: outputs/<dataset_id>/ica/  (models, reduced_data, reconstructed_data, metrics, graphs, logs,
         metrics.csv, metrics.json, metrics_report.txt). Hyper-parameters: algorithms/ica/config.py
"""
# ============================== CONFIGURATION ==============================
DATASET_PATH = ""   # file or folder, e.g. r"data/Indian_pines_corrected.mat"; "" = ask when run
OUTPUT_ROOT = ""    # "" = Kartavya/outputs
STANDARDIZE = None  # None = use config.py; False = no z-scoring (results in <algorithm>_nostd/)
# ===========================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms.ica import config as cfg  # noqa: E402
from algorithms.ica import pipeline  # noqa: E402
from common.pipeline import cli_option, run_single  # noqa: E402


def main():
    if STANDARDIZE is not None:
        cfg.STANDARDIZE = STANDARDIZE
    if cli_option("--no-standardize", False, bool):
        cfg.STANDARDIZE = False
    return run_single(pipeline, DATASET_PATH or None, output_root=OUTPUT_ROOT or None)


if __name__ == "__main__":
    sys.exit(main())
