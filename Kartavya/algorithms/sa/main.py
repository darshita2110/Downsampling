"""
Simulated Annealing (SA) channel selection - runs on ANY supported dataset.

    python algorithms/sa/main.py                          # asks for the dataset path
    python algorithms/sa/main.py --dataset PATH           # file or folder (PNG/JPG, TIFF/GeoTIFF, ENVI, .npy/.npz, .mat, .nc, .h5)

Or set DATASET_PATH below and simply run the file.
Results: outputs/<dataset_id>/sa/  (models, reduced_data, reconstructed_data, metrics, graphs, logs,
         metrics.csv, metrics.json, metrics_report.txt). Hyper-parameters: algorithms/sa/config.py
"""
# ============================== CONFIGURATION ==============================
DATASET_PATH = ""   # file or folder, e.g. r"data/Indian_pines_corrected.mat"; "" = ask when run
OUTPUT_ROOT = ""    # "" = Kartavya/outputs
# ===========================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms.sa import config as cfg  # noqa: E402
from algorithms.sa import pipeline  # noqa: E402
from common.pipeline import cli_option, run_single  # noqa: E402


def main():
    return run_single(pipeline, DATASET_PATH or None, output_root=OUTPUT_ROOT or None)


if __name__ == "__main__":
    sys.exit(main())
