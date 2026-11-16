"""
UBS-CNN (BS-Net-Conv) channel selection - needs PyTorch - runs on ANY supported dataset.

    python algorithms/ubscnn/main.py                          # asks for the dataset path
    python algorithms/ubscnn/main.py --dataset PATH           # file or folder (PNG/JPG, TIFF/GeoTIFF, ENVI, .npy/.npz, .mat, .nc, .h5)
    python algorithms/ubscnn/main.py --dataset PATH --epochs 20
Or set DATASET_PATH below and simply run the file.
Results: outputs/<dataset_id>/ubscnn/  (models, reduced_data, reconstructed_data, metrics, graphs, logs,
         metrics.csv, metrics.json, metrics_report.txt). Hyper-parameters: algorithms/ubscnn/config.py
"""
# ============================== CONFIGURATION ==============================
DATASET_PATH = ""   # file or folder, e.g. r"data/Indian_pines_corrected.mat"; "" = ask when run
OUTPUT_ROOT = ""    # "" = Kartavya/outputs
EPOCHS = None       # None = config.EPOCHS (100); a smaller number gives a quicker run
# ===========================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms.ubscnn import config as cfg  # noqa: E402
from algorithms.ubscnn import pipeline  # noqa: E402
from common.pipeline import cli_option, run_single  # noqa: E402


def main():
    epochs = cli_option("--epochs", EPOCHS, int)
    return run_single(pipeline, DATASET_PATH or None, output_root=OUTPUT_ROOT or None, options={"epochs": epochs})


if __name__ == "__main__":
    sys.exit(main())
