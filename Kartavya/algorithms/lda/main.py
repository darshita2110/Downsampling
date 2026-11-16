"""
Supervised Linear Discriminant Analysis (LDA) - needs ground-truth labels - runs on ANY supported dataset.

    python algorithms/lda/main.py                          # asks for the dataset path
    python algorithms/lda/main.py --dataset PATH           # file or folder (PNG/JPG, TIFF/GeoTIFF, ENVI, .npy/.npz, .mat, .nc, .h5)
    python algorithms/lda/main.py --dataset PATH --no-standardize
    python algorithms/lda/main.py --dataset PATH --labels LABELS
    python algorithms/lda/main.py --self-check
Or set DATASET_PATH below and simply run the file.
Results: outputs/<dataset_id>/lda/  (models, reduced_data, reconstructed_data, metrics, graphs, logs,
         metrics.csv, metrics.json, metrics_report.txt). Hyper-parameters: algorithms/lda/config.py
"""
# ============================== CONFIGURATION ==============================
DATASET_PATH = ""   # file or folder, e.g. r"data/Indian_pines_corrected.mat"; "" = ask when run
LABEL_PATH = ""     # optional label raster/folder; "" = auto-detect files named *gt* / *label* next to the data
OUTPUT_ROOT = ""    # "" = Kartavya/outputs
STANDARDIZE = None  # None = use config.py; False = no z-scoring (results in <algorithm>_nostd/)
SELF_CHECK = False  # True = only verify the LDA code on scikit-learn's Iris and Wine data
# ===========================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms.lda import config as cfg  # noqa: E402
from algorithms.lda import pipeline  # noqa: E402
from common.pipeline import cli_option, run_single  # noqa: E402


def main():
    if STANDARDIZE is not None:
        cfg.STANDARDIZE = STANDARDIZE
    if cli_option("--no-standardize", False, bool):
        cfg.STANDARDIZE = False
    if SELF_CHECK or cli_option("--self-check", False, bool):
        from common.config import OUTPUT_ROOT as DEFAULT_ROOT
        pipeline.self_check(Path(OUTPUT_ROOT) if OUTPUT_ROOT else DEFAULT_ROOT)
        return 0
    return run_single(pipeline, DATASET_PATH or None, label_path=LABEL_PATH or None, output_root=OUTPUT_ROOT or None)


if __name__ == "__main__":
    sys.exit(main())
