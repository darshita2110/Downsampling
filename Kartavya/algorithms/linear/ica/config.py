"""
All settings for the ICA experiments. Change values here, not inside train.py/test.py.

Experiments:
  "png" : the 4 processed products in data/processed/sentinel2/ (RGB, False_Color, NDVI, NDWI)
          -> 8 channels per pixel (3 + 3 display levels + 2 derived indices)
  "raw" : the ORIGINAL Sentinel-2 bands B02, B03, B04, B08 from data/raw/sentinel2_l1c/*.nc, only if
          those files exist. Nothing is invented if they are missing.
The dataset settings are the same as the PCA experiment's so the two can be compared directly.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[4]
KARTAVYA_ROOT = REPO_ROOT / "Kartavya"
OUTPUT_ROOT = KARTAVYA_ROOT / "outputs"
PNG_DATASET_DIR = REPO_ROOT / "data" / "processed" / "sentinel2"
RAW_DATASET_DIR = REPO_ROOT / "data" / "raw" / "sentinel2_l1c"

ALGORITHM = "ica"


def output_dirs(experiment):
    """Every output goes to Kartavya/outputs/<category>/ica/<experiment>/."""
    cats = ["models", "reduced_data", "reconstructed_data", "metrics", "graphs", "logs"]
    return {c: OUTPUT_ROOT / c / ALGORITHM / experiment for c in cats}


# ------------------------------------------------------------------ PNG experiment
PNG_FOLDERS = {   # folder order = channel order; do not reorder between runs
    "RGB":         {"kind": "composite", "channels": ["R(B04)", "G(B03)", "B(B02)"]},
    "False_Color": {"kind": "composite", "channels": ["R(B08)", "G(B04)", "B(B03)"]},
    "NDVI":        {"kind": "colormap", "cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0, "channel": "NDVI"},
    "NDWI":        {"kind": "colormap", "cmap": "RdBu",   "vmin": -1.0, "vmax": 1.0, "channel": "NDWI"},
}
EXPECTED_IMAGES_PER_FOLDER = 5
PNG_NORMALIZATION = [("RGB:", 0.0, 255.0), ("False_Color:", 0.0, 255.0),
                     ("NDVI", -1.0, 1.0), ("NDWI", -1.0, 1.0)]
TARGET_WIDTH = 1024               # common grid width; height follows the map's aspect ratio

# ------------------------------------------------------------------ raw experiment
RAW_BANDS = ["B02", "B03", "B04", "B08"]
RAW_SCALE = 10000.0
RAW_NORMALIZATION = [("B", 0.0, 1.2)]
RAW_DOWNSAMPLE = 3

# ------------------------------------------------------------------ train/test split
# Indices refer to the sorted filename order (metrics/filename_index_mapping.json).
# None = automatic: >= 2 dates -> first 60 % of indices train, the rest test (5 dates -> [0,1,2] / [3,4]);
# 1 date -> top half of the image trains, bottom half tests.
TRAIN_INDICES = None
TEST_INDICES = None
TRAIN_FRACTION = 0.6
SPATIAL_TRAIN_FRACTION = 0.5

# ------------------------------------------------------------------ ICA hyper-parameters
STANDARDIZE = True          # z-score channels with TRAINING mean/std before ICA
MAX_TRAIN_PIXELS = 300_000  # random training-pixel sample used to fit (FastICA iterates over it)
RANDOM_SEED = 42            # pixel sampling and the random initial rotation
ICA_ALGORITHM = "parallel"  # all components estimated together (symmetric decorrelation)
ICA_FUN = "logcosh"         # contrast G(u) = log cosh(alpha u) / alpha, g(u) = tanh(alpha u)
ICA_FUN_ALPHA = 1.0
ICA_MAX_ITER = 1000
ICA_TOL = 1e-4
RANK_TOL = 1e-10            # eigenvalue / largest below this = zero variance -> not a usable component
# K values to fit. None = every K from 1 to the numerical rank of the training data (ICA cannot whiten
# zero-variance directions, e.g. the duplicated B04/B03 channels give rank 6 for the 8 PNG channels).
COMPONENT_COUNTS = None

# ------------------------------------------------------------------ evaluation
SAVE_RECONSTRUCTED_K = None  # K values whose full reconstructed arrays are saved (None = all)
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5           # transforms are timed this many times; the median is reported
