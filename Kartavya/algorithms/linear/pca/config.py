"""
All settings for the PCA experiments in one place. Change values here, not inside train.py/test.py.

Two experiments are supported:
  "png" : the 4 processed products in data/processed/sentinel2/ (RGB, False_Color, NDVI, NDWI)
          -> 8 channels per pixel: 3 + 3 display levels + 2 derived indices.
  "raw" : the ORIGINAL Sentinel-2 L1C bands (B02, B03, B04, B08) from data/raw/sentinel2_l1c/*.nc,
          used only if those files exist. Nothing is invented if they are missing.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[4]          # .../Downsampling
KARTAVYA_ROOT = REPO_ROOT / "Kartavya"
OUTPUT_ROOT = KARTAVYA_ROOT / "outputs"

PNG_DATASET_DIR = REPO_ROOT / "data" / "processed" / "sentinel2"
RAW_DATASET_DIR = REPO_ROOT / "data" / "raw" / "sentinel2_l1c"

ALGORITHM = "pca"


def output_dirs(experiment):
    """Every output of one experiment goes to Kartavya/outputs/<category>/pca/<experiment>/."""
    cats = ["models", "reduced_data", "reconstructed_data", "metrics", "graphs", "logs"]
    return {c: OUTPUT_ROOT / c / ALGORITHM / experiment for c in cats}


# ------------------------------------------------------------------ PNG experiment
# Folder order = channel order in the feature matrix. Do not reorder between runs.
PNG_FOLDERS = {
    "RGB":         {"kind": "composite", "channels": ["R(B04)", "G(B03)", "B(B02)"]},
    "False_Color": {"kind": "composite", "channels": ["R(B08)", "G(B04)", "B(B03)"]},
    "NDVI":        {"kind": "colormap", "cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0, "channel": "NDVI"},
    "NDWI":        {"kind": "colormap", "cmap": "RdBu",   "vmin": -1.0, "vmax": 1.0, "channel": "NDWI"},
}
EXPECTED_IMAGES_PER_FOLDER = 5

# Fixed physical ranges used to scale every channel to [0, 1] (same for every date).
PNG_NORMALIZATION = [("RGB:", 0.0, 255.0), ("False_Color:", 0.0, 255.0),
                     ("NDVI", -1.0, 1.0), ("NDWI", -1.0, 1.0)]

# The figures are ~2800 px wide; all panels are resampled to this common width (height keeps the
# aspect ratio). 1024 keeps a run to about a minute; use 2048 for more detail (4x the work).
TARGET_WIDTH = 1024

# ------------------------------------------------------------------ raw experiment
RAW_BANDS = ["B02", "B03", "B04", "B08"]
RAW_SCALE = 10000.0                     # stored as reflectance x 10000
RAW_NORMALIZATION = [("B", 0.0, 1.2)]   # TOA reflectance; bright clouds reach ~1.13 in this data
RAW_DOWNSAMPLE = 3                      # average 3x3 blocks (3300 px -> 1100 px); 1 = full resolution

# ------------------------------------------------------------------ train/test split
# Indices refer to the sorted filename order (see filename_index_mapping.json).
# None = automatic: with >= 2 dates the first 60 % of indices train and the rest test
# (5 dates -> train [0, 1, 2], test [3, 4]); with 1 date the image is split spatially:
# the top half trains and the bottom half tests, so test pixels are never seen in training.
TRAIN_INDICES = None
TEST_INDICES = None
TRAIN_FRACTION = 0.6
SPATIAL_TRAIN_FRACTION = 0.5

# ------------------------------------------------------------------ PCA settings
STANDARDIZE = True            # z-score each channel with training mean/std before PCA
MAX_TRAIN_PIXELS = 1_000_000  # random pixel sample used to fit PCA (None = all valid pixels)
RANDOM_SEED = 42
COMPONENT_COUNTS = None       # None = every K from 1 to the number of channels
SAVE_RECONSTRUCTED_K = None   # K values whose full reconstructed arrays are saved (None = all)
GALLERY_K = [1, 2, 3, 4, 6]   # K values shown in the reconstruction picture (filtered to valid K)
VARIANCE_THRESHOLDS = (0.90, 0.95, 0.99)
TIMING_REPEATS = 5            # forward/inverse transforms are timed this many times; the median is reported
