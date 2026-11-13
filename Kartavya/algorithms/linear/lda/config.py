"""
Settings for the LDA experiments. LDA is SUPERVISED: it needs a class label for every training pixel.

Labels are NOT part of the dataset in this repository (see documentation/LDA.md). To run LDA on the
Raichur images, put one label raster per date in LABEL_DIR, named after LABEL_PATTERNS:
  * integer class id per pixel (.npy, or a single-band 8/16-bit .png), IGNORE_LABEL = unlabelled
  * covering exactly the same ground area as the images (any resolution; it is resampled with
    nearest-neighbour to the common grid, aspect ratio is checked)
  * from an INDEPENDENT source: field survey, an existing land-cover map (e.g. ESA WorldCover
    reprojected to this tile), or manual annotation - NOT thresholds of NDVI/NDWI or clusters of these
    same pixels (that would be circular).
Optional: CLASS_NAMES maps ids to names for the reports.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[4]
KARTAVYA_ROOT = REPO_ROOT / "Kartavya"
OUTPUT_ROOT = KARTAVYA_ROOT / "outputs"
PNG_DATASET_DIR = REPO_ROOT / "data" / "processed" / "sentinel2"
RAW_DATASET_DIR = REPO_ROOT / "data" / "raw" / "sentinel2_l1c"
ALGORITHM = "lda"


def output_dirs(experiment):
    cats = ["models", "reduced_data", "reconstructed_data", "metrics", "graphs", "logs"]
    return {c: OUTPUT_ROOT / c / ALGORITHM / experiment for c in cats}


# ------------------------------------------------------------------ dataset (same as PCA / ICA)
PNG_FOLDERS = {
    "RGB":         {"kind": "composite", "channels": ["R(B04)", "G(B03)", "B(B02)"]},
    "False_Color": {"kind": "composite", "channels": ["R(B08)", "G(B04)", "B(B03)"]},
    "NDVI":        {"kind": "colormap", "cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0, "channel": "NDVI"},
    "NDWI":        {"kind": "colormap", "cmap": "RdBu",   "vmin": -1.0, "vmax": 1.0, "channel": "NDWI"},
}
EXPECTED_IMAGES_PER_FOLDER = 5
PNG_NORMALIZATION = [("RGB:", 0.0, 255.0), ("False_Color:", 0.0, 255.0),
                     ("NDVI", -1.0, 1.0), ("NDWI", -1.0, 1.0)]
TARGET_WIDTH = 1024
RAW_BANDS = ["B02", "B03", "B04", "B08"]
RAW_SCALE = 10000.0
RAW_NORMALIZATION = [("B", 0.0, 1.2)]
RAW_DOWNSAMPLE = 3

# ------------------------------------------------------------------ labels
LABEL_DIR = REPO_ROOT / "data" / "labels" / "sentinel2"
LABEL_PATTERNS = ["raichur_labels_{date}.npy", "raichur_labels_{date}.png"]
IGNORE_LABEL = 0                 # pixels with this id are unlabelled and never used
MIN_PIXELS_PER_CLASS = 50        # in the TRAINING part; rarer classes make LDA's class means unreliable
CLASS_NAMES = {}                 # e.g. {1: "water", 2: "vegetation", 3: "built-up", 4: "bare soil"}

# ------------------------------------------------------------------ split (same rules as PCA / ICA)
TRAIN_INDICES = None
TEST_INDICES = None
TRAIN_FRACTION = 0.6
SPATIAL_TRAIN_FRACTION = 0.5

# ------------------------------------------------------------------ LDA hyper-parameters
STANDARDIZE = True               # z-score with TRAINING statistics (LDA itself is scale-invariant,
                                 # but this keeps the shrinkage term meaningful)
SHRINKAGE = 1e-4                 # small ridge on S_W: RGB:B04 = False_Color:B04 etc. make S_W singular
MAX_TRAIN_PIXELS = 1_000_000
RANDOM_SEED = 42
COMPONENT_COUNTS = None          # None = 1 .. min(n_classes - 1, n_channels); larger values are refused
TIMING_REPEATS = 5
