"""
All settings for the UBS-CNN (BS-Net-Conv) channel-selection experiments.
Change values here, not inside train.py / test.py.

What is being selected
  The 8 channels built from the processed PNG products (see PNG_FOLDERS):
    RGB:R(B04), RGB:G(B03), RGB:B(B02)                 -> 8-bit DISPLAY levels of a true-colour composite
    False_Color:R(B08), False_Color:G(B04), False_Color:B(B03) -> display levels of a false-colour composite
    NDVI, NDWI                                         -> derived indices decoded from colour-mapped PNGs
  These are NOT original Sentinel-2 bands: they are stretched 8-bit renderings and derived indices.
  The B-numbers in the names only say which band each display channel was rendered from.
  "raw" experiment: original B02/B03/B04/B08 from data/raw/sentinel2_l1c/*.nc, only if present.

The dataset settings are identical to the PCA/ICA/LDA experiments so the results can be compared.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[4]
KARTAVYA_ROOT = REPO_ROOT / "Kartavya"
OUTPUT_ROOT = KARTAVYA_ROOT / "outputs"
PNG_DATASET_DIR = REPO_ROOT / "data" / "processed" / "sentinel2"
RAW_DATASET_DIR = REPO_ROOT / "data" / "raw" / "sentinel2_l1c"

ALGORITHM = "ubscnn"


def output_dirs(experiment):
    """Every output goes to Kartavya/outputs/<category>/ubscnn/<experiment>/."""
    cats = ["models", "reduced_data", "reconstructed_data", "metrics", "graphs", "logs"]
    return {c: OUTPUT_ROOT / c / ALGORITHM / experiment for c in cats}


# ------------------------------------------------------------------ PNG experiment (same as PCA/ICA)
PNG_FOLDERS = {   # folder order = channel order = input channel order of the CNN; do not reorder
    "RGB":         {"kind": "composite", "channels": ["R(B04)", "G(B03)", "B(B02)"]},
    "False_Color": {"kind": "composite", "channels": ["R(B08)", "G(B04)", "B(B03)"]},
    "NDVI":        {"kind": "colormap", "cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0, "channel": "NDVI"},
    "NDWI":        {"kind": "colormap", "cmap": "RdBu",   "vmin": -1.0, "vmax": 1.0, "channel": "NDWI"},
}
EXPECTED_IMAGES_PER_FOLDER = 5
PNG_NORMALIZATION = [("RGB:", 0.0, 255.0), ("False_Color:", 0.0, 255.0),
                     ("NDVI", -1.0, 1.0), ("NDWI", -1.0, 1.0)]
TARGET_WIDTH = 1024

# ------------------------------------------------------------------ raw experiment
RAW_BANDS = ["B02", "B03", "B04", "B08"]
RAW_SCALE = 10000.0
RAW_NORMALIZATION = [("B", 0.0, 1.2)]
RAW_DOWNSAMPLE = 3

# ------------------------------------------------------------------ train/test split
# None = automatic: >= 2 dates -> first 60 % of indices train, the rest test (5 dates -> [0,1,2] / [3,4]);
# 1 date -> top half of the image trains, bottom half tests.
TRAIN_INDICES = None
TEST_INDICES = None
TRAIN_FRACTION = 0.6
SPATIAL_TRAIN_FRACTION = 0.5

# ------------------------------------------------------------------ data used inside training
RANDOM_SEED = 42
PATCH_SIZE = 16              # a x a patches, as in the authors' code (view_as_windows(..., (16, 16, n_band)))
PATCH_STRIDE = 16            # DEVIATION: the authors use stride 1 (~500k overlapping patches here); 16 gives
                             # non-overlapping patches that still cover the whole training region and make
                             # CPU training feasible (~2,100 patches, ~5 min per 100 epochs)
VAL_PATCH_FRACTION = 0.1     # training patches held out to monitor over-fitting (still TRAINING region)
MAX_TRAIN_PIXELS = 400_000   # pixel sample for the linear reconstruction model / objective (as GA/PSO/ACO/SA)
VALIDATION_FRACTION = 0.3

# ------------------------------------------------------------------ BS-Net-Conv hyper-parameters
L1_LAMBDA = 0.01             # paper Table II and code: l1_regularizer(0.01) on the band weights
LEARNING_RATE = 1e-4         # authors' code (BS_Net_Conv.py main block); the paper's Table II lists 2e-3
BATCH_SIZE = 32              # authors' code
EPOCHS = 100                 # paper: "maximum iterations 100"; code: EPOCH = 100 (full passes over the patches)
SEEDS = [42, 43, 44]         # independent trainings; band weights are averaged over them (consensus ranking)
TORCH_THREADS = None         # None = PyTorch default

# ------------------------------------------------------------------ selection
FEATURE_COUNTS = None        # K values: None = 1 .. C-1 (top-K of the consensus ranking)

# ------------------------------------------------------------------ comparison (evaluation only, never used to select)
# Objective of the GA/PSO/ACO/SA experiments, used ONLY to score the CNN's choice against the exhaustive optimum
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1
RUN_EXHAUSTIVE_CHECK = True

# ------------------------------------------------------------------ evaluation
REDUCED_DTYPE = "float32"    # storage type of the saved reduced data (same as PCA/ICA for a fair comparison)
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5           # reduce/reconstruct are timed this many times; the median is reported
