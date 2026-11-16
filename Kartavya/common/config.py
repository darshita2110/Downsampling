"""
Dataset-level settings shared by EVERY algorithm.

These used to be copied into each algorithm's config.py together with hard-coded paths, folder names,
channel names and value ranges of one particular dataset. Everything dataset-specific is now detected
from the data itself (see data_loader.py); what remains here are the experimental-protocol settings that
must be identical for all algorithms so that their results can be compared fairly.

You normally do not need to edit this file: the dataset path is the only required input, and it is given
in an algorithm's main.py, in run_all_algorithms.py, or on the command line.
"""
from pathlib import Path

KARTAVYA_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = KARTAVYA_ROOT / "outputs"          # outputs/<dataset_id>/<algorithm>/...

# ------------------------------------------------------------------ train / test split
# Indices refer to the sorted sample order written to dataset/filename_index_mapping.json.
# None = automatic (the rule of the original repository):
#   >= 2 samples : the first TRAIN_FRACTION of the samples train, the rest test (5 -> [0,1,2] / [3,4])
#   1 sample     : the image is split spatially: the top SPATIAL_TRAIN_FRACTION of the rows trains and the
#                  bottom part tests, so test pixels are never seen during training.
TRAIN_INDICES = None
TEST_INDICES = None
TRAIN_FRACTION = 0.6
SPATIAL_TRAIN_FRACTION = 0.5

# ------------------------------------------------------------------ normalisation to [0, 1]
# All metrics (PSNR with MAX = 1, RMSE "fraction of full range", SSIM data_range = 1) are defined on [0, 1].
# "auto"   : 8-bit integer data -> fixed range 0..255 (exactly what the original PNG experiment did);
#            any other type      -> per-channel min/max of the TRAINING pixels (fixed for every sample, so a
#                                   model learnt on training data applies unchanged to test data).
# "dtype"  : integer data -> full range of the integer type; float data -> training min/max.
# "minmax" : per-channel training min/max for every type.
# Test values outside the training range are clipped to [0, 1] (as in the original implementation).
NORMALIZATION = "auto"

# ------------------------------------------------------------------ size limits (memory safety only)
# None = use the data at its native resolution (default; nothing is resampled).
# An integer N = area-average images whose longer side exceeds N down to N pixels (keeps the aspect ratio),
# like the original TARGET_WIDTH option. Useful for very large GeoTIFF scenes.
MAX_IMAGE_SIDE = None
# The training feature matrix is built from at most this many valid pixels (random, seeded) per training
# sample. Every algorithm then draws its own MAX_TRAIN_PIXELS sample from it, exactly as before.
TRAIN_PIXEL_POOL_PER_SAMPLE = 2_000_000
# Loaded images are kept in memory up to this many bytes so that algorithms run back-to-back do not re-read
# the files; above it, files are re-read on demand.
CACHE_LIMIT_BYTES = 2_000_000_000

RANDOM_SEED = 42

# ------------------------------------------------------------------ numbers of retained channels K
# Number of channels C <= MAX_C_FOR_ALL_K: every K = 1 .. C-1 is evaluated (the original behaviour).
# Larger C (hyperspectral data): a fixed grid of K values below C is used, so that a 200-band cube does not
# need 199 separate searches per algorithm. All algorithms receive the SAME K list (fair comparison).
MAX_C_FOR_ALL_K = 16
K_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 64, 75, 100, 128, 150, 200, 256]
MAX_K_VALUES = 12                  # at most this many values from K_GRID (the smallest ones are kept)

# ------------------------------------------------------------------ exhaustive checks (band selection)
# GA/PSO/ACO/SA/UBS-CNN verify their result against an exhaustive search of all C-choose-K subsets.
# That is only feasible for small search spaces (the original data had at most 70); larger ones are skipped
# and reported as "not run" rather than silently approximated.
EXHAUSTIVE_MAX_SUBSETS = 500

# ------------------------------------------------------------------ labels (supervised LDA only)
# Label rasters are found automatically next to the data: files whose name contains one of these words are
# treated as labels (and never as image data), e.g. "Indian_pines_gt.mat", "scene_labels.tif".
LABEL_KEYWORDS = ["gt", "groundtruth", "ground_truth", "label", "labels", "annotation", "classes"]
IGNORE_LABEL = 0                   # label id meaning "unlabelled"
