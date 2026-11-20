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

# ------------------------------------------------------------------ number of bands (band-axis check)
# A 3-D array can arrive as (H, W, C), (C, H, W) or (H, C, W). Guessing "the smallest axis is the band axis" is
# WRONG for hyperspectral cubes whose band count exceeds a side of the image: Indian Pines (145 x 145 x 200)
# was read as 145 x 200 with 145 "channels". The band axis is therefore resolved as follows:
#   1. EXPECTED_BANDS (or --bands N on the command line), if set: the axis of that length is the band axis.
#   2. else the file name is looked up in KNOWN_BAND_COUNTS (standard benchmark cubes).
#   3. else the smallest-axis rule is used, BUT loading stops with an error when the result is doubtful
#      (>= AMBIGUOUS_MIN_BANDS channels and more than AMBIGUOUS_RATIO x the shorter image side); set
#      EXPECTED_BANDS / --bands to confirm the band count in that case.
EXPECTED_BANDS = None
KNOWN_BAND_COUNTS = {            # key = file stem in lower case without "_", "-", " "; longest matching prefix wins
    "indianpinescorrected": 200, "indianpines": 220,
    "salinascorrected": 204, "salinas": 224, "salinasacorrected": 204, "salinasa": 224,
    "paviau": 103, "pavia": 102, "ksc": 176, "botswana": 145,
}
AMBIGUOUS_MIN_BANDS = 32
AMBIGUOUS_RATIO = 0.5

# ------------------------------------------------------------------ evaluation protocol
# "spatial" : (original behaviour) one image -> top half trains, bottom half tests; several images -> earlier
#             images train, later ones test. Measures how well a reduction learnt on one region transfers.
# "full"    : every pixel of every image is used to fit AND to evaluate (no held-out region). This is the
#             protocol of the team's other pipelines (e.g. Abhishek/universal_pipeline.py: PSNR on the whole
#             cube), so its PSNR/SSIM can be put in the same table. Set on the command line with --protocol.
# Results of the two protocols go to different folders (outputs/<dataset_id>/ vs outputs/<dataset_id>__full/)
# and every summary table carries a "protocol" column, so they are never mixed by accident.
PROTOCOL = "spatial"

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
# PCA / ICA can also be run at K = C (all channels kept: a lossless sanity check, PSNR ~ infinite). That single
# point squashes every K-axis graph (e.g. K = 200 next to K = 1..25 on Indian Pines) and is skipped by default.
# True = evaluate K = C again (only for checking that the reconstruction is exact).
INCLUDE_FULL_K = False

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
