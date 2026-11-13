"""
All settings for the Ant Colony Optimization (ACO) channel-selection experiments.
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

ALGORITHM = "aco"


def output_dirs(experiment):
    """Every output goes to Kartavya/outputs/<category>/aco/<experiment>/."""
    cats = ["models", "reduced_data", "reconstructed_data", "metrics", "graphs", "logs"]
    return {c: OUTPUT_ROOT / c / ALGORITHM / experiment for c in cats}


# ------------------------------------------------------------------ PNG experiment (same as PCA/ICA)
PNG_FOLDERS = {   # folder order = channel order = pheromone vector order; do not reorder
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
MAX_TRAIN_PIXELS = 400_000   # random sample of TRAINING pixels used by the fitness function
VALIDATION_FRACTION = 0.3    # part of that sample held out INSIDE training to score each subset
                             # (the reconstruction model is fitted on the rest), so ACO never
                             # rewards a subset for fitting noise. Test pixels are never used.

# ------------------------------------------------------------------ fitness function (identical to the GA and PSO)
# fitness(S) = mean_R2(S) + W_MIN_R2 * min_R2(S) - W_REDUNDANCY * redundancy(S)
#   mean_R2    : information preservation - share of every channel's variance that a linear model
#                can rebuild from the selected channels (averaged over all C channels, so each
#                channel counts equally whatever its units)
#   min_R2     : reconstruction quality of the WORST channel, so no product (e.g. NDWI) is silently lost
#   redundancy : mean |Pearson r| between the selected channels (0 when only one is selected);
#                two copies of the same information are penalised
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1

# ------------------------------------------------------------------ ACO hyper-parameters
FEATURE_COUNTS = None        # K values (channels to keep). None = 1 .. C-1
N_ANTS = 12                  # ants per iteration (same budget per step as the GA population / PSO swarm)
MAX_ITERATIONS = 40          # termination 1: hard limit
PATIENCE = 10                # termination 2: best-so-far unchanged for this many iterations
ALPHA = 1.0                  # weight of pheromone (learned memory) in the selection probability
BETA = 2.0                   # weight of the heuristic (prior: information content, non-redundancy)
RHO = 0.2                    # evaporation rate: 20 % of the pheromone disappears every iteration
Q = 1.0                      # deposit = Q * fitness on each channel of the iteration-best and best-so-far subset
TAU_INIT = 1.0               # initial pheromone, equal on every channel (no preference yet)
TAU_MIN, TAU_MAX = 0.1, 5.0  # MAX-MIN bounds: keep every channel selectable, stop one from dominating
HEURISTIC_EPS = 1e-3         # floor so that no channel's heuristic is exactly 0
ACO_SEEDS = [42, 43, 44, 45, 46]  # independent colonies per K; best run is kept
RUN_EXHAUSTIVE_CHECK = True  # also score every C-choose-K subset to verify ACO found the optimum
                             # (feasible here: at most 70 subsets for 8 channels)

# ------------------------------------------------------------------ evaluation
REDUCED_DTYPE = "float32"    # storage type of the saved reduced data (same as PCA/ICA for a fair comparison)
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5           # reduce/reconstruct are timed this many times; the median is reported
