"""
All settings for the Genetic Algorithm (GA) channel-selection experiments.
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

ALGORITHM = "ga"


def output_dirs(experiment):
    """Every output goes to Kartavya/outputs/<category>/ga/<experiment>/."""
    cats = ["models", "reduced_data", "reconstructed_data", "metrics", "graphs", "logs"]
    return {c: OUTPUT_ROOT / c / ALGORITHM / experiment for c in cats}


# ------------------------------------------------------------------ PNG experiment (same as PCA/ICA)
PNG_FOLDERS = {   # folder order = channel order = gene order in the chromosome; do not reorder
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
                             # (the reconstruction model is fitted on the rest), so the GA never
                             # rewards a subset for fitting noise. Test pixels are never used.

# ------------------------------------------------------------------ fitness function
# fitness(S) = mean_R2(S) + W_MIN_R2 * min_R2(S) - W_REDUNDANCY * redundancy(S)
#   mean_R2    : information preservation - share of every channel's variance that a linear model
#                can rebuild from the selected channels (averaged over all C channels, so each
#                channel counts equally whatever its units)
#   min_R2     : reconstruction quality of the WORST channel, so no product (e.g. NDWI) is silently lost
#   redundancy : mean |Pearson r| between the selected channels (0 when only one is selected);
#                two copies of the same information are penalised
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1

# ------------------------------------------------------------------ GA hyper-parameters
FEATURE_COUNTS = None        # K values (channels to keep). None = 1 .. C-1 (keeping all C is not a reduction)
POPULATION_SIZE = 12
MAX_GENERATIONS = 40         # termination 1: hard limit
PATIENCE = 10                # termination 2: stop when the best fitness has not improved for this many generations
TOURNAMENT_SIZE = 3          # selection: best of 3 random individuals becomes a parent
CROSSOVER_RATE = 0.9         # probability that two parents are recombined (otherwise copied)
MUTATION_RATE = 0.2          # probability of a swap mutation (one kept channel <-> one dropped channel)
ELITISM = 2                  # best individuals copied unchanged into the next generation
GA_SEEDS = [42, 43, 44, 45, 46]   # independent GA runs per K (robustness check); best run is kept
RUN_EXHAUSTIVE_CHECK = True  # also score every C-choose-K subset to verify the GA found the optimum
                             # (feasible here: at most 70 subsets for 8 channels)

# ------------------------------------------------------------------ evaluation
REDUCED_DTYPE = "float32"    # storage type of the saved reduced data (same as PCA/ICA for a fair comparison)
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5           # reduce/reconstruct are timed this many times; the median is reported
