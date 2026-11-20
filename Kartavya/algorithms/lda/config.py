"""
LDA settings - values unchanged from the original. LDA is SUPERVISED: it needs a class label per training pixel.

Labels are found automatically next to the data (files whose name contains "gt", "label", "labels",
"groundtruth" ..., e.g. Indian_pines_gt.mat), or given explicitly with LABEL_PATH in main.py / --labels PATH.
A label raster must be a 2-D integer class map covering the same area as its image (it is resampled with
nearest-neighbour if its resolution differs; the aspect ratio is checked). IGNORE_LABEL marks unlabelled pixels.
Labels must come from an INDEPENDENT source (field data, an external land-cover map, manual annotation):
labels derived from these same pixels (thresholds, clustering) are circular and make the accuracy meaningless.
Without labels LDA is reported as "not applicable" - no labels are ever invented.
"""
IGNORE_LABEL = 0                 # pixels with this id are unlabelled and never used
MIN_PIXELS_PER_CLASS = 50        # in the TRAINING part; rarer classes make LDA's class means unreliable
CLASS_NAMES = {}                 # optional, e.g. {1: "water", 2: "vegetation"}

STANDARDIZE = True               # z-score with TRAINING statistics (keeps the shrinkage term meaningful)
SHRINKAGE = 1e-4                 # small ridge on S_W (duplicated channels make S_W singular)
MAX_TRAIN_PIXELS = 1_000_000
RANDOM_SEED = 42
COMPONENT_COUNTS = None          # None = the common K list limited to 1 .. min(n_classes - 1, n_channels)
TIMING_REPEATS = 5
