"""
ICA hyper-parameters - values unchanged from the original. Dataset settings are detected from the data.
"""
STANDARDIZE = True          # z-score channels with TRAINING mean/std before ICA (False -> outputs/<id>/ica_nostd/)
MAX_TRAIN_PIXELS = 300_000  # random training-pixel sample used to fit (FastICA iterates over it)
RANDOM_SEED = 42            # pixel sampling and the random initial rotation
ICA_ALGORITHM = "parallel"  # all components estimated together (symmetric decorrelation)
ICA_FUN = "logcosh"         # contrast G(u) = log cosh(alpha u) / alpha, g(u) = tanh(alpha u)
ICA_FUN_ALPHA = 1.0
ICA_MAX_ITER = 1000
ICA_TOL = 1e-4
RANK_TOL = 1e-10            # eigenvalue / largest below this = zero variance -> not a usable component
# K values to fit. None = the common K list (plus K = C), limited to the numerical rank of the training data
# (ICA cannot whiten zero-variance directions, e.g. duplicated channels).
COMPONENT_COUNTS = None
SKLEARN_CHECK = True        # cross-check every K against scikit-learn's FastICA

SAVE_RECONSTRUCTED_K = None
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5
