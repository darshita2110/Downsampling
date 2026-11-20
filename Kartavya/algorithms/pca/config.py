"""
PCA settings - values unchanged from the original. Dataset settings (paths, channel names, value ranges,
split) are detected from the data (common/data_loader.py) and shared by every algorithm (common/config.py).
"""
STANDARDIZE = True            # z-score each channel with training mean/std before PCA
                              # (False = PCA on the [0,1] values directly; results go to outputs/<id>/pca_nostd/)
MAX_TRAIN_PIXELS = 1_000_000  # random pixel sample used to fit PCA (None = all valid pixels)
RANDOM_SEED = 42
COMPONENT_COUNTS = None       # None = the common K list (plus K = C only if common/config.py INCLUDE_FULL_K = True)
SAVE_RECONSTRUCTED_K = None   # K values whose full reconstructed arrays are saved (None = all)
GALLERY_K = [1, 2, 3, 4, 6]   # K values shown in the reconstruction picture (filtered to valid K)
VARIANCE_THRESHOLDS = (0.90, 0.95, 0.99)
TIMING_REPEATS = 5            # forward/inverse transforms are timed this many times; the median is reported
SKLEARN_CHECK = True          # cross-check against scikit-learn's PCA (skipped if scikit-learn is missing)
