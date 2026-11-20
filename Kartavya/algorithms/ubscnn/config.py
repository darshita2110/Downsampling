"""
UBS-CNN (BS-Net-Conv) settings - values unchanged from the original. Dataset settings are detected from the data.
"""
RANDOM_SEED = 42
PATCH_SIZE = 16              # a x a patches, as in the authors' code
PATCH_STRIDE = 16            # DEVIATION (as before): the authors use stride 1; 16 gives non-overlapping patches
                             # that cover the training region and keep CPU training feasible on large images
MIN_TRAIN_PATCHES = 512      # small images: the stride is lowered step by step (towards the authors' 1) until
                             # at least this many training patches exist
VAL_PATCH_FRACTION = 0.1     # training patches held out to monitor over-fitting (still TRAINING region)
MAX_TRAIN_PIXELS = 400_000   # pixel sample for the linear reconstruction model / objective (as GA/PSO/ACO/SA)
VALIDATION_FRACTION = 0.3

L1_LAMBDA = 0.01             # paper Table II and code: l1_regularizer(0.01) on the band weights
LEARNING_RATE = 1e-4         # authors' code (BS_Net_Conv.py main block)
BATCH_SIZE = 32              # authors' code
EPOCHS = 100                 # paper: "maximum iterations 100" (override: --epochs N)
SEEDS = [42, 43, 44]         # independent trainings; band weights are averaged over them (consensus ranking)
TORCH_THREADS = None         # None = PyTorch default

FEATURE_COUNTS = None        # K values: None = the common K list (top-K of the consensus ranking)

# Objective of GA/PSO/ACO/SA, used ONLY to score the CNN's choice against the exhaustive optimum
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1
RUN_EXHAUSTIVE_CHECK = True

REDUCED_DTYPE = "float32"
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5
