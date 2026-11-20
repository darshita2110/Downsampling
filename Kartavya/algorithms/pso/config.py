"""
Hyper-parameters of the Particle Swarm Optimization (PSO) channel selection - values unchanged from the
original. Dataset settings are detected from the data (common/data_loader.py, common/config.py).
"""
# ------------------------------------------------------------------ data used inside training
RANDOM_SEED = 42
MAX_TRAIN_PIXELS = 400_000   # random sample of TRAINING pixels used by the fitness function
VALIDATION_FRACTION = 0.3    # held out INSIDE training to score each subset. Test pixels are never used.

# ------------------------------------------------------------------ fitness function (identical to the GA)
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1

# ------------------------------------------------------------------ PSO hyper-parameters
FEATURE_COUNTS = None        # K values. None = the common K list (1 .. C-1 for C <= 16)
N_PARTICLES = 12             # swarm size (same budget per step as the GA population)
MAX_ITERATIONS = 40          # termination 1: hard limit
PATIENCE = 10                # termination 2: global best unchanged for this many iterations
W_START, W_END = 0.9, 0.4    # inertia weight, decreased linearly (explore first, then refine)
C1 = 1.5                     # cognitive coefficient: pull towards the particle's own best
C2 = 1.5                     # social coefficient: pull towards the swarm's best
V_MAX = 0.5                  # velocity clamp per dimension (positions live in [0, 1])
PSO_SEEDS = [42, 43, 44, 45, 46]  # independent swarms per K; best run is kept
RUN_EXHAUSTIVE_CHECK = True  # score every C-choose-K subset (only if <= EXHAUSTIVE_MAX_SUBSETS)

# ------------------------------------------------------------------ evaluation
REDUCED_DTYPE = "float32"
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5
