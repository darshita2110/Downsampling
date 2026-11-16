"""
Hyper-parameters of the Simulated Annealing (SA) channel selection - values unchanged from the original.
Dataset settings are detected from the data (common/data_loader.py, common/config.py).
"""
# ------------------------------------------------------------------ data used inside training
RANDOM_SEED = 42
MAX_TRAIN_PIXELS = 400_000
VALIDATION_FRACTION = 0.3

# ------------------------------------------------------------------ objective (identical to GA, PSO and ACO; SA maximises it)
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1

# ------------------------------------------------------------------ SA hyper-parameters
FEATURE_COUNTS = None        # K values. None = the common K list (1 .. C-1 for C <= 16)
MAX_ITERATIONS = 300         # termination: hard limit on proposed moves (one objective evaluation each)
COOLING_RATE = 0.95          # geometric cooling: T <- 0.95 * T after every move
INITIAL_ACCEPT_PROB = 0.8    # T0 is set so that an average WORSE move is accepted with this probability
T0_SAMPLES = 20              # random (solution, neighbour) pairs used to estimate T0
T_MIN_RATIO = 1e-4           # termination: "frozen" when T < 1e-4 * T0 (about 180 moves at rate 0.95)
PATIENCE = 100               # termination: best solution unchanged for this many moves
SA_SEEDS = [42, 43, 44, 45, 46]   # independent runs per K (different start points); best run is kept
RUN_EXHAUSTIVE_CHECK = True

# ------------------------------------------------------------------ evaluation
REDUCED_DTYPE = "float32"
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5
