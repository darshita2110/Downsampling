"""
Hyper-parameters of the Ant Colony Optimization (ACO) channel selection - values unchanged from the
original. Dataset settings are detected from the data (common/data_loader.py, common/config.py).
"""
# ------------------------------------------------------------------ data used inside training
RANDOM_SEED = 42
MAX_TRAIN_PIXELS = 400_000
VALIDATION_FRACTION = 0.3

# ------------------------------------------------------------------ fitness function (identical to the GA and PSO)
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1

# ------------------------------------------------------------------ ACO hyper-parameters
FEATURE_COUNTS = None        # K values. None = the common K list (1 .. C-1 for C <= 16)
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
RUN_EXHAUSTIVE_CHECK = True

# ------------------------------------------------------------------ evaluation
REDUCED_DTYPE = "float32"
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5
