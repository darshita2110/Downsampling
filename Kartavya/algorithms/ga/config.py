"""
Hyper-parameters of the Genetic Algorithm (GA) channel selection - values unchanged from the original.
Dataset settings (paths, channel names, ranges, split) are no longer here: they are detected from the
data (common/data_loader.py) and shared by every algorithm (common/config.py).
"""
# ------------------------------------------------------------------ data used inside training
RANDOM_SEED = 42
MAX_TRAIN_PIXELS = 400_000   # random sample of TRAINING pixels used by the fitness function
VALIDATION_FRACTION = 0.3    # part of that sample held out INSIDE training to score each subset
                             # (the reconstruction model is fitted on the rest). Test pixels are never used.

# ------------------------------------------------------------------ fitness function
# fitness(S) = mean_R2(S) + W_MIN_R2 * min_R2(S) - W_REDUNDANCY * redundancy(S)
#   mean_R2    : information preservation - share of every channel's variance that a linear model
#                can rebuild from the selected channels (averaged over all C channels)
#   min_R2     : reconstruction quality of the WORST channel, so no channel is silently lost
#   redundancy : mean |Pearson r| between the selected channels (0 when only one is selected)
W_MIN_R2 = 0.5
W_REDUNDANCY = 0.1

# ------------------------------------------------------------------ GA hyper-parameters
FEATURE_COUNTS = None        # K values (channels to keep). None = the common K list (1 .. C-1 for C <= 16)
POPULATION_SIZE = 12
MAX_GENERATIONS = 40         # termination 1: hard limit
PATIENCE = 10                # termination 2: best fitness not improved for this many generations
TOURNAMENT_SIZE = 3          # selection: best of 3 random individuals becomes a parent
CROSSOVER_RATE = 0.9         # probability that two parents are recombined (otherwise copied)
MUTATION_RATE = 0.2          # probability of a swap mutation (one kept channel <-> one dropped channel)
ELITISM = 2                  # best individuals copied unchanged into the next generation
GA_SEEDS = [42, 43, 44, 45, 46]   # independent GA runs per K (robustness check); best run is kept
RUN_EXHAUSTIVE_CHECK = True  # also score every C-choose-K subset (only if <= EXHAUSTIVE_MAX_SUBSETS)

# ------------------------------------------------------------------ evaluation
REDUCED_DTYPE = "float32"    # storage type of the saved reduced data (same for every algorithm)
GALLERY_K = [1, 2, 3, 4, 6]
TIMING_REPEATS = 5           # reduce/reconstruct are timed this many times; the median is reported
