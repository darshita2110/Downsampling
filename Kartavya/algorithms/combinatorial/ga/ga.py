"""
Genetic Algorithm for selecting a subset of K image channels, written from scratch with NumPy.

Three pieces:
  LinearReconstructor  - rebuilds ALL C channels from the K kept channels with a least-squares
                         linear model (an intercept plus one weight per kept channel, per output channel).
                         This is what the receiver of the reduced data needs to get the full image back.
  SubsetFitness        - scores one subset: information preserved, worst-channel quality, redundancy.
  ChannelSubsetGA      - the evolutionary search: population, tournament selection, crossover,
                         swap mutation, elitism and two termination rules.

Chromosome encoding
  A binary vector of length C (one gene per channel). gene = 1 -> channel kept, 0 -> dropped.
  Example with the 8 PNG channels:  [0 0 1 1 0 0 0 1]  keeps RGB:B, False_Color:R and NDWI.
  Every experiment fixes the number of kept channels K, so every chromosome has exactly K ones.
  The operators below are designed so that this always stays true (no repair step is needed).
"""
from itertools import combinations

import numpy as np


# ============================================================================ reconstruction model
class LinearReconstructor:
    """
    x_all ~ [1, x_selected] @ W        (W has shape (K + 1) x C)

    Fitted by least squares on TRAINING pixels. For a kept channel the best weight is simply 1 on
    itself, so kept channels are reproduced exactly; dropped channels are predicted from the kept ones.
    """

    def __init__(self, selected):
        self.selected = np.asarray(sorted(selected), dtype=int)
        self.coef_ = None

    def _design(self, x):
        # x: (N, C) all channels -> (N, K + 1) with a leading column of ones (the intercept)
        return np.column_stack([np.ones(len(x), dtype=x.dtype), x[:, self.selected]])

    def fit(self, x):
        # lstsq copes with duplicated channels (rank-deficient design) by returning the minimum-norm solution
        self.coef_, *_ = np.linalg.lstsq(self._design(x).astype(np.float64), x.astype(np.float64), rcond=None)
        return self

    def reduce(self, x):
        """The reduced data are just the kept channels, unchanged (no new values are created)."""
        return x[:, self.selected]

    def reconstruct_from_reduced(self, r):
        """Full C-channel estimate from the K stored channels."""
        design = np.column_stack([np.ones(len(r), dtype=np.float64), r.astype(np.float64)])
        return (design @ self.coef_).astype(np.float32)

    def n_model_values(self):
        """Floats the receiver must store besides the reduced data: W plus the K channel indices."""
        return int(self.coef_.size + len(self.selected))

    def state(self):
        return {"selected": self.selected, "coef": self.coef_}

    @classmethod
    def from_state(cls, st):
        m = cls(st["selected"])
        m.coef_ = np.asarray(st["coef"], dtype=np.float64)
        return m


# ============================================================================ fitness
class SubsetFitness:
    """
    fitness(S) = mean_R2(S) + w_min * min_R2(S) - w_red * redundancy(S)

    R2_c(S) = 1 - SSE_c / SST_c for channel c, measured on VALIDATION pixels (held out from the
    pixels the reconstruction model is fitted on). It is the fraction of channel c's variance that
    survives the reduction S -> reconstruct. Being a ratio, it does not depend on the channel's units,
    so 8-bit display levels and NDVI/NDWI count equally.

    Results are cached: a subset is only ever evaluated once (the GA revisits subsets often).
    """

    def __init__(self, x_fit, x_val, names, w_min=0.5, w_red=0.1):
        self.x_fit, self.x_val, self.names = x_fit, x_val, list(names)
        self.w_min, self.w_red = w_min, w_red
        self.corr = np.corrcoef(x_fit.T)                    # channel correlations (fit pixels)
        self.sst = ((x_val - x_val.mean(0)) ** 2).sum(0)   # total variance per channel (validation)
        self.cache = {}

    def details(self, mask):
        """All terms of the fitness for one chromosome (cached)."""
        key = tuple(int(g) for g in mask)
        if key in self.cache:
            return self.cache[key]
        sel = np.flatnonzero(mask)
        rec = LinearReconstructor(sel).fit(self.x_fit)
        pred = rec.reconstruct_from_reduced(rec.reduce(self.x_val))
        sse = ((self.x_val - pred) ** 2).sum(0)
        r2 = 1.0 - sse / np.maximum(self.sst, 1e-12)
        if len(sel) > 1:
            sub = np.abs(self.corr[np.ix_(sel, sel)])
            redundancy = float(sub[np.triu_indices(len(sel), 1)].mean())
        else:
            redundancy = 0.0
        out = {"selected": [self.names[i] for i in sel], "selected_idx": sel.tolist(),
               "mean_R2": float(r2.mean()), "min_R2": float(r2.min()),
               "worst_channel": self.names[int(r2.argmin())], "redundancy": redundancy,
               "per_channel_R2": dict(zip(self.names, r2.astype(float).tolist()))}
        out["fitness"] = out["mean_R2"] + self.w_min * out["min_R2"] - self.w_red * redundancy
        self.cache[key] = out
        return out

    def __call__(self, mask):
        return self.details(mask)["fitness"]

    @property
    def n_evaluations(self):
        return len(self.cache)


def exhaustive_search(fitness, n_features, k):
    """Score every subset of size k (only feasible for few channels). Used to check the GA."""
    rows = []
    for combo in combinations(range(n_features), k):
        mask = np.zeros(n_features, dtype=np.int8)
        mask[list(combo)] = 1
        rows.append(fitness.details(mask))
    return sorted(rows, key=lambda r: -r["fitness"])


# ============================================================================ the genetic algorithm
class ChannelSubsetGA:
    """
    Evolve fixed-size channel subsets.

    One generation:
      1. evaluate   - fitness of every individual
      2. elitism    - the `elitism` best individuals go to the next generation unchanged
      3. selection  - tournament: pick `tournament_size` individuals at random, the fittest is a parent
      4. crossover  - with probability `crossover_rate`: keep the channels BOTH parents share, fill the
                      remaining K - |shared| places with random channels that ONE of the parents has
                      (so the child only inherits parental genes and always has exactly K)
      5. mutation   - with probability `mutation_rate`: swap one kept channel for one dropped channel
    Termination: `max_generations` reached, or the best fitness did not improve for `patience`
    generations, or every possible subset has already been evaluated (nothing left to discover).
    """

    def __init__(self, n_features, k, fitness, population_size=12, max_generations=40, patience=10,
                 tournament_size=3, crossover_rate=0.9, mutation_rate=0.2, elitism=2, seed=42):
        if not 1 <= k <= n_features:
            raise ValueError(f"k must be in 1..{n_features}, got {k}")
        self.C, self.k, self.fitness = n_features, k, fitness
        self.pop_size, self.max_gen, self.patience = population_size, max_generations, patience
        self.t_size, self.p_cx, self.p_mut, self.n_elite = tournament_size, crossover_rate, mutation_rate, elitism
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        from math import comb
        self.search_space = comb(n_features, k)

    # ------------------------------------------------------------------ operators
    def random_individual(self):
        mask = np.zeros(self.C, dtype=np.int8)
        mask[self.rng.choice(self.C, self.k, replace=False)] = 1
        return mask

    def initial_population(self):
        """Random K-subsets, distinct where the search space allows it (more initial diversity)."""
        pop, seen = [], set()
        target_unique = min(self.pop_size, self.search_space)
        while len(pop) < self.pop_size:
            ind = self.random_individual()
            key = ind.tobytes()
            if key in seen and len(seen) < target_unique:
                continue
            seen.add(key)
            pop.append(ind)
        return np.array(pop)

    def tournament(self, pop, fit):
        idx = self.rng.choice(len(pop), self.t_size, replace=False)
        return pop[idx[np.argmax(fit[idx])]]

    def crossover(self, a, b):
        if self.rng.random() >= self.p_cx:
            return a.copy()
        shared = np.flatnonzero(a & b)
        either = np.flatnonzero(a ^ b)                     # genes only one parent has
        fill = self.rng.choice(either, self.k - len(shared), replace=False) if len(either) else []
        child = np.zeros(self.C, dtype=np.int8)
        child[shared] = 1
        child[np.asarray(fill, dtype=int)] = 1
        return child

    def mutate(self, ind):
        ind = ind.copy()
        if self.rng.random() < self.p_mut and 0 < self.k < self.C:
            on, off = np.flatnonzero(ind == 1), np.flatnonzero(ind == 0)
            ind[self.rng.choice(on)] = 0
            ind[self.rng.choice(off)] = 1
        return ind

    # ------------------------------------------------------------------ main loop
    def run(self):
        pop = self.initial_population()
        history, best, best_fit, stall = [], None, -np.inf, 0
        evaluated = set()                                  # distinct subsets this run has scored
        stop_reason = "max_generations"
        for gen in range(self.max_gen + 1):                # generation 0 = initial population
            fit = np.array([self.fitness(ind) for ind in pop])
            evaluated.update(p.tobytes() for p in pop)
            order = np.argsort(-fit)
            if fit[order[0]] > best_fit + 1e-12:
                best, best_fit, stall = pop[order[0]].copy(), float(fit[order[0]]), 0
            else:
                stall += 1
            history.append({"generation": gen, "best_so_far": best_fit, "gen_best": float(fit.max()),
                            "gen_mean": float(fit.mean()), "gen_worst": float(fit.min()),
                            "unique_individuals": len({p.tobytes() for p in pop}),
                            "evaluations_cumulative": len(evaluated),
                            "best_selected_idx": np.flatnonzero(best).tolist()})
            if len(evaluated) == self.search_space:
                stop_reason = "every possible subset evaluated"
                break
            if stall >= self.patience:
                stop_reason = f"no improvement for {self.patience} generations"
                break
            if gen == self.max_gen:
                break
            # next generation: elites + children
            children = [pop[i].copy() for i in order[:self.n_elite]]
            while len(children) < self.pop_size:
                child = self.crossover(self.tournament(pop, fit), self.tournament(pop, fit))
                children.append(self.mutate(child))
            pop = np.array(children)
        self.best_ = best
        self.best_fitness_ = best_fit
        self.history_ = history
        self.generations_ = history[-1]["generation"]
        self.stop_reason_ = stop_reason
        self.n_evaluations_ = len(evaluated)
        return self
