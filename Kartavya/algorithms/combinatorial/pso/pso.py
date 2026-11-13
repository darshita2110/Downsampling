"""
Particle Swarm Optimization (PSO) for selecting K of C image channels, written from scratch with NumPy.

The fitness function and the linear reconstruction model are the SAME as the GA's
(Kartavya/algorithms/combinatorial/ga/ga.py: SubsetFitness, LinearReconstructor), so GA and PSO are
compared on identical terms. Only the search strategy differs.

Particle encoding ("top-K decoding")
  Each particle has a continuous position x in [0, 1]^C: one "preference score" per channel.
  The subset it stands for is the K channels with the highest scores:
      x = [0.1 0.7 0.2 0.9 0.3 0.4 0.0 0.8], K = 3  ->  keep channels 3, 7, 1
  Standard PSO moves continuous points, and this decoding turns any point into a valid K-subset,
  so no repair step is needed and the classic update equations can be used unchanged.

Update equations (per particle i, per dimension d, per iteration t)
  v <- w(t) * v  +  c1 * r1 * (pbest_i - x)  +  c2 * r2 * (gbest - x)       r1, r2 ~ U(0, 1)
       inertia      cognitive: pull to own best    social: pull to swarm best
  v <- clip(v, -v_max, v_max)        (stops particles from flying off)
  x <- clip(x + v, 0, 1)
  w(t) falls linearly from w_start to w_end: explore early, refine late.
"""
from math import comb

import numpy as np


def decode(position, k):
    """Continuous position -> binary mask with exactly k ones (the k largest scores)."""
    mask = np.zeros(len(position), dtype=np.int8)
    mask[np.argsort(-position, kind="stable")[:k]] = 1
    return mask


class ChannelSubsetPSO:
    def __init__(self, n_features, k, fitness, n_particles=12, max_iterations=40, patience=10,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, v_max=0.5, seed=42):
        if not 1 <= k <= n_features:
            raise ValueError(f"k must be in 1..{n_features}, got {k}")
        self.C, self.k, self.fitness = n_features, k, fitness
        self.n, self.max_iter, self.patience = n_particles, max_iterations, patience
        self.w_start, self.w_end, self.c1, self.c2, self.v_max = w_start, w_end, c1, c2, v_max
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.search_space = comb(n_features, k)

    def inertia(self, t):
        return self.w_start - (self.w_start - self.w_end) * t / max(self.max_iter, 1)

    def run(self):
        rng = self.rng
        # ---------------------------------------------------- initialisation
        x = rng.random((self.n, self.C))                                    # positions in [0, 1]
        v = rng.uniform(-self.v_max, self.v_max, (self.n, self.C))          # velocities
        masks = [decode(p, self.k) for p in x]
        fit = np.array([self.fitness(m) for m in masks])
        evaluated = {m.tobytes() for m in masks}
        pbest_x, pbest_f = x.copy(), fit.copy()                             # personal bests
        g = int(np.argmax(pbest_f))
        gbest_x, gbest_f = pbest_x[g].copy(), float(pbest_f[g])             # global best
        history = [self._record(0, fit, masks, gbest_x, gbest_f, len(evaluated), v)]
        stall, stop_reason = 0, "max_iterations"

        for t in range(1, self.max_iter + 1):
            if len(evaluated) == self.search_space:
                stop_reason = "every possible subset evaluated"
                break
            # ---------------------------------------------------- velocity and position update
            r1, r2 = rng.random((self.n, self.C)), rng.random((self.n, self.C))
            v = (self.inertia(t) * v + self.c1 * r1 * (pbest_x - x) + self.c2 * r2 * (gbest_x - x))
            v = np.clip(v, -self.v_max, self.v_max)
            x = np.clip(x + v, 0.0, 1.0)
            # ---------------------------------------------------- evaluate and update bests
            masks = [decode(p, self.k) for p in x]
            fit = np.array([self.fitness(m) for m in masks])
            evaluated.update(m.tobytes() for m in masks)
            better = fit > pbest_f
            pbest_x[better], pbest_f[better] = x[better], fit[better]
            g = int(np.argmax(pbest_f))
            if pbest_f[g] > gbest_f + 1e-12:
                gbest_x, gbest_f, stall = pbest_x[g].copy(), float(pbest_f[g]), 0
            else:
                stall += 1
            history.append(self._record(t, fit, masks, gbest_x, gbest_f, len(evaluated), v))
            if stall >= self.patience:
                stop_reason = f"no improvement for {self.patience} iterations"
                break

        self.best_ = decode(gbest_x, self.k)
        self.best_position_ = gbest_x
        self.best_fitness_ = gbest_f
        self.history_ = history
        self.iterations_ = history[-1]["iteration"]
        self.stop_reason_ = stop_reason
        self.n_evaluations_ = len(evaluated)
        return self

    def _record(self, t, fit, masks, gbest_x, gbest_f, n_eval, v):
        return {"iteration": t, "inertia": self.inertia(t), "gbest": gbest_f,
                "iter_best": float(fit.max()), "iter_mean": float(fit.mean()), "iter_worst": float(fit.min()),
                "unique_subsets_in_swarm": len({m.tobytes() for m in masks}),
                "mean_abs_velocity": float(np.abs(v).mean()), "evaluations_cumulative": n_eval,
                "gbest_selected_idx": np.flatnonzero(decode(gbest_x, self.k)).tolist()}
