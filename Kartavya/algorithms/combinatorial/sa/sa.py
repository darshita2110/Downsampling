"""
Simulated Annealing (SA) for selecting K of C image channels, written from scratch with NumPy.

The objective (fitness) and the linear reconstruction model are the SAME as the GA's, PSO's and ACO's
(Kartavya/algorithms/combinatorial/ga/ga.py), so all four searches are compared on identical terms.
SA MAXIMISES the objective.

The idea
  Annealing metal: heated metal atoms move freely; cooled slowly, they settle into a low-energy
  crystal. SA keeps ONE current solution and repeatedly tries a small random change (a neighbour):
    better neighbour  -> always accepted
    worse neighbour   -> accepted with probability exp(delta / T)     (delta < 0 = how much worse)
  At high temperature T almost any move is accepted (exploration, escaping local optima);
  as T falls, worse moves become rare and the search settles (exploitation).

Pieces
  solution     : binary mask with exactly K ones (the kept channels)
  neighbour    : swap one kept channel for one dropped channel (stays a valid K-subset)
  T0           : chosen from the data: sample random neighbour moves from random subsets, take the mean
                 size of the WORSENING moves |delta|, and set T0 so that such a move is accepted with
                 probability P0 (default 0.8):  T0 = -mean|delta_worse| / ln(P0)
  cooling      : geometric, T_{t+1} = cooling_rate * T_t
  best-so-far  : the best subset ever EVALUATED (accepted or not) is kept and returned
  termination  : (a) T below t_min_ratio * T0 ("frozen"), (b) best not improved for `patience`
                 iterations, (c) max_iterations, or (d) every possible subset evaluated
"""
from math import comb, exp, log

import numpy as np


class ChannelSubsetSA:
    def __init__(self, n_features, k, objective, max_iterations=300, cooling_rate=0.95, initial_accept_prob=0.8,
                 t0_samples=20, t_min_ratio=1e-4, patience=100, seed=42):
        if not 1 <= k <= n_features:
            raise ValueError(f"k must be in 1..{n_features}, got {k}")
        self.C, self.k, self.objective = n_features, k, objective
        self.max_iter, self.alpha, self.p0 = max_iterations, cooling_rate, initial_accept_prob
        self.t0_samples, self.t_min_ratio, self.patience = t0_samples, t_min_ratio, patience
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.search_space = comb(n_features, k)
        self.evaluated = set()
        self._seen_best, self._seen_best_f = None, -np.inf

    # ------------------------------------------------------------------ building blocks
    def _score(self, mask):
        """Evaluate a subset and remember the best one EVER evaluated (including rejected candidates and the
        T0-estimation samples): a rejected move can still be the best subset seen, and losing it would waste
        the evaluation."""
        self.evaluated.add(mask.tobytes())
        f = self.objective(mask)
        if f > self._seen_best_f + 1e-12:
            self._seen_best, self._seen_best_f = mask.copy(), f
        return f

    def random_solution(self):
        mask = np.zeros(self.C, dtype=np.int8)
        mask[self.rng.choice(self.C, self.k, replace=False)] = 1
        return mask

    def neighbour(self, mask):
        """Swap one kept channel with one dropped channel. Identity if K = C (no neighbour exists)."""
        new = mask.copy()
        on, off = np.flatnonzero(mask == 1), np.flatnonzero(mask == 0)
        if len(off) == 0:
            return new
        new[self.rng.choice(on)] = 0
        new[self.rng.choice(off)] = 1
        return new

    def initial_temperature(self):
        """T0 such that an average worsening move is accepted with probability p0 at the start."""
        worse = []
        for _ in range(self.t0_samples):
            s = self.random_solution()
            d = self._score(self.neighbour(s)) - self._score(s)
            if d < 0:
                worse.append(-d)
        mean_worse = float(np.mean(worse)) if worse else 1e-3       # no worsening move seen: tiny T0
        return -mean_worse / log(self.p0), mean_worse

    # ------------------------------------------------------------------ main loop
    def run(self):
        t0, mean_worse = self.initial_temperature()
        T, t_min = t0, t0 * self.t_min_ratio
        current = self.random_solution()
        f_cur = self._score(current)
        best, f_best, stall = self._seen_best.copy(), self._seen_best_f, 0     # may come from the T0 samples
        history = [{"iteration": 0, "temperature": T, "current": f_cur, "best": f_best, "candidate": f_cur,
                    "delta": 0.0, "accept_prob": 1.0, "accepted": True, "evaluations_cumulative": len(self.evaluated),
                    "current_selected_idx": np.flatnonzero(current).tolist(),
                    "best_selected_idx": np.flatnonzero(best).tolist()}]
        stop_reason = "max_iterations"
        for it in range(1, self.max_iter + 1):
            cand = self.neighbour(current)
            f_cand = self._score(cand)
            delta = f_cand - f_cur
            # acceptance probability (Metropolis criterion, for MAXIMISATION)
            p = 1.0 if delta >= 0 else exp(delta / T)
            accepted = bool(self.rng.random() < p)
            if accepted:
                current, f_cur = cand, f_cand
            if self._seen_best_f > f_best + 1e-12:                    # best-ever evaluated, accepted or not
                best, f_best, stall = self._seen_best.copy(), self._seen_best_f, 0
            else:
                stall += 1
            history.append({"iteration": it, "temperature": T, "current": f_cur, "best": f_best,
                            "candidate": f_cand, "delta": delta, "accept_prob": p, "accepted": accepted,
                            "evaluations_cumulative": len(self.evaluated),
                            "current_selected_idx": np.flatnonzero(current).tolist(),
                            "best_selected_idx": np.flatnonzero(best).tolist()})
            T *= self.alpha                                              # geometric cooling
            if len(self.evaluated) == self.search_space:
                stop_reason = "every possible subset evaluated"
                break
            if T < t_min:
                stop_reason = f"frozen (T < {self.t_min_ratio:g} * T0)"
                break
            if stall >= self.patience:
                stop_reason = f"best unchanged for {self.patience} iterations"
                break
        self.best_, self.best_fitness_ = best, f_best
        self.t0_, self.mean_worse_delta_, self.final_temperature_ = t0, mean_worse, T
        self.history_, self.iterations_ = history, history[-1]["iteration"]
        self.stop_reason_, self.n_evaluations_ = stop_reason, len(self.evaluated)
        self.acceptance_rate_ = float(np.mean([h["accepted"] for h in history[1:]])) if len(history) > 1 else 1.0
        return self
