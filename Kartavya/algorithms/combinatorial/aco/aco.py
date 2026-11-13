"""
Ant Colony Optimization (ACO) for selecting K of C image channels, written from scratch with NumPy.

The fitness function and the linear reconstruction model are the SAME as the GA's and PSO's
(Kartavya/algorithms/combinatorial/ga/ga.py), so the three searches are compared on identical terms.

How a solution is built
  An ant builds one subset by picking channels one at a time, without replacement, until it has K.
  The probability of picking channel j next is

      P(j) = tau_j^alpha * eta_j^beta / sum over still-available channels of the same

  tau_j : PHEROMONE on channel j - learned memory: channels in good subsets get more of it
  eta_j : HEURISTIC desirability - prior knowledge that needs no fitness evaluation:
          eta_j = info_j * (1 - max r^2 between j and the channels this ant has already picked) + eps
          info_j = mean r^2 between channel j and every channel (how much of the image j "explains").
          The second factor makes a near-copy of an already chosen channel almost never picked next.
  alpha, beta weight learned memory against prior knowledge.

Pheromone update (after every ant of an iteration has built and been scored)
  evaporation : tau <- (1 - rho) * tau             (forget slowly, so bad early choices fade)
  deposit     : tau_j += q * fitness   for every channel j of the iteration-best subset and of the
                best-so-far subset (elitist ant)
  bounds      : tau kept in [tau_min, tau_max]   (MAX-MIN Ant System: no channel's probability can reach
                exactly 0 or swamp the rest, which prevents the colony freezing on one early answer)
"""
from math import comb

import numpy as np


class ChannelSubsetACO:
    def __init__(self, n_features, k, fitness, corr, n_ants=12, max_iterations=40, patience=10,
                 alpha=1.0, beta=2.0, rho=0.2, q=1.0, tau_init=1.0, tau_min=0.1, tau_max=5.0,
                 eps=1e-3, seed=42):
        if not 1 <= k <= n_features:
            raise ValueError(f"k must be in 1..{n_features}, got {k}")
        self.C, self.k, self.fitness = n_features, k, fitness
        self.r2 = np.asarray(corr, dtype=float) ** 2              # channel-to-channel r^2 (training pixels)
        self.info = self.r2.mean(axis=1)                          # static part of the heuristic
        self.n_ants, self.max_iter, self.patience = n_ants, max_iterations, patience
        self.alpha, self.beta, self.rho, self.q = alpha, beta, rho, q
        self.tau_init, self.tau_min, self.tau_max, self.eps = tau_init, tau_min, tau_max, eps
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.search_space = comb(n_features, k)

    # ------------------------------------------------------------------ one ant
    def heuristic(self, chosen):
        """eta for every channel given the channels already chosen by this ant."""
        if not chosen:
            return self.info + self.eps
        overlap = self.r2[:, chosen].max(axis=1)                  # how much j duplicates what we have
        return self.info * (1.0 - overlap) + self.eps

    def construct(self, tau):
        chosen = []
        available = np.ones(self.C, dtype=bool)
        for _ in range(self.k):
            w = (tau ** self.alpha) * (self.heuristic(chosen) ** self.beta)
            w[~available] = 0.0
            j = int(self.rng.choice(self.C, p=w / w.sum()))        # roulette-wheel choice
            chosen.append(j)
            available[j] = False
        mask = np.zeros(self.C, dtype=np.int8)
        mask[chosen] = 1
        return mask

    # ------------------------------------------------------------------ the colony
    def run(self):
        tau = np.full(self.C, self.tau_init)                      # 2. initial pheromone: all equal
        best, best_fit, stall = None, -np.inf, 0
        evaluated, history = set(), []
        stop_reason = "max_iterations"
        for t in range(self.max_iter + 1):                        # iteration 0 = first colony
            ants = [self.construct(tau) for _ in range(self.n_ants)]
            fit = np.array([self.fitness(a) for a in ants])
            evaluated.update(a.tobytes() for a in ants)
            i_best = int(np.argmax(fit))
            if fit[i_best] > best_fit + 1e-12:
                best, best_fit, stall = ants[i_best].copy(), float(fit[i_best]), 0
            elif t > 0:
                stall += 1
            # pheromone update: evaporate, then deposit for iteration-best and best-so-far
            tau = (1.0 - self.rho) * tau
            tau += self.q * fit[i_best] * ants[i_best]
            tau += self.q * best_fit * best
            tau = np.clip(tau, self.tau_min, self.tau_max)
            p = tau / tau.sum()
            history.append({"iteration": t, "gbest": best_fit, "iter_best": float(fit[i_best]),
                            "iter_mean": float(fit.mean()), "iter_worst": float(fit.min()),
                            "unique_subsets_in_colony": len({a.tobytes() for a in ants}),
                            "evaluations_cumulative": len(evaluated),
                            "pheromone_entropy": float(-(p * np.log(p)).sum() / np.log(self.C)),
                            "pheromone": tau.round(5).tolist(),
                            "gbest_selected_idx": np.flatnonzero(best).tolist()})
            if len(evaluated) == self.search_space:
                stop_reason = "every possible subset evaluated"
                break
            if stall >= self.patience:
                stop_reason = f"no improvement for {self.patience} iterations"
                break
        self.best_, self.best_fitness_, self.pheromone_ = best, best_fit, tau
        self.history_, self.iterations_ = history, history[-1]["iteration"]
        self.stop_reason_, self.n_evaluations_ = stop_reason, len(evaluated)
        return self
