"""
search.py - the search-based channel selection methods.

The idea in one line:
    try MANY groups of K channels, check how well each group can rebuild the image,
    and keep the best group.

Unlike ranking, a group is judged TOGETHER, so a group of near-copies scores badly
and is thrown away. That is exactly what ranking could not do.

The score of a group ("cost", lower = better):
    cost(group) = mean squared error when ALL channels are rebuilt from the channels
                  in the group (linear regression, on the calibration pixels only)

Methods:
  1. SFS  (Sequential Forward Selection) - greedy: add the best channel one at a time
  2. GA   (Genetic Algorithm)            - evolution: groups breed and mutate, the fittest survive
  3. PSO  (Particle Swarm Optimisation)   - a swarm of particles flies towards good groups
  4. Exhaustive                           - try EVERY possible group (only possible when there are
                                            few groups, e.g. 4 of 12 bands = 495 groups). Gives the
                                            true best answer, to check how close GA / PSO get.
"""
from itertools import combinations

import numpy as np


# ================================================================ the cost
def make_cost(pixels01, calib, n_sample=3000, seed=0):
    """Build the cost function. Uses a sample of the calibration pixels, so the test pixels
    (used later for the final score) are never seen during the search."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(calib, min(n_sample, len(calib)), replace=False)
    Y = pixels01[rows].astype(np.float64)                # all channels of the sample pixels
    n, B = Y.shape

    # Simple version (easy to read, but slow because the search calls it thousands of times):
    #     X = [chosen channels, 1];  w = least_squares(X, Y);  cost = mean((X @ w - Y)^2)
    # Fast version below gives the SAME number. It precomputes all channel-to-channel products
    # once (matrix G), so each call only solves a tiny (k+1) x (k+1) system.
    Ya = np.hstack([Y, np.ones((n, 1))])                 # last column = the constant
    G = Ya.T @ Ya
    total = np.trace(G[:B, :B])                          # sum of squares of all channels

    def cost(group):
        idx = list(group) + [B]                          # chosen channels + the constant
        A = G[np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx))
        R = G[idx, :B]
        explained = np.sum(R * np.linalg.solve(A, R))    # part of Y the group can rebuild
        return float(max(total - explained, 0.0) / (n * B))   # mean squared rebuild error

    return cost


# ================================================================ method 1
def sfs(cost, n_channels, k):
    """Start with nothing. At every step add the ONE channel that lowers the cost most."""
    group, history = [], []
    for _ in range(k):
        remaining = [c for c in range(n_channels) if c not in group]
        best = min(remaining, key=lambda c: cost(group + [c]))   # try each, keep the best
        group.append(best)
        history.append(cost(group))
    return sorted(group), history


# ================================================================ method 2
def genetic_algorithm(cost, n_channels, k, pop_size=40, generations=150, seed=0):
    """
    population : pop_size groups, each = k different channels
    each generation:
      - keep the 2 best groups unchanged ("elitism")
      - make children until the population is full again:
          parents   : pick 3 random groups, the best one becomes a parent (twice -> 2 parents)
          crossover : child = k channels picked at random from the two parents' channels
          mutation  : each channel of the child is swapped for a random unused one with chance 1/k
    returns the best group found and the best cost of every generation
    """
    rng = np.random.default_rng(seed)
    population = [rng.choice(n_channels, k, replace=False) for _ in range(pop_size)]
    costs = np.array([cost(g) for g in population])
    history = [costs.min()]

    def pick_parent():
        three = rng.choice(pop_size, 3, replace=False)       # "tournament" of 3
        return population[three[costs[three].argmin()]]

    for _ in range(generations):
        order = np.argsort(costs)
        children = [population[order[0]], population[order[1]]]     # elitism
        while len(children) < pop_size:
            mum, dad = pick_parent(), pick_parent()
            pool = np.union1d(mum, dad)                          # all channels of both parents
            child = rng.choice(pool, k, replace=False)            # crossover
            for i in range(k):                                    # mutation
                if rng.random() < 1.0 / k:
                    unused = np.setdiff1d(np.arange(n_channels), child)
                    child[i] = rng.choice(unused)
            children.append(child)
        population = children
        costs = np.array([cost(g) for g in population])
        history.append(costs.min())
    return sorted(population[costs.argmin()].tolist()), history


# ================================================================ method 3
def pso(cost, n_channels, k, n_particles=30, iterations=150, seed=0):
    """
    Each particle is a list of n_channels numbers between 0 and 1 ("how much I want each channel").
    Its group = the k channels with the biggest numbers.
    Every iteration each particle moves:
        velocity = w * velocity                         (keep going)
                 + c1 * random * (my best - me)          (go back to my own best spot)
                 + c2 * random * (swarm best - me)       (go towards the swarm's best spot)
    w goes down from 0.9 to 0.4: explore first, then settle.
    """
    rng = np.random.default_rng(seed)
    c1 = c2 = 1.5

    def group_of(position):
        return np.argsort(position)[::-1][:k]                    # the k largest values

    pos = rng.random((n_particles, n_channels))
    vel = rng.uniform(-0.1, 0.1, (n_particles, n_channels))
    my_best = pos.copy()
    my_best_cost = np.array([cost(group_of(p)) for p in pos])
    swarm_best = my_best[my_best_cost.argmin()].copy()
    history = [my_best_cost.min()]

    for it in range(iterations):
        w = 0.9 - 0.5 * it / max(iterations - 1, 1)
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = w * vel + c1 * r1 * (my_best - pos) + c2 * r2 * (swarm_best - pos)
        vel = np.clip(vel, -0.2, 0.2)                            # no huge jumps
        pos = np.clip(pos + vel, 0, 1)
        for i, p in enumerate(pos):
            c = cost(group_of(p))
            if c < my_best_cost[i]:
                my_best[i], my_best_cost[i] = p.copy(), c
        swarm_best = my_best[my_best_cost.argmin()].copy()
        history.append(my_best_cost.min())
    return sorted(group_of(swarm_best).tolist()), history


# ================================================================ method 4
def n_groups(n_channels, k):
    """How many different groups of k channels exist (n choose k)."""
    from math import comb
    return comb(n_channels, k)


def exhaustive(cost, n_channels, k, limit=5000):
    """Try every group. Only when there are at most `limit` groups (otherwise it takes forever:
    10 of 200 channels = 22 million million groups)."""
    if n_groups(n_channels, k) > limit:
        return None, None
    best = min(combinations(range(n_channels), k), key=cost)
    return sorted(best), [cost(best)]
