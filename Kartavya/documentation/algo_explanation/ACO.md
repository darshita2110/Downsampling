# Ant Colony Optimization (ACO) channel selection

Code: `Kartavya/algorithms/combinatorial/aco/` (`config.py`, `aco.py`, `train.py`, `test.py`).
Outputs: `Kartavya/outputs/<category>/aco/png/`.

## 1. The idea from the basics

Real ants find short paths to food without a map:
- each ant leaves a scent (**pheromone**) on the path it walks;
- ants prefer paths that smell stronger;
- paths to good food get walked and marked more;
- scent **evaporates**, so abandoned paths fade.

The colony's shared memory lives in the pheromone, not in any one ant.

For channel selection:

| Ant world | Here |
|---|---|
| path | a subset of K channels |
| one step of the path | picking one more channel |
| pheromone τ_j | learned score of channel j; starts equal (1.0) for every channel |
| heuristic η_j | prior desirability of channel j, known without evaluating any subset |
| food quality | fitness of the finished subset |

### Building a solution (one ant)

An ant picks K channels one at a time, without replacement. The probability of
picking channel j next is

```
P(j) = τ_j^α · η_j^β  /  Σ over channels not yet picked of τ^α · η^β       (α = 1, β = 2)
```

The heuristic is

```
η_j = info_j × (1 − max r² between j and channels already picked) + 0.001
```

- `info_j` is the mean r² of channel j with all 8 channels: how much of the image it
  explains.
- The second factor makes an exact duplicate of an already-picked channel nearly
  impossible to pick next. This is how the heuristic encodes **redundancy**.

### Pheromone update (after every iteration of 12 ants)

1. **Evaporation:** τ ← (1 − ρ)·τ with ρ = 0.2. Old information fades by 20 % each iteration.
2. **Deposit:** for each channel of the iteration-best subset, and again for the
   best-so-far subset, τ_j += Q × fitness, with Q = 1.
3. **Bounds:** τ is kept within [0.1, 5.0] (MAX-MIN Ant System). No channel's probability
   reaches 0, and no channel can swamp the rest. This guards against the colony freezing
   on an early answer.

### Termination

A run stops on whichever comes first:
- 40 iterations;
- 10 iterations without improving the best-so-far subset;
- every possible subset has been seen.

## 2. Fitness function

ACO uses the same function as the GA and PSO (imported from `ga/ga.py`):

```
fitness(S) = mean_R2(S) + 0.5 * min_R2(S) - 0.1 * redundancy(S)
```

- **mean_R2 (information preservation and reconstruction quality).** A least-squares
  linear model rebuilds all 8 channels from the kept ones. The term is the average share
  of each channel's variance recovered. It is unit-free.
- **min_R2.** The worst channel's R², so no product is silently lost.
- **redundancy.** The mean |r| among the kept channels.

**Separation from test data.** R² is measured on validation pixels: 30 % of a 400k-pixel
sample from the **training** half. The heuristic's correlations also come from training
pixels. The test half is used only by `test.py`, which loads the saved subset and weights.

**Downstream utility** (e.g. classification) is not used, because no valid labels exist
(see LDA.md).

## 3. Data notes

These are the same 8 channels as the other methods, and they are **not** original
Sentinel-2 bands:
- 6 are RGB/False-Color display levels;
- 2 are NDVI/NDWI decoded from colour maps.

Two pairs are exact duplicates (r = 1.0):
- `RGB:R(B04)` = `False_Color:G(B04)`
- `RGB:G(B03)` = `False_Color:B(B03)`

Swapping one copy for the other gives identical fitness, so success is judged by fitness
value, not by channel names.

There is one date, so the split is spatial (top half trains, bottom half tests). With 5
dates the scripts switch to a temporal split (indices 0–2 train, 3–4 test).

## 4. Results (actual run, 2026-09-17)

### Selected subsets

| K | Kept channels | Fitness |
|---|---|---|
| 1 | RGB:B(B02) | 0.8158 |
| 2 | FC:B(B03), NDWI | 1.3708 |
| 3 | RGB:B(B02), FC:R(B08), NDVI | 1.4081 |
| 4 | RGB:B(B02), FC:R(B08), FC:G(B04), NDWI | 1.4320 |
| 5 | RGB:B(B02), FC:R(B08), FC:G(B04), NDVI, NDWI | 1.4364 |
| 6 | the 6 distinct bands | 1.4376 |
| 7 | 6 distinct + one duplicate | 1.4338 |

The training fitness is identical to the GA's and PSO's at all 7 values of K.

### Convergence and cost

Each K was run with 5 colonies (seeds 42–46), and every subset was also scored
exhaustively as a check.

| K | Subsets | Colonies reaching optimum (ACO / GA / PSO) | Subsets evaluated per run (ACO / GA / PSO) | Iterations | Time per run (s) |
|---|---|---|---|---|---|
| 1 | 8 | 4/5 / 5/5 / 5/5 | 7.6 / 8.0 / 7.8 | 1–10 | 0.41 |
| 2 | 28 | 5/5 / 4/5 / 4/5 | 14.8 / 19.6 / 19.8 | 10 | 0.93 |
| 3 | 56 | 3/5 / 3/5 / 1/5 | 13.4 / 26.4 / 24.0 | 10–14 | 0.96 |
| 4 | 70 | 5/5 / 3/5 / 4/5 | 13.6 / 26.0 / 29.4 | 10 | 0.99 |
| 5 | 56 | 5/5 / 5/5 / 5/5 | 11.6 / 26.2 / 26.0 | 10 | 1.06 |
| 6 | 28 | 5/5 / 5/5 / 5/5 | 4.0 / 20.2 / 17.0 | 10 | 0.47 |
| 7 | 8 | 5/5 / 5/5 / 5/5 | 4.0 / 8.0 / 7.6 | 10 | 0.48 |

- **The best colony reached the global optimum at every K.**
- **ACO is the most economical of the three searches.** It needs about half the fitness
  evaluations of the GA and PSO at K = 3–5, and runs fastest (0.3–1.4 s per run, 45–72 MB
  peak).
  - The heuristic does much of the work: at K = 6 every ant avoids the two duplicates, so
    only 4 distinct subsets are ever built, all of them optimal.
- **Two misses:**
  - At K = 1, one colony settled on RGB:G(B03), with fitness 0.8136 against the
    optimum's 0.8158. The heuristic slightly prefers green, and the fitness gap is tiny.
  - At K = 3, two colonies settled on {NIR, red, NDWI} (1.3966), the same trap PSO fell
    into.
- **Pheromone behaviour** (`aco_pheromone.png`): within 1–2 iterations the chosen channels
  rise to the 5.0 cap and the others evaporate towards the 0.1 floor. The colony commits
  quickly.
- **Total cost.** The full train.py took about 60 s, including image loading (9 s) and the
  exhaustive check.

### Test region (bottom half, 541,531 pixels)

| K | Reduction | PSNR (dB) | SSIM | RMSE | Reduced .npy | PCA PSNR |
|---|---|---|---|---|---|---|
| 1 | 87.5 % | 20.16 | 0.778 | 0.098 | 2.07 MB | 21.36 |
| 2 | 75 % | 27.31 | 0.953 | 0.043 | 4.13 MB | 28.88 |
| 3 | 62.5 % | 30.71 | 0.966 | 0.029 | 6.20 MB | 35.23 |
| 4 | 50 % | 38.29 | 0.993 | 0.012 | 8.27 MB | 39.91 |
| 5 | 37.5 % | 38.91 | 0.995 | 0.011 | 10.33 MB | 45.29 |
| 6 | 25 % | lossless | 1.000 | 0 | 12.40 MB | lossless |
| 7 | 12.5 % | lossless | 1.000 | 0 | 14.46 MB | lossless |

- The original 8-channel float32 image is 16.53 MB.
- Kept channels come back exactly.
- Reducing takes 1.5–5 ms and rebuilding 20–35 ms.

## 5. Verdict

- **All three searches agree.** GA, PSO and ACO find the same best subsets. Six channels
  reconstruct everything, and the two duplicate copies can be dropped with no loss.
- **ACO reaches the answer with the fewest evaluations.** The prior knowledge in its
  heuristic (information content, non-redundancy) helps. This matters when each fitness
  evaluation is expensive, for example with many channels or a classifier-based fitness.
- **Selection costs quality against PCA.** All three are 1.2–6.4 dB behind PCA at K = 1–5,
  in exchange for keeping real, interpretable channels.

## 6. Limitations

- There is only one date.
- With 8 channels, exhaustive search is cheap. ACO is implemented and verified here so it
  scales to larger channel sets.
- The heuristic uses training correlations. If those change across seasons, it should be
  recomputed per training set (the code already does this).
- No labels exist, so downstream utility was not evaluated.
- Reduced data are stored as float32 for comparability with the other methods.
- The `--dataset raw` option exits cleanly, because no original band files exist. No bands
  are fabricated.

## 7. Run

```bash
pip install -r Kartavya/requirements.txt
python Kartavya/algorithms/combinatorial/aco/train.py   # ~60 s
python Kartavya/algorithms/combinatorial/aco/test.py    # ~20 s
```

The test script also writes `aco_vs_pca.csv/.png` and, if GA/PSO results exist,
`aco_vs_ga_pso.csv`.
