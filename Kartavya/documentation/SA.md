# Simulated Annealing (SA) channel selection

Code: `Kartavya/algorithms/combinatorial/sa/` (`config.py`, `sa.py`, `train.py`, `test.py`).
Outputs: `Kartavya/outputs/<category>/sa/png/`.

## 1. The idea from the basics

When metal is heated, its atoms move freely. When it cools **slowly**, they settle into an
orderly, low-energy crystal. When it cools too fast, they freeze in a messy state.

SA copies this process. Unlike the GA, PSO and ACO, it works with **one** solution at a time:

1. Start from a random subset of K channels (the **current solution**).
2. Propose a **neighbour**: swap one kept channel for one dropped channel. The neighbour
   is still a valid K-subset.
3. Compute the change in the objective, Δ = f(neighbour) − f(current).
4. **Acceptance** (Metropolis rule, since we maximise):
   - if Δ ≥ 0, always accept;
   - if Δ < 0, accept with probability exp(Δ / T).
5. **Cool:** T ← 0.95 · T, then repeat from step 2.

At high T, even clearly worse moves are often accepted, so the search wanders and can
escape local optima. As T falls, worse moves are rarely accepted and the search settles
on the best region.

**Initial temperature (from the data).**
- Score 20 random (subset, neighbour) pairs and take the average size of the
  *worsening* moves, |Δ|.
- Set T0 = −mean|Δ_worse| / ln(0.8), so that a typical worse move starts with an 80 %
  chance of acceptance.
- T0 adapts to each K: it is 1.6 at K=2, where moves change the objective a lot, and
  0.07 at K=6, where they barely do.

**Best-so-far.** The best subset ever *evaluated* is kept, including rejected candidates
and the T0 samples. An earlier version kept only accepted moves. It lost optima that had
been evaluated and then rejected (at K=1 and K=7 only 1 of 5 runs reported the optimum
even though all 8 subsets had been scored). Verification caught this, and it is fixed.

**Termination.** A run stops on whichever comes first:
- T < 1e-4 · T0 ("frozen", about 180 moves);
- the best solution unchanged for 100 moves;
- 300 moves;
- every possible subset has been evaluated.

## 2. Objective function

SA uses the same function as the GA, PSO and ACO (imported from `ga/ga.py`), and maximises it:

```
f(S) = mean_R2(S) + 0.5 * min_R2(S) - 0.1 * redundancy(S)
```

- **mean_R2 (information preservation and reconstruction quality).** A least-squares
  linear model rebuilds all 8 channels from the kept ones. The term is the share of each
  channel's variance recovered, averaged over channels.
- **min_R2.** The worst channel's R², so no product is silently lost.
- **redundancy.** The mean |r| among the kept channels.

**Separation from test data.** R² is measured on validation pixels from the **training**
half (30 % of a 400k-pixel sample). The model is fitted on the other 70 %. `test.py`
performs no optimisation: it loads the saved subset and weights and only evaluates.

**Downstream utility** (e.g. classification) is not used, because no valid labels exist
(see LDA.md).

## 3. Data notes

These are the same 8 channels as the other methods, and they are **not** original
Sentinel-2 bands:
- 6 are RGB/False-Color display levels;
- 2 are NDVI/NDWI decoded from colour maps.

Two pairs are exact duplicates (B04 and B03 appear in both composites). Swapping one copy
for the other gives identical objective values, so success is judged by value, not by
channel names.

There is one date, so the split is spatial (top half trains, bottom half tests). With 5
dates the scripts switch to a temporal split.

## 4. Results (actual run, 2026-09-17)

### Selected subsets

| K | Kept channels | Objective |
|---|---|---|
| 1 | RGB:B(B02) | 0.8158 |
| 2 | FC:B(B03), NDWI | 1.3708 |
| 3 | RGB:B(B02), FC:R(B08), NDVI | 1.4081 |
| 4 | RGB:R(B04), RGB:B(B02), FC:R(B08), NDWI | 1.4320 |
| 5 | RGB:R(B04), RGB:B(B02), FC:R(B08), NDVI, NDWI | 1.4364 |
| 6 | the 6 distinct bands | 1.4376 |
| 7 | 6 distinct + one duplicate | 1.4338 |

The objective values are identical to the GA's, PSO's and ACO's at all 7 values of K.

### Convergence, temperature and cost

Each K was run from 5 random starts (seeds 42–46), and every subset was also scored
exhaustively as a check.

| K | Subsets | Runs reaching optimum | Subsets evaluated per run | Moves | T0 (median) | Final T | Acceptance rate | Time per run (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 5/5 | 8.0 | 1–12 | 1.27 | 1.20 | 0.98 | 0.29 |
| 2 | 28 | 5/5 | 27.8 | 47–100 | 1.61 | 0.022 | 0.61 | 1.23 |
| 3 | 56 | 5/5 | 49.2 | 100–139 | 0.79 | 0.0034 | 0.56 | 2.52 |
| 4 | 70 | 5/5 | 56.0 | 100–135 | 0.33 | 0.0019 | 0.51 | 3.23 |
| 5 | 56 | 5/5 | 49.6 | 100–118 | 0.096 | 0.0005 | 0.53 | 3.11 |
| 6 | 28 | 5/5 | 27.4 | 38–100 | 0.070 | 0.0004 | 0.50 | 1.89 |
| 7 | 8 | 5/5 | 8.0 | 1 | 0.091 | 0.087 | 0.80 | 0.63 |

- **Every run reached the global optimum.** Of the four searches, SA is the only one with
  5/5 runs at every K.
- **That reliability is bought with evaluations.** SA scored 80–100 % of all subsets, for
  example 49 of 56 at K=3 and 56 of 70 at K=4. The GA, PSO and ACO used 13–29.
  - SA proposes one move at a time and keeps going for 100 moves after its last
    improvement, so on a search space this small it ends up close to exhaustive search.
  - The patience and cooling settings trade evaluations for reliability. A faster cooling
    rate or a smaller patience would use fewer evaluations, at more risk of missing the
    optimum.
- **Stopping.** 19 runs stopped because the best was unchanged for 100 moves. The other 16
  ran out of new subsets to evaluate; at K=1 and K=7 the T0 samples alone cover all 8.
- **Temperature behaviour** (`sa_temperature.png`): acceptance starts near 0.8–1.0 and
  falls to about 0.4 as T cools geometrically, as designed.
- **Cost.** Peak memory was 45–72 MB per run. The full train.py took 89 s, including image
  loading and the exhaustive check.

### Test region (bottom half, 541,531 pixels; no optimisation in testing)

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
- Reducing takes 1–3 ms and rebuilding 10–23 ms.

## 5. Verdict

- **All four search methods agree.** GA, PSO, ACO and SA find the same optimal subsets.
  Six channels are lossless, and the duplicate copies of B04 and B03 can be dropped.
- **SA is the simplest and the most reliable per run.** It has one solution, one move type
  and two key settings, but here it spends the most evaluations.
- **On large channel sets** the balance shifts: SA explores only a sliver of the space,
  and the cooling schedule becomes the critical setting.
- **Selection costs quality against PCA.** All four selection methods are 1.2–6.4 dB
  behind PCA at K = 1–5, in exchange for keeping real, interpretable channels.

## 6. Limitations

- There is only one date.
- With 8 channels, exhaustive search is cheap. SA is built to scale to larger channel sets.
- No labels exist, so downstream utility was not evaluated.
- Reduced data are stored as float32 for comparability with the other methods.
- The `--dataset raw` option exits cleanly, because no original band files exist. No bands
  are fabricated.

## 7. Run

```bash
pip install -r Kartavya/requirements.txt
python Kartavya/algorithms/combinatorial/sa/train.py   # ~90 s
python Kartavya/algorithms/combinatorial/sa/test.py    # ~15 s, evaluation only
```

The test script also writes `sa_vs_pca.csv/.png` and, if GA/PSO/ACO results exist,
`sa_vs_ga_pso_aco.csv`.
