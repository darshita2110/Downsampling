# Particle Swarm Optimization (PSO) channel selection

Code: `Kartavya/algorithms/combinatorial/pso/` (`config.py`, `pso.py`, `train.py`, `test.py`).
Outputs: `Kartavya/outputs/<category>/pso/png/`.

## 1. The idea from the basics

Like the GA, PSO **selects** K of the 8 channels and keeps their values unchanged. Unlike
PCA or ICA, it creates no new mixed values. The two methods differ in how they search for
the best subset.

PSO is modelled on a flock of birds looking for food:
- Each **particle** is a point in space with a **velocity**.
- Each particle remembers the best place *it* has found (its **personal best**, pbest).
- The whole swarm shares the best place *anyone* has found (the **global best**, gbest).
- At each step, every particle keeps some of its momentum, is pulled back towards its
  own best, and is pulled towards the swarm's best. The random weights on those pulls
  make it explore around them.

```
v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)     r1, r2 random in [0,1], per dimension
v <- clip(v, -v_max, v_max)
x <- clip(x + v, 0, 1)
```

| Symbol | Meaning | Value |
|---|---|---|
| w | inertia: how much of the old velocity is kept; falls linearly from 0.9 to 0.4 (explore early, refine late) | 0.9 → 0.4 |
| c1 | cognitive pull towards the particle's own best | 1.5 |
| c2 | social pull towards the swarm's best | 1.5 |
| v_max | speed limit per dimension | 0.5 |
| particles | swarm size (same as the GA population) | 12 |

**Particle → subset ("top-K decoding").**
- A position is 8 numbers in [0, 1], one "preference score" per channel.
- The subset is the K channels with the highest scores. For example, with
  `x = [.1 .7 .2 .9 .3 .4 .0 .8]` and K = 3, the particle keeps channels 3, 7 and 1.
- Because classic PSO moves continuous points, this decoding lets the standard equations
  run unchanged and always yields a valid K-subset, so no repair step is needed.

**Initialisation.** Positions are uniform in [0, 1]; velocities are uniform in [−v_max, v_max].

**Personal and global bests.** pbest is replaced whenever a particle finds a better
fitness. gbest is the best of all the pbests.

**Termination.** A run stops on whichever comes first:
- 40 iterations;
- 10 iterations without gbest improving;
- every possible subset has been seen.

## 2. Fitness function

PSO uses **the same function as the GA**, imported from `ga/ga.py`, so the two searches
are compared on identical terms:

```
fitness(S) = mean_R2(S) + 0.5 * min_R2(S) - 0.1 * redundancy(S)
```

- **mean_R2 (information preservation and reconstruction quality).** A least-squares
  linear model rebuilds all 8 channels from the kept ones. R2 is the share of each
  channel's variance recovered, averaged over channels. It is unit-free, so display
  levels and NDVI/NDWI count equally.
- **min_R2.** The worst channel's R2, so no product is silently lost.
- **redundancy.** Mean |r| between the kept channels, which penalises keeping both copies
  of a duplicate.

**Separation from test data.**
- R2 is scored on validation pixels: 30 % of a 400k-pixel training sample. The model is
  fitted on the other 70 %.
- Test pixels are never used in the optimisation.
- `test.py` only loads the saved subset and weights.

**Downstream utility** (e.g. classification) is not used, because no valid labels exist
(see LDA.md).

## 3. Data notes (same as GA)

These are 8 channels from the PNGs, **not** original Sentinel-2 bands:
- 6 are display levels of the RGB and False-Color composites;
- 2 are NDVI/NDWI decoded from their colour maps.

Two pairs are exact duplicates (r = 1.0):
- `RGB:R(B04)` = `False_Color:G(B04)`
- `RGB:G(B03)` = `False_Color:B(B03)`

Swapping one copy for the other gives identical fitness, so success is judged by fitness
value, not by channel names.

There is one date, so the split is spatial (top half trains, bottom half tests). With 5
dates the scripts switch automatically to a temporal split (indices 0–2 train, 3–4 test).

## 4. Results (actual run, 2026-09-17)

### Selected subsets

| K | Kept channels | Fitness | Same fitness as GA? |
|---|---|---|---|
| 1 | RGB:B(B02) | 0.8158 | yes |
| 2 | FC:B(B03), NDWI | 1.3708 | yes |
| 3 | RGB:B(B02), FC:R(B08), NDVI | 1.4081 | yes |
| 4 | RGB:B(B02), FC:R(B08), FC:G(B04), NDWI | 1.4320 | yes (GA chose the duplicate RGB:R instead of FC:G) |
| 5 | RGB:B(B02), FC:R(B08), FC:G(B04), NDVI, NDWI | 1.4364 | yes |
| 6 | the 6 distinct bands | 1.4376 | yes |
| 7 | 6 distinct + one duplicate | 1.4338 | yes |

### Convergence and verification

Each K was run with 5 seeds. All 8 to 70 subsets per K were also scored exhaustively.

| K | Subsets | Seeds reaching optimum, PSO / GA | Subsets evaluated per run, PSO / GA | Iterations | Time per run (s) |
|---|---|---|---|---|---|
| 1 | 8 | 5/5 / 5/5 | 7.8 / 8.0 | 0–10 | 0.37 |
| 2 | 28 | 4/5 / 4/5 | 19.8 / 19.6 | 10 | 1.00 |
| 3 | 56 | **1/5** / 3/5 | 24.0 / 26.4 | 10–13 | 1.35 |
| 4 | 70 | 4/5 / 3/5 | 29.4 / 26.0 | 10–17 | 1.95 |
| 5 | 56 | 5/5 / 5/5 | 26.0 / 26.2 | 10–12 | 2.02 |
| 6 | 28 | 5/5 / 5/5 | 17.0 / 20.2 | 10 | 1.50 |
| 7 | 8 | 5/5 / 5/5 | 7.6 / 8.0 | 1–10 | 0.73 |

- **The best swarm reached the global optimum at every K.**
- **At K=3, four of five swarms got trapped** on {NIR, red, NDWI}, fitness 1.3966 against
  the optimum's 1.4081.
  - The swarm converges on whatever gbest it finds first.
  - Once all particles share the same top-3 ranking, the velocities shrink and nothing
    moves them away.
  - This is PSO's known weakness (premature convergence), and it is why several seeds are
    run.
- **Cost.** Peak memory per run was 45–72 MB. The full train.py took 75 s, including image
  loading (9 s, 523 MB peak) and the exhaustive check.

### Test region (bottom half, 541,531 pixels)

| K | Reduction | PSNR (dB) | SSIM | RMSE | Reduced .npy | PCA PSNR / SSIM |
|---|---|---|---|---|---|---|
| 1 | 87.5 % | 20.16 | 0.778 | 0.098 | 2.07 MB | 21.36 / 0.813 |
| 2 | 75 % | 27.31 | 0.953 | 0.043 | 4.13 MB | 28.88 / 0.968 |
| 3 | 62.5 % | 30.71 | 0.966 | 0.029 | 6.20 MB | 35.23 / 0.985 |
| 4 | 50 % | 38.29 | 0.993 | 0.012 | 8.27 MB | 39.91 / 0.995 |
| 5 | 37.5 % | 38.91 | 0.995 | 0.011 | 10.33 MB | 45.29 / 0.997 |
| 6 | 25 % | lossless | 1.000 | 0 | 12.40 MB | lossless |
| 7 | 12.5 % | lossless | 1.000 | 0 | 14.46 MB | lossless |

- The original 8-channel float32 image is 16.53 MB.
- Kept channels come back exactly (maximum error about 1e-15).
- Reducing takes 1–5 ms and rebuilding 15–32 ms.
- Test metrics equal the GA's at every K, because both found equally good subsets (at K=4
  and K=5 they are the duplicate-swapped twins).

## 5. Verdict

- **PSO and GA reach the same answer on this problem.** Six channels are lossless, and two
  channels are pure duplicates that can be dropped.
- **Selection costs quality against PCA.** It is 1.2–6.4 dB behind PCA at K = 1–5, because
  PCA can mix channels. In exchange, the stored data stay real, readable channels.
- **GA is the more reliable single run on this landscape.** PSO is equally fast, but its
  single runs were less reliable at K=3 (1/5 against 3/5), because the swarm collapses
  onto one ranking.
- **Possible improvements, not implemented:** a ring (local-best) topology, a restart when
  velocities die out, or a larger swarm.

## 6. Limitations

- There is only one date.
- With 8 channels, exhaustive search is cheap. PSO is implemented and verified here so it
  can scale to larger channel sets.
- No labels exist, so downstream utility was not evaluated.
- Reduced data are stored as float32 for comparability with the other methods.
- The `--dataset raw` option exits cleanly, because no original band files exist. No bands
  are fabricated.

## 7. Run

```bash
pip install -r Kartavya/requirements.txt
python Kartavya/algorithms/combinatorial/pso/train.py   # ~75 s
python Kartavya/algorithms/combinatorial/pso/test.py    # ~20 s
```

The test script also writes `pso_vs_pca.csv/.png` and, if GA results exist, `pso_vs_ga.csv`.
