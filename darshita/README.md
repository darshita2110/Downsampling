# Darshita: band selection for downlinking

My part of the project: choose which spectral bands a satellite should send, using
ranking-based, search-based and graph/clustering-based methods, and compare them.

## Part 1: Ranking-based selection (done)

| File | What it does |
|---|---|
| `step1_load_and_look.py` | Downloads Indian Pines, shows bands and pixel spectra |
| `ranking.py` | The three ranking methods: Variance, Entropy, MVPCA |
| `step2_ranking_selection.py` | Scores every band, keeps the top K, saves the choice |
| `step3_evaluate_ranking.py` | Rebuilds all bands from the chosen K, measures PSNR and SAM |

Run from the repo folder, in order:

```bash
pip install numpy scipy matplotlib
python darshita/step1_load_and_look.py
python darshita/step2_ranking_selection.py
python darshita/step3_evaluate_ranking.py
```

Pictures go to `darshita/outputs/`. The image is saved in `data/darshita/` (not on GitHub).

## What Part 1 shows (Indian Pines, K = 10 of 200 bands)

| Method | PSNR (dB) | SAM (deg) | Redundancy |
|---|---|---|---|
| Variance | 25.62 | 5.84 | 0.876 |
| Entropy | 24.82 | 6.30 | 0.984 |
| MVPCA | 24.70 | 6.36 | 0.995 |
| Uniform (baseline) | 29.47 | 4.54 | 0.432 |

All three ranking methods lose to evenly spaced bands. Each band is scored on its own,
so the neighbours of a high-scoring band also score high, and the chosen bands are
near-copies of each other (redundancy close to 1). This weakness is what search-based
and clustering-based methods fix.

## Next

- Part 2: search-based (SFS, Genetic Algorithm, PSO)
- Part 3: graph / clustering-based (WaLuDi, WaLuMI, Affinity Propagation)
- Part 4: run everything on real Sentinel-2 / EnMAP data and compare
