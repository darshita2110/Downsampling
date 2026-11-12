# Darshita: channel (band) selection for satellite downlinking

A satellite records many channels (bands) but cannot downlink all of them. My part of the
project chooses **which K channels to send** so that all channels can be rebuilt on the
ground, with three families of methods, and compares them.

| Family | Idea | Methods | Code |
|---|---|---|---|
| Ranking | Score every channel alone, keep the top K | Variance, Entropy, MVPCA | `ranking.py` |
| Search | Try many groups of K, keep the group that rebuilds the image best | SFS, Genetic Algorithm, PSO (+ Exhaustive when possible) | `search.py` |
| Graph / clustering | Group look-alike channels into K groups, send one per group | WaLuDi, WaLuMI, Affinity Propagation | `clustering.py` |
| Baseline | Evenly spaced channels | Uniform | `common.py` |

**Test used for every method:** only the K chosen channels are "sent"; all channels are
rebuilt with linear regression fitted on 10% of the pixels; PSNR (dB, higher = better),
SAM (degrees, lower = better) and redundancy (mean |correlation| of the sent channels,
lower = better) are measured on the other 90%.

## Datasets

| Name | What it is | Channels | K used |
|---|---|---|---|
| `raichur` | **Real Sentinel-2 L2A satellite data over Raichur**, bands B01–B12, 20 m, clean 512 × 512 window | 12 | 4 (33%) |
| `indian_pines` | Real airborne hyperspectral image (AVIRIS), standard benchmark in band-selection papers | 200 | 10 (5%) |

The team's PNG folders (RGB, False_Color, NDVI, NDWI) are not used: they are display images
or indices computed from bands, not the measured bands.

## Run everything (from the repo folder)

```bash
pip install numpy scipy matplotlib pandas openeo xarray netcdf4

# data (once)
python darshita/data_1_download_s2.py        # Copernicus login in the browser the first time
python darshita/data_2_check_s2.py           # cloud check, picks a clean 512 x 512 window

# the three families, one by one (add "raichur" for the real data, nothing for Indian Pines)
python darshita/step2_ranking_selection.py raichur
python darshita/step3_evaluate_ranking.py raichur
python darshita/step4_search_selection.py raichur
python darshita/step5_clustering_selection.py raichur

# final comparison and the demo
python darshita/step6_compare_all.py raichur
python darshita/demo.py raichur
```

`step1_load_and_look.py` and `app.py` are the earlier Indian Pines-only introduction and slider app.

## Files

| File | What it does |
|---|---|
| `common.py` | Load a dataset, rebuild, PSNR, SAM, redundancy, uniform baseline |
| `ranking.py`, `search.py`, `clustering.py` | The 9 methods |
| `data_1_download_s2.py` | Downloads the 12 Sentinel-2 bands + cloud layer (SCL) for 4 clear dates |
| `data_2_check_s2.py` | Checks each date, cuts the cleanest window, saves `data_raichur_<date>.png` |
| `step2` / `step3` | Ranking: choose, then evaluate (+ PSNR vs K) |
| `step4_search_selection.py` | SFS, GA, PSO (+ exhaustive on 12 bands); convergence graph |
| `step5_clustering_selection.py` | WaLuDi, WaLuMI, Affinity Propagation; tree and similarity map |
| `step6_compare_all.py` | All methods × all K: `final_results_*.csv`, graphs, `final_summary_*.md` |
| `demo.py` | 6-screen demo for the teacher; terminal prints what to say on each screen |

All pictures and tables go to `darshita/outputs/`; data goes to `data/darshita/` (not on GitHub).

## Results on Indian Pines (K = 10 of 200 channels)

| Method | Family | PSNR (dB) | SAM (°) | Redundancy |
|---|---|---|---|---|
| GA | Search | 30.45 | 4.08 | 0.331 |
| SFS | Search | 30.41 | 4.05 | 0.361 |
| PSO | Search | 30.20 | 4.19 | 0.382 |
| Affinity Propagation | Clustering | 29.77 | 4.44 | 0.289 |
| WaLuDi | Clustering | 29.60 | 4.46 | 0.614 |
| Uniform | Baseline | 29.47 | 4.54 | 0.432 |
| WaLuMI | Clustering | 28.92 | 4.84 | 0.554 |
| Variance | Ranking | 25.62 | 5.84 | 0.876 |
| Entropy | Ranking | 24.82 | 6.30 | 0.984 |
| MVPCA | Ranking | 24.70 | 6.36 | 0.995 |

Search > clustering > ranking at every K from 5 to 30. Ranking sends near-copies (redundancy
0.88–0.995); search and clustering avoid them.

## Results on Raichur (K = 4 of 12 bands, 33% of the data)

From `python darshita/step6_compare_all.py raichur` (crop `raichur_2026-10-12_crop.npy`, 512 × 512 pixels).

| Method | Family | PSNR (dB) | SAM (°) | Redundancy | Time (s) | Channels sent |
|---|---|---|---|---|---|---|
| Exhaustive | Best possible | 35.83 | 2.63 | 0.353 | 0.01 | B01 B07 B09 B12 |
| PSO | Search | 35.83 | 2.63 | 0.353 | 0.25 | B01 B07 B09 B12 |
| GA | Search | 35.83 | 2.63 | 0.353 | 1.21 | B01 B07 B09 B12 |
| SFS | Search | 35.12 | 3.25 | 0.409 | 0.00 | B01 B04 B8A B09 |
| MVPCA | Ranking | 35.02 | 2.84 | 0.398 | 0.07 | B01 B03 B8A B11 |
| AffinityProp | Clustering | 34.92 | 2.88 | 0.405 | 0.06 | B01 B03 B8A B12 |
| WaLuDi | Clustering | 34.64 | 2.90 | 0.390 | 0.01 | B01 B03 B07 B12 |
| WaLuMI | Clustering | 33.12 | 3.83 | 0.392 | 0.02 | B01 B03 B07 B09 |
| Uniform | Baseline | 32.57 | 3.34 | 0.439 | 0.00 | B01 B05 B08 B12 |
| Entropy | Ranking | 31.91 | 3.97 | 0.609 | 0.05 | B01 B07 B8A B09 |
| Variance | Ranking | 30.04 | 4.33 | 0.590 | 0.03 | B08 B8A B09 B11 |

GA / PSO over 5 seeds: GA 35.83 ± 0.00 dB, PSO 35.83 ± 0.00 dB

Best PSNR (dB) of each family at every K:

| Family | K=2 | K=3 | K=4 | K=5 | K=6 | K=8 |
|---|---|---|---|---|---|---|
| Clustering | 30.73 | 32.48 | 34.92 | 39.80 | 41.06 | 45.71 |
| Ranking | 26.31 | 30.04 | 35.02 | 35.11 | 41.34 | 41.99 |
| Search | 30.73 | 33.16 | 35.83 | 39.81 | 41.46 | 46.14 |

**Findings**

- **Search wins.** GA and PSO find B01 B07 B09 B12, the same group as the exhaustive search over all 495 groups, so they reach the true optimum (35.83 dB) on every one of 5 seeds.
- **Clustering is a close second** and stays within 0.91 dB of search at every K, while choosing in milliseconds.
- **Ranking is unreliable.** MVPCA comes close at K = 4 (0.81 dB behind), but Variance and Entropy fall below uniform spacing because they send near-copies (redundancy 0.59–0.61). At K = 8, ranking is 4.15 dB behind search.
- **The best PSNR doesn't always give the best-looking colours.** PSNR averages all 12 bands, and the search group contains none of the true-colour bands (B02, B03, B04), so its true-colour rebuild looks less faithful than MVPCA's (demo screen 4). One date and one 512 × 512 crop were tested; times depend on the machine.
