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

# the same comparison on each of the 4 dates, then one summary across dates
python darshita/step6_compare_all.py raichur_2026-09-27
python darshita/step6_compare_all.py raichur_2026-10-09
python darshita/step6_compare_all.py raichur_2026-10-12
python darshita/step6_compare_all.py raichur_2026-10-27
python darshita/step7_all_dates.py
```

`raichur` is the cleanest date (named in `data/darshita/raichur_best.txt`); `raichur_<date>` loads that date's crop.

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
| `step7_all_dates.py` | Combines step 6 of every Raichur date: `final_all_dates_raichur.md/.csv/.png` |
| `step8_classification.py` | Indian Pines only: crop classification (SVM) from the chosen bands; needs `Indian_pines_gt.mat` and scikit-learn |
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
- **The best PSNR doesn't always give the best-looking colours.** PSNR averages all 12 bands, and the search group contains none of the true-colour bands (B02, B03, B04), so its true-colour rebuild looks less faithful than MVPCA's (demo screen 4). Times depend on the machine.

## Results on all 4 Raichur dates (K = 4 of 12 bands)

From `python darshita/step7_all_dates.py`. PSNR (dB), sorted by the mean over the dates:

| Method | Family | 2026-09-27 | 2026-10-09 | 2026-10-12 | 2026-10-27 | Mean | Std |
|---|---|---|---|---|---|---|---|
| Exhaustive | Best possible | 36.33 | 34.75 | 35.83 | 35.85 | 35.69 | 0.67 |
| GA | Search | 36.33 | 34.75 | 35.83 | 35.85 | 35.69 | 0.67 |
| PSO | Search | 36.33 | 34.75 | 35.83 | 35.50 | 35.60 | 0.66 |
| SFS | Search | 35.85 | 34.70 | 35.12 | 35.43 | 35.28 | 0.48 |
| AffinityProp | Clustering | 35.06 | 34.45 | 34.92 | 35.50 | 34.98 | 0.44 |
| WaLuMI | Clustering | 36.16 | 34.61 | 33.12 | 35.19 | 34.77 | 1.27 |
| WaLuDi | Clustering | 34.31 | 34.61 | 34.64 | 35.50 | 34.76 | 0.51 |
| MVPCA | Ranking | 31.80 | 34.75 | 35.02 | 31.08 | 33.16 | 2.01 |
| Uniform | Baseline | 32.90 | 32.45 | 32.57 | 33.30 | 32.81 | 0.38 |
| Variance | Ranking | 33.23 | 30.81 | 30.04 | 30.82 | 31.22 | 1.39 |
| Entropy | Ranking | 25.40 | 24.50 | 31.91 | 26.52 | 27.08 | 3.32 |

**Findings**

- **The order holds on every date.** Search is best or tied best in all 24 cases (4 dates × 6 values of K), clustering in 7 and ranking in only 1.
- **GA reaches the true best on all 4 dates.** PSO does on 3 of 4: on 2026-10-27 it settles 0.35 dB lower. SFS is close but never reaches it.
- **Clustering is the steady runner-up**, 0.14–0.91 dB behind search at K = 4. Ranking depends on the day: MVPCA ties search on 2026-10-09 but is 4.77 dB behind on 2026-10-27, below the uniform baseline. Entropy is last on 3 of 4 dates.
- **Caveat:** each date's 512 × 512 window was chosen separately as its cleanest area (`data_2_check_s2.py`). The windows overlap but are shifted by up to 224 columns and 64 rows (about 4.5 km), so the dates also differ slightly in area, not only in time.

## Classification test (Indian Pines)

Can a computer still tell the 16 crop types apart using only the K channels that were sent?
`python darshita/step8_classification.py` (needs `data/darshita/Indian_pines_gt.mat` and
`pip install scikit-learn`) uses the same band choices as step 6. It trains an SVM (RBF kernel,
standardised channels, C = 100 for every method) on 10% of the labelled pixels and tests on the
other 90%, using one stratified split with seed 0.

| Method | Family | OA (%) | Kappa | OA behind all 200 bands (points) |
|---|---|---|---|---|
| All 200 bands | Reference | 81.98 | 0.794 | 0.00 |
| WaLuDi | Clustering | 76.02 | 0.725 | 5.96 |
| WaLuMI | Clustering | 75.60 | 0.721 | 6.38 |
| PSO | Search | 72.26 | 0.681 | 9.72 |
| Uniform | Baseline | 69.38 | 0.649 | 12.61 |
| GA | Search | 68.10 | 0.636 | 13.89 |
| AffinityProp | Clustering | 66.48 | 0.615 | 15.50 |
| SFS | Search | 63.08 | 0.577 | 18.91 |
| Variance | Ranking | 58.93 | 0.517 | 23.06 |
| MVPCA | Ranking | 58.17 | 0.503 | 23.82 |
| Entropy | Ranking | 56.34 | 0.490 | 25.65 |

**Findings**

- **Clustering gives the best accuracy.** With 10 bands (5% of the data), WaLuDi reaches 76.0% OA, 6.0 points below all 200 bands (82.0%). Clustering is best at every K from 5 to 25.
- **Rebuilding and classifying reward different bands.** Search wins on PSNR (rebuilding the image), but for telling crops apart, clustering's one-band-per-group spread works better.
- **Ranking is worst at small K** (56–59% OA at K = 10), because its bands are near-copies of each other.
- **Uniform spacing is a strong baseline when many bands are sent.** At K = 25 it reaches 77.9%, ahead of every selected set, and accuracy doesn't always rise with K. This is one split with a fixed SVM setting, so gaps of a few points shouldn't be over-read.
