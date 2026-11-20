## indian_pines: classification with the chosen channels

16 classes, 10% of labelled pixels for training, 5 random splits (mean ± std).

### SVM, K = 10 of 200 channels

| Method | Family | OA (%) | AA (%) | Kappa | AOA over all K (%) |
|---|---|---|---|---|---|
| All 200 bands | Reference | 80.97 ± 0.53 | 73.37 | 0.782 | – |
| SegWard | Hybrid | 77.41 ± 0.57 | 71.90 ± 1.18 | 0.741 | 74.01 |
| WaLuDi | Clustering | 75.97 ± 0.37 | 69.19 ± 1.77 | 0.725 | 72.29 |
| WaLuMI | Clustering | 75.61 ± 0.33 | 70.23 ± 1.35 | 0.720 | 71.54 |
| PSO | Search | 72.20 ± 0.33 | 63.76 ± 1.72 | 0.681 | 70.38 |
| Uniform | Baseline | 70.40 ± 0.60 | 61.58 ± 1.09 | 0.660 | 72.15 |
| GA | Search | 69.22 ± 0.75 | 60.76 ± 0.71 | 0.648 | 68.19 |
| AffinityProp | Clustering | 67.31 ± 0.61 | 61.16 ± 2.31 | 0.624 | 66.13 |
| SFS | Search | 63.66 ± 0.52 | 54.88 ± 0.73 | 0.584 | 65.51 |
| Variance | Ranking | 58.95 ± 0.13 | 46.61 ± 0.22 | 0.518 | 62.69 |
| MVPCA | Ranking | 58.18 ± 0.74 | 43.52 ± 1.55 | 0.504 | 57.01 |
| Entropy | Ranking | 55.67 ± 0.57 | 42.24 ± 1.06 | 0.480 | 63.44 |
