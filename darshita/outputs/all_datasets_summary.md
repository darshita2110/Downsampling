# All datasets: summary

## Rebuild quality: PSNR (dB) at the main K

| Method | Family | indian_pines | Average rank |
|---|---|---|---|
| GA | Search | 30.45 | 1.00 |
| SFS | Search | 30.41 | 2.00 |
| PSO | Search | 30.20 | 3.00 |
| AffinityProp | Clustering | 29.76 | 4.00 |
| WaLuDi | Clustering | 29.60 | 5.00 |
| Uniform | Baseline | 29.47 | 6.00 |
| SegWard | Hybrid | 29.17 | 7.00 |
| WaLuMI | Clustering | 28.92 | 8.00 |
| Variance | Ranking | 25.62 | 9.00 |
| Entropy | Ranking | 24.82 | 10.00 |
| MVPCA | Ranking | 24.70 | 11.00 |

Family average rank (1 = best): Search 2.00, Clustering 5.67, Baseline 6.00, Hybrid 7.00, Ranking 10.00

## Rebuild quality: SSIM (0-1) at the main K

| Method | Family | indian_pines | Average rank |
|---|---|---|---|
| GA | Search | 0.9105 | 1.00 |
| PSO | Search | 0.9061 | 2.00 |
| SFS | Search | 0.9053 | 3.00 |
| AffinityProp | Clustering | 0.9048 | 4.00 |
| WaLuDi | Clustering | 0.8955 | 5.00 |
| Uniform | Baseline | 0.8943 | 6.00 |
| SegWard | Hybrid | 0.8932 | 7.00 |
| WaLuMI | Clustering | 0.8891 | 8.00 |
| Variance | Ranking | 0.7950 | 9.00 |
| MVPCA | Ranking | 0.7576 | 10.00 |
| Entropy | Ranking | 0.7450 | 11.00 |

Family average rank (1 = best): Search 2.00, Clustering 5.67, Baseline 6.00, Hybrid 7.00, Ranking 10.00

## Classification: AOA (%), OA averaged over K = 5..30

| Method | Family | indian_pines | Average rank |
|---|---|---|---|
| SegWard | Hybrid | 74.01 | 1.00 |
| WaLuDi | Clustering | 72.29 | 2.00 |
| Uniform | Baseline | 72.15 | 3.00 |
| WaLuMI | Clustering | 71.54 | 4.00 |
| PSO | Search | 70.38 | 5.00 |
| GA | Search | 68.19 | 6.00 |
| AffinityProp | Clustering | 66.13 | 7.00 |
| SFS | Search | 65.51 | 8.00 |
| Entropy | Ranking | 63.44 | 9.00 |
| Variance | Ranking | 62.69 | 10.00 |
| MVPCA | Ranking | 57.01 | 11.00 |

Family average rank (1 = best): Hybrid 1.00, Baseline 3.00, Clustering 4.33, Search 6.33, Ranking 10.00
