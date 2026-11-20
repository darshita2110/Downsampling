## indian_pines: 10 of 200 channels (5% of the data)

| Method | Family | PSNR (dB) | SSIM | SAM (°) | Redundancy | Time (s) | Channels sent |
|---|---|---|---|---|---|---|---|
| GA | Search | 30.45 | 0.9105 | 4.08 | 0.331 | 4.27 | ch 0 ch 14 ch 27 ch 51 ch 60 ch 82 ch 89 ch 118 ch 161 ch 199 |
| SFS | Search | 30.41 | 0.9053 | 4.05 | 0.361 | 0.32 | ch 0 ch 9 ch 27 ch 58 ch 60 ch 88 ch 103 ch 128 ch 153 ch 199 |
| PSO | Search | 30.20 | 0.9061 | 4.19 | 0.382 | 1.14 | ch 0 ch 9 ch 24 ch 28 ch 49 ch 60 ch 90 ch 93 ch 127 ch 154 |
| AffinityProp | Clustering | 29.76 | 0.9048 | 4.44 | 0.290 | 3.22 | ch 0 ch 9 ch 52 ch 63 ch 89 ch 90 ch 91 ch 95 ch 100 ch 173 |
| WaLuDi | Clustering | 29.60 | 0.8955 | 4.46 | 0.614 | 0.11 | ch 0 ch 6 ch 17 ch 18 ch 20 ch 25 ch 52 ch 98 ch 122 ch 163 |
| Uniform | Baseline | 29.47 | 0.8943 | 4.54 | 0.432 | 0.00 | ch 0 ch 22 ch 44 ch 66 ch 88 ch 111 ch 133 ch 155 ch 177 ch 199 |
| SegWard | Hybrid | 29.17 | 0.8932 | 4.68 | 0.596 | 1.36 | ch 9 ch 25 ch 34 ch 47 ch 67 ch 91 ch 132 ch 175 ch 190 ch 199 |
| WaLuMI | Clustering | 28.92 | 0.8891 | 4.84 | 0.554 | 5.19 | ch 1 ch 13 ch 24 ch 49 ch 67 ch 78 ch 91 ch 100 ch 122 ch 165 |
| Variance | Ranking | 25.62 | 0.7950 | 5.84 | 0.876 | 0.03 | ch 22 ch 23 ch 24 ch 25 ch 26 ch 27 ch 28 ch 29 ch 31 ch 41 |
| Entropy | Ranking | 24.82 | 0.7450 | 6.30 | 0.984 | 0.05 | ch 30 ch 107 ch 108 ch 109 ch 110 ch 111 ch 112 ch 113 ch 114 ch 115 |
| MVPCA | Ranking | 24.70 | 0.7576 | 6.36 | 0.995 | 0.93 | ch 112 ch 113 ch 114 ch 115 ch 116 ch 117 ch 118 ch 163 ch 164 ch 165 |

GA / PSO over 5 seeds: GA 30.44 ± 0.08 dB, PSO 30.20 ± 0.08 dB

Best PSNR (dB) of each family at every K:

| Family | K=5 | K=10 | K=15 | K=20 | K=25 | K=30 |
|---|---|---|---|---|---|---|
| Clustering | 28.59 | 29.76 | 30.42 | 31.74 | 32.73 | 33.37 |
| Hybrid | 28.57 | 29.17 | 30.23 | 31.28 | 31.99 | 32.28 |
| Ranking | 24.27 | 25.62 | 25.91 | 26.27 | 26.80 | 28.80 |
| Search | 29.05 | 30.45 | 31.58 | 32.53 | 33.55 | 34.58 |
