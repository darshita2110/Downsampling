# Easy comparison: PCA vs Isomap vs Kernel PCA vs LLE

Every number comes from the saved runs (one random sample of 8,000 pixels, seed 42). 'k' = how many numbers are kept per pixel.

## PSNR (dB)  (higher is better)

**Indian Pines**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **28.16** | **29.01** | **30.98** | **33.27** |
| Isomap | 26.69 | 27.43 | 28.06 | 28.53 |
| Kernel PCA | 23.91 | 25.11 | 28.14 | 29.53 |
| LLE | 22.97 | 24.76 | 26.48 | 27.86 |

**Salinas**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **34.68** | **41.90** | **46.86** | **50.50** |
| Isomap | 33.05 | 34.14 | 35.14 | 35.61 |
| Kernel PCA | 26.51 | 28.98 | 32.52 | 36.83 |
| LLE | 19.78 | 25.30 | 31.06 | 35.16 |

## SSIM  (higher is better)

**Indian Pines**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **0.879** | **0.901** | **0.924** | **0.948** |
| Isomap | 0.827 | 0.844 | 0.856 | 0.866 |
| Kernel PCA | 0.697 | 0.749 | 0.864 | 0.894 |
| LLE | 0.625 | 0.727 | 0.804 | 0.860 |

**Salinas**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **0.973** | **0.988** | **0.995** | **0.997** |
| Isomap | 0.945 | 0.944 | 0.947 | 0.949 |
| Kernel PCA | 0.857 | 0.895 | 0.948 | 0.976 |
| LLE | 0.750 | 0.849 | 0.912 | 0.937 |

## SAM (deg)  (LOWER is better)

**Indian Pines**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **2.19** | **1.96** | **1.63** | **1.36** |
| Isomap | 2.39 | 2.17 | 2.01 | 1.90 |
| Kernel PCA | 2.87 | 2.61 | 1.89 | 1.61 |
| LLE | 3.50 | 2.79 | 2.36 | 1.98 |

**Salinas**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **2.62** | **0.98** | **0.60** | **0.35** |
| Isomap | 3.15 | 2.69 | 2.44 | 2.35 |
| Kernel PCA | 4.14 | 2.23 | 1.86 | 1.05 |
| LLE | 11.62 | 6.08 | 1.96 | 1.53 |

## SVM accuracy (%)  (higher is better)

**Indian Pines**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **65.0** | **68.2** | **74.2** | **76.0** |
| Isomap | 60.3 | 63.3 | 64.7 | 64.2 |
| Kernel PCA | 58.1 | 63.0 | 68.6 | 72.4 |
| LLE | 52.5 | 54.0 | 57.2 | 58.2 |

**Salinas**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **85.7** | **89.8** | **90.9** | **91.9** |
| Isomap | 84.9 | 85.8 | 86.3 | 86.7 |
| Kernel PCA | 83.2 | 87.4 | 88.8 | 89.8 |
| LLE | 66.2 | 76.5 | 80.5 | 83.6 |

## Neighbourhood preservation  (higher is better)

**Indian Pines**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | 0.174 | **0.225** | **0.342** | **0.520** |
| Isomap | **0.175** | 0.206 | 0.225 | 0.240 |
| Kernel PCA | 0.117 | 0.129 | 0.188 | 0.266 |
| LLE | 0.108 | 0.121 | 0.151 | 0.164 |

**Salinas**

| Method | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| PCA | **0.200** | **0.315** | **0.687** | **0.925** |
| Isomap | 0.152 | 0.200 | 0.232 | 0.265 |
| Kernel PCA | 0.118 | 0.147 | 0.177 | 0.388 |
| LLE | 0.056 | 0.076 | 0.110 | 0.134 |

## Scoreboard: how many of the contests each method won

A contest = one measure, one scene, one k (40 contests in total).

| Method | Contests won |
|---|---|
| PCA | 39 of 40 |
| Isomap | 1 of 40 |
| Kernel PCA | 0 of 40 |
| LLE | 0 of 40 |

## Average gap to PCA (positive = better than PCA)

| Method | PSNR (dB) | SSIM | SVM accuracy (points) | Neighbourhood preservation |
|---|---|---|---|---|
| Isomap | -5.84 | -0.053 | -5.7 | -0.212 |
| Kernel PCA | -7.98 | -0.090 | -3.8 | -0.232 |
| LLE | -10.25 | -0.142 | -14.1 | -0.309 |

## What each method costs (8,000-pixel fit, development machine, single core)

| Method | Fit time (s), Indian Pines | Memory (MB) | Exact way back to the bands? | Needs a stored sample on the receiver to embed new pixels? |
|---|---|---|---|---|
| PCA | 0.0 | 1 | Yes | No (just 3 x 200 numbers) |
| Isomap | 22.4 | 2053 | No, needs a learned decoder | Yes |
| Kernel PCA | 1.5 | 1040 | No, needs a learned decoder | Yes |
| LLE | 14.4 | 577 | No, needs a learned decoder | Yes |

LLE uses the dense solver here; with the default solver on our pinned libraries it broke on Indian Pines (see HOW_TO_RUN.md section 4).
Times and memory are from one machine and one run each.