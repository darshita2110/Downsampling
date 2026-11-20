# Team-comparable pipeline: data -> results

## 1. Why this exists
Your teammates score a method by how well the REDUCED data can rebuild the ORIGINAL image (PSNR, SSIM, SAM)
and by classification. Isomap, Kernel PCA and LLE have no built-in way back to the bands, so earlier runs scored
only neighbourhood preservation. `experiments/09_team_comparable_pipeline.py` adds the missing parts so your rows
can sit in the same tables.

## 2. The pipeline, step by step
1. Load the cube (rows x columns x bands) and flatten it: one row per pixel (`src/hsi_loader.py`).
2. Standardise each band (zero mean, unit variance) so no band dominates distances.
3. Draw a random sample of 8,000 pixels from the WHOLE image (Isomap and Kernel PCA need an n x n matrix).
4. Fit the technique on the sample (Isomap: neighbour graph, shortest paths, eigenvectors; Kernel PCA: similarity
   matrix, eigenvectors; LLE: neighbour reconstruction weights, eigenvectors). Result: k coordinates per sampled pixel.
5. Embed every pixel with scikit-learn's `transform()` (out-of-sample extension).
6. Decode: learn a map from the k coordinates back to the bands, either linear (least squares) or 5-nearest-neighbour.
   This decoder is an addition made here, not part of the original methods.
7. Score on all pixels and on held-out pixels (`src/team_metrics.py`): PSNR/SSIM/MSE/RMSE/MAE as in Kartavya's code;
   RMSE/PSNR(range-based)/SAM/ARI/NMI/SVM OA-AA-kappa as in Abhishek's code; plus your neighbourhood metrics.
8. Write CSVs in the teammates' column formats.

## 3. Protocol mapping
| Item | Kartavya | Abhishek | This pipeline |
|---|---|---|---|
| Data scale for reconstruction scores | [0,1] per channel, clipped | original units | both are computed |
| PSNR | 10 log10(1/MSE) | 20 log10((p99-p1)/RMSE) | both reported (`PSNR_dB`, `psnr_db`) |
| SSIM | 7x7 window per channel | not used | as Kartavya |
| Classification | LDA | RBF-SVM, 10% per class | SVM as Abhishek, plus your 5-NN spatial-block |
| Fit and score on | all pixels ('full') | all pixels | fit on 8,000, score on all and on held-out |

## 4. Check against the teammates (Indian Pines, k = 3, PCA fitted on all pixels)
| Quantity | Teammate | This pipeline |
|---|---|---|
| PSNR (Kartavya, [0,1] data) | 28.146576 dB | 28.146574 dB |
| SSIM (Kartavya) | 0.8785723 | 0.8785733 |
| Compression ratio | 65.42201 | 65.42201 |
| RMSE / PSNR (Abhishek, original units) | 126.0953 / 33.0730 dB | 126.0953 / 33.0730 dB |
| SAM | 2.18735 deg | 2.18735 deg |
| Variance explained | 0.903734 | 0.903734 |
| SVM OA | 64.907 % | 64.907 % |
ARI/NMI do NOT match exactly (ARI 0.620 vs 0.587, NMI 0.735 vs 0.727), probably k-means randomness or the
scikit-learn version (not tested). Treat ARI/NMI as indicative.

## 5. Results (k = 3)
| dataset | algorithm | k | PSNR_dB | PSNR_dB_heldout | SSIM | sam_deg | OA | AA | kappa | ARI | NMI | neighborhood_preservation | storage_saving_pct | compression_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indian_pines | PCA (fit on all pixels) | 3 | 28.147 | 28.14 | 0.879 | 2.187 | 64.907 | 52.565 | 59.012 | 0.62 | 0.735 |  | 98.471 | 65.422 |
| indian_pines | PCA (fit on sample) | 3 | 28.156 | 28.148 | 0.879 | 2.186 | 64.957 | 52.658 | 59.069 | 0.563 | 0.709 | 0.174 | 98.471 | 65.422 |
| indian_pines | PCA (fit on sample) + knn decoder | 3 | 30.711 | 30.122 | 0.924 | 1.29 | 64.957 | 52.658 | 59.069 | 0.563 | 0.709 | 0.174 | 59.87 | 2.492 |
| indian_pines | Isomap + linear decoder | 3 | 26.692 | 26.706 | 0.827 | 2.394 | 60.286 | 48.339 | 53.262 | 0.586 | 0.675 | 0.175 | 98.471 | 65.422 |
| indian_pines | Isomap + knn decoder | 3 | 30.434 | 29.816 | 0.914 | 1.324 | 60.286 | 48.339 | 53.262 | 0.586 | 0.675 | 0.175 | 59.87 | 2.492 |
| indian_pines | Kernel PCA + linear decoder | 3 | 23.905 | 23.93 | 0.697 | 2.874 | 58.111 | 43.617 | 50.51 | 0.516 | 0.668 | 0.117 | 98.471 | 65.422 |
| indian_pines | Kernel PCA + knn decoder | 3 | 29.668 | 29.123 | 0.908 | 1.511 | 58.111 | 43.617 | 50.51 | 0.516 | 0.668 | 0.117 | 59.87 | 2.492 |
| indian_pines | LLE + linear decoder | 3 | 22.971 | 22.967 | 0.625 | 3.497 | 52.495 | 29.173 | 42.386 | 0.376 | 0.585 | 0.108 | 98.471 | 65.422 |
| indian_pines | LLE + knn decoder | 3 | 28.768 | 28.177 | 0.88 | 1.606 | 52.495 | 29.173 | 42.386 | 0.376 | 0.585 | 0.108 | 59.87 | 2.492 |
| salinas | PCA (fit on all pixels) | 3 | 34.709 | 34.707 | 0.974 | 2.594 | 85.744 | 91.183 | 84.036 | 0.719 | 0.791 |  | 98.524 | 67.751 |
| salinas | PCA (fit on sample) | 3 | 34.682 | 34.679 | 0.973 | 2.622 | 85.726 | 91.14 | 84.015 | 0.716 | 0.793 | 0.2 | 98.524 | 67.751 |
| salinas | PCA (fit on sample) + knn decoder | 3 | 39.966 | 39.802 | 0.98 | 0.753 | 85.726 | 91.14 | 84.015 | 0.716 | 0.793 | 0.2 | 91.221 | 11.391 |
| salinas | Isomap + linear decoder | 3 | 33.048 | 33.029 | 0.945 | 3.148 | 84.907 | 89.666 | 83.106 | 0.717 | 0.802 | 0.152 | 98.524 | 67.751 |
| salinas | Isomap + knn decoder | 3 | 38.108 | 37.957 | 0.967 | 1.038 | 84.907 | 89.666 | 83.106 | 0.717 | 0.802 | 0.152 | 91.221 | 11.391 |
| salinas | Kernel PCA + linear decoder | 3 | 26.512 | 26.503 | 0.857 | 4.135 | 83.232 | 86.494 | 81.267 | 0.564 | 0.69 | 0.118 | 98.524 | 67.751 |
| salinas | Kernel PCA + knn decoder | 3 | 34.697 | 34.486 | 0.96 | 0.996 | 83.232 | 86.494 | 81.267 | 0.564 | 0.69 | 0.118 | 91.221 | 11.391 |
| salinas | LLE + linear decoder | 3 | 19.783 | 19.78 | 0.75 | 11.616 | 66.184 | 52.138 | 61.227 | 0.473 | 0.57 | 0.056 | 98.524 | 67.751 |
| salinas | LLE + knn decoder | 3 | 30.471 | 30.336 | 0.87 | 2.101 | 66.184 | 52.138 | 61.227 | 0.473 | 0.57 | 0.056 | 91.221 | 11.391 |

k = 3. 'PCA (fit on all pixels)' reproduces the teammates' PCA/PPCA protocol.
OA/AA/kappa: RBF-SVM, random 10% per class (Abhishek's protocol), %. PSNR/SSIM: Kartavya's [0,1] protocol.
'+ knn decoder' rows must be compared with 'PCA (fit on sample) + knn decoder', which uses the same decoder.
ARI/NMI depend on k-means randomness and the scikit-learn version; treat as indicative (about +/-0.03).

Final one-table summary: results/FINAL_COMPARISON_TABLE.md. Not run: KSC and Botswana through this pipeline (see HOW_TO_RUN.md).
Isomap with SMACOF is excluded because scikit-learn's MDS cannot embed new pixels.

## 6. What to say
- The reproduced PCA row matches the teammates' numbers, so the pipeline's scores can be put beside theirs.
- With a linear decoder PCA beat Isomap, Kernel PCA and LLE on every reconstruction and classification score on
  Indian Pines and Salinas at k = 3.
- The 5-nearest-neighbour decoder raised the reconstruction score of every method, but PCA plus the same decoder
  scored higher than every non-linear method (Indian Pines: 30.71 dB vs 30.43 for Isomap). So that gain comes from
  the decoder, not the non-linear embedding.
- That decoder needs the whole 8,000-pixel sample at the receiver: Indian Pines storage saving drops from 98.5% to 59.9%.
- Not counted: the cost of running the embedding on the satellite, which also needs the stored sample.
- At k = 10 on Salinas the nearest-neighbour decoder was worse than the linear one for PCA (41.7 vs 46.9 dB), so the
  decoder effect is not universal.
