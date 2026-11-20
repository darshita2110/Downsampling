| Dataset | Method | Pixels | Trust. | Contin. | Nbr. pres. | Time (s) | Mem (MB) | OA random | OA spatial | AA spatial | Kappa spatial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| indian_pines | PCA baseline | 8000 | 0.9884 | 0.9942 | 0.153 | 0.02 | 1.6 | 0.665 | 0.435 | 0.288 | 0.352 |
| indian_pines | Isomap | 8000 | 0.9876 | 0.9943 | 0.146 | 14.88 | 1540.8 | 0.621 | 0.414 | 0.268 | 0.328 |
| indian_pines | Kernel PCA | 8000 | 0.9779 | 0.9888 | 0.102 | 1.92 | 528.0 | 0.579 | 0.369 | 0.240 | 0.277 |
| indian_pines | LLE | 8000 | 0.9548 | 0.9784 | 0.083 | 42.62 | 577.6 | 0.541 | 0.381 | 0.262 | 0.288 |
| indian_pines | All bands (reference) | 8000 | - | - | - | - | - | 0.762 | 0.510 | 0.346 | 0.437 |
| salinas | PCA baseline | 8000 | 0.9896 | 0.9951 | 0.148 | 0.02 | 1.7 | 0.871 | 0.746 | 0.790 | 0.715 |
| salinas | Isomap | 8000 | 0.9873 | 0.9933 | 0.127 | 15.40 | 1540.4 | 0.844 | 0.738 | 0.796 | 0.707 |
| salinas | Kernel PCA | 8000 | 0.9705 | 0.9895 | 0.104 | 1.91 | 528.2 | 0.831 | 0.711 | 0.747 | 0.676 |
| salinas | LLE | 8000 | 0.9755 | 0.9844 | 0.072 | 44.98 | 577.6 | 0.836 | 0.738 | 0.793 | 0.707 |
| salinas | All bands (reference) | 8000 | - | - | - | - | - | 0.876 | 0.765 | 0.826 | 0.737 |
| ksc | PCA baseline | 5211 | 0.9899 | 0.9975 | 0.410 | 0.01 | 1.3 | 0.860 | 0.531 | 0.372 | 0.479 |
| ksc | Isomap | 5211 | 0.9888 | 0.9957 | 0.235 | 5.51 | 654.6 | 0.835 | 0.495 | 0.359 | 0.441 |
| ksc | Kernel PCA | 5211 | 0.9823 | 0.9954 | 0.231 | 0.63 | 226.6 | 0.806 | 0.563 | 0.384 | 0.513 |
| ksc | LLE | 5211 | 0.8875 | 0.9656 | 0.075 | 11.18 | 245.4 | 0.581 | 0.339 | 0.230 | 0.269 |
| ksc | All bands (reference) | 5211 | - | - | - | - | - | 0.924 | 0.573 | 0.406 | 0.524 |
| botswana | PCA baseline | 8000 | 0.9959 | 0.9986 | 0.295 | 0.01 | 0.9 | no labels | no labels | no labels | no labels |
| botswana | Isomap | 8000 | 0.9921 | 0.9961 | 0.171 | 15.50 | 1540.4 | no labels | no labels | no labels | no labels |
| botswana | Kernel PCA | 8000 | 0.9538 | 0.9856 | 0.087 | 1.80 | 524.4 | no labels | no labels | no labels | no labels |
| botswana | LLE | 8000 | 0.9523 | 0.9826 | 0.068 | 39.73 | 577.6 | no labels | no labels | no labels | no labels |

Mean rank (1 = best) across scenes; lower is better:

| Method | Neighbourhood preservation (4 scenes) | Spatial-block accuracy (3 labelled scenes) | Peak memory (4 scenes) |
|---|---|---|---|
| PCA baseline | 1.00 (n=4) | 1.33 (n=3) | 1.00 (n=4) |
| Isomap | 2.00 (n=4) | 2.33 (n=3) | 4.00 (n=4) |
| Kernel PCA | 3.00 (n=4) | 3.00 (n=3) | 2.00 (n=4) |
| LLE | 4.00 (n=4) | 3.33 (n=3) | 3.00 (n=4) |

Notes: SMACOF excluded from ranks (2,000-pixel sample). Timings: single machine, one run.
Botswana has no ground truth supplied, so only neighbourhood metrics exist for it.
Pavia University could not be run: only its ground-truth file was supplied, not the image cube.