| Technique | Dataset | Pixels | Trust. | Contin. | Nbr. pres. | Time (s) | Memory (MB) | 5-NN OA (random) | 5-NN OA (spatial) | 5-NN kappa (spatial) |
|---|---|---|---|---|---|---|---|---|---|---|
| Isomap | Raichur | 8000 | 0.9987 | 0.9995 | 0.427 | 14.1 | 1541.1 | no labels | no labels | no labels |
| Isomap | Indian Pines | 8000 | 0.9876 | 0.9943 | 0.146 | 17.8 | 1540.8 | 0.621 | 0.414 | 0.328 |
| Isomap with SMACOF | Raichur | 2000 | 0.9969 | 0.9972 | 0.611 | 178.7 | 272.8 | no labels | no labels | no labels |
| Isomap with SMACOF | Indian Pines | MISSING |  |  |  |  |  |  |  |  |
| Kernel PCA | Raichur | 8000 | 0.9945 | 0.9987 | 0.293 | 1.75 | 1027.8 | no labels | no labels | no labels |
| Kernel PCA | Indian Pines | 8000 | 0.9779 | 0.9888 | 0.102 | 2.16 | 528.0 | 0.579 | 0.369 | 0.277 |
| LLE | Raichur | 8000 | 0.9949 | 0.9967 | 0.329 | 2.08 | 13.7 | no labels | no labels | no labels |
| LLE | Indian Pines | 8000 | 0.9564 | 0.9800 | 0.086 | 3.03 | 17.0 | 0.545 | 0.384 | 0.291 |
| PCA baseline | Indian Pines | 8000 | 0.9884 | 0.9942 | 0.153 | 0.15 | 1.63 | 0.665 | 0.435 | 0.352 |
| All 200 bands (no reduction) | Indian Pines | 8000 | - | - | - | - | - | 0.762 | 0.510 | 0.437 |

Notes:
- SMACOF used 2,000 pixels on both datasets; the others used 8,000. Do not rank SMACOF's neighbourhood preservation against them.
- Raichur uses 6 channels (LST unavailable); Indian Pines uses 200 bands. Metric values across datasets are not on a common scale; compare rankings within a dataset.
- Reconstruction error is omitted: it measures a different thing for each technique.
- 5-NN accuracy is on the 3-D embedding. 'spatial' = whole 29x29-pixel blocks held out; 'random' leaks neighbouring pixels across folds and is optimistic.
- Embeddings are fitted on all sampled pixels (labels never used); this compares methods fairly but is not an out-of-sample deployment test.