# LLE on real Raichur data (2026-09-27)

Run on the user's machine: 10,901,746 valid pixels loaded, grid (3358, 3247), 6 channels
(LST unavailable), 8,000 random pixels, k = 12, reg = 1e-3, 3 components, seed 42.

| Metric | Synthetic (1,600 px, 7 ch) | Real (8,000 px, 6 ch) |
|---|---|---|
| Trustworthiness | 0.9757 | 0.9949 |
| Continuity | 0.9840 | 0.9967 |
| Neighbourhood preservation | 0.259 | 0.329 |
| Native reconstruction error | 1.14e-07 | 1.08e-07 |
| Fit time (s) | 0.39 | 2.08 |
| Peak memory (MB) | 2.47 | 13.7 |

Findings
- The disconnect partly repeats: trustworthiness/continuity near 1, strict neighbourhood preservation low (0.329).
- Real-data ordering on neighbourhood preservation: Isomap 0.427 > LLE 0.329 > Kernel PCA 0.293. LLE is NOT the weakest on real data (it was on synthetic data).
- Cost is roughly linear in n (5x pixels: 5.3x time, 5.5x memory), consistent with no dense n x n matrix; two data points only.
- The reconstruction error is almost identical on two very different datasets. Hypothesis (untested): it is driven mainly by regularisation and eigenvalue selection. A reg sweep would test it.
- Reconstruction error is not comparable across techniques (see src/evaluation.py).
Caveats: no cloud masking; random subsample; single seed; synthetic vs real differ in size and channel count.
