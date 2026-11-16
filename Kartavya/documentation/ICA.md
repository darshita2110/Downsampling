# ICA for satellite-image channel reduction

## 1. ICA from the basics

**The model.** ICA assumes that each observed channel is a weighted mix of a few hidden *sources*:
`x = A s + mean`. Here x holds the C channels of a pixel, s holds K hidden sources, and A is the
C × K *mixing matrix*. The classic example is the cocktail party: two microphones each record a
mix of two speakers, and ICA separates the voices. For an image, sources might be things like
"vegetation", "water" or "cloud brightness", each spread over several channels.

**What ICA looks for.** ICA estimates an *unmixing* matrix W, so that `s = W(x − mean)`, and
chooses W to make the recovered sources **statistically independent**.
- *Uncorrelated*, which is what PCA gives, only rules out linear relationships.
- *Independent* rules out every relationship. Knowing one source tells you nothing about the others.

**Why non-Gaussianity matters.** The Central Limit Theorem says that adding independent signals
together makes the result more Gaussian (more bell-shaped). So reversing the mixing means finding
the directions whose projections are the *least* Gaussian. FastICA measures this with the
"logcosh" contrast: `G(u) = log cosh(u)`, whose derivative is `g(u) = tanh(u)`.

**FastICA, step by step** (X is the N × C pixel matrix):

1. **Centre**: subtract each channel's mean.
2. **Whiten**: run PCA, keep the top K axes, and scale each to unit variance:
   `Xw = Xc E_K D_K^(−1/2)`. **This is where the dimension drops from C to K.**
3. **Rotate**: find an orthogonal K × K matrix R that makes the columns of `Xw Rᵀ` as non-Gaussian as possible. The fixed-point update is
   `R ← E[g(R xw) xwᵀ] − diag(E[g′(R xw)]) R`, followed by `R ← (R Rᵀ)^(−1/2) R` to keep the rows
   orthonormal. Repeat until no row turns by more than `tol`.
4. **Reduce**: `S = (X − mean) Wᵀ`, where `W = R D_K^(−1/2) E_Kᵀ`.
5. **Reconstruct**: `X̂ = S Aᵀ + mean`, where `A = E_K D_K^(1/2) Rᵀ`, the pseudo-inverse of W.

**Key consequence.** Step 3 is only a *rotation* inside the K-dimensional PCA subspace from step 2.
So the reconstruction from K ICA components is *exactly* the K-component PCA reconstruction. ICA
does not compress better than PCA. What it changes is what each stored channel *means*. Its other
differences from PCA:
- Its components have no natural order. Here they are sorted by the energy of their mixing column.
- Their signs are arbitrary. Here the largest mixing weight is made positive.
- Models for different K are not nested, so each K is fitted separately.

## 2. Data checks

The checks are the same as for PCA (see `PCA.md`, section 2):
- The PNGs are matplotlib figures, so each figure's map panel is cut out and resampled to one 1058 × 1024 grid.
- NDVI and NDWI are decoded from their colour maps. The decoded values agree with the repo's own statistics to within 0.004.
- Alignment is confirmed: the duplicated bands match exactly (r = 1.0000), and NDVI against pseudo-NDVI peaks at zero shift (r = 0.891).
- B04 and B03 each appear in two products, so the 8 channels have a **numerical rank of 6**. ICA cannot whiten zero-variance directions (it would divide by about 0), so K runs from 1 to 6. The code computes the rank and enforces this limit.

## 3. Train / test discipline

- `train.py` loads **only the training pixels**. It fits the standardiser (mean and std) and every ICA model on them, then saves `ica_K{k}.npz`.
- `test.py` **only loads** these files. Nothing is refitted.
- With one date, the top half of the image trains and the bottom half tests. With five dates, indices 0–2 train and 3–4 test.

**Hyperparameters**, all recorded in `train_summary.json`:

| Setting | Value |
|---|---|
| Algorithm | parallel (symmetric) FastICA |
| Contrast | logcosh, alpha = 1.0 |
| Whitening | PCA, unit variance |
| tol | 1e-4 |
| max_iter | 1000 |
| Seed | 42 |
| Fit sample | 300,000 random training pixels |
| Standardisation | on by default; `--no-standardize` for comparison |

## 4. Results (actual run, 13 Nov 2026)

**Data available:** only the 2026-09-17 image from each folder. The split was spatial:
541,696 training pixels, of which 300,000 were sampled for fitting, and 541,531 test pixels.

### Convergence

| K | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Standardised: iterations (converged?) | 1 ✓ | 3 ✓ | 51 ✓ | 9 ✓ | 12 ✓ | 7 ✓ |
| Fit time (s) | 0.02 | 0.07 | 1.11 | 0.20 | 0.33 | 0.25 |
| Fit peak memory (MB) | 31 | 48 | 62 | 77 | 91 | 106 |
| Not standardised: iterations | 1 ✓ | 16 ✓ | 20 ✓ | 14 ✓ | **1000 ✗** | 8 ✓ |

Without standardisation, K = 5 never reaches the tolerance: it oscillates at a change of about 0.039.
scikit-learn's FastICA also stops at its 1000-iteration limit on the same data. A likely cause is
that eigenvalues 4 and 5 (0.00072 and 0.00063) are almost equal and close to Gaussian, which leaves
the rotation poorly defined. Reconstruction is not affected, because any rotation gives the same
reconstruction. Standardising avoids the problem.

Cross-check with scikit-learn FastICA: in the standardised run, every K matches component for
component (lowest |r| 0.9975), with reconstruction differences of about 1e-12. In the
unstandardised run K = 4 matches with |r| ≥ 0.83: ICA on near-Gaussian data has several
near-equivalent solutions.

### Reconstruction on the test region (standardised)

| K | Reduction | RMSE | MAE | PSNR (dB) | SSIM | Storage (MB) |
|---|---|---|---|---|---|---|
| 1 | 87.5 % | 0.0856 | 0.0539 | 21.35 | 0.813 | 17.33 → 2.17 |
| 2 | 75.0 % | 0.0360 | 0.0231 | 28.88 | 0.968 | 17.33 → 4.33 |
| 3 | 62.5 % | 0.0173 | 0.0108 | 35.23 | 0.985 | 17.33 → 6.50 |
| 4 | 50.0 % | 0.0101 | 0.0064 | 39.90 | 0.995 | 17.33 → 8.66 |
| 5 | 37.5 % | 0.0054 | 0.0036 | 45.28 | 0.997 | 17.33 → 10.83 |
| 6 | 25.0 % | 0 | 0 | >300 (lossless) | 1.000 | 17.33 → 13.00 |

The largest difference from the PCA projection is 0 to about 1e-14 at every K: the reconstruction is the same, as the theory predicts.
Without standardisation, PSNR is 22.43, 29.28, 37.93, 41.10, 51.46 dB for K = 1–5.
Unmixing takes 14–24 ms and mixing 35–46 ms for 541k pixels (median of 5 runs). Forward peak memory is 39–61 MB.

### Are the components more independent than PCA's?

This compares, within the same subspace, the ICA sources with the whitened PCA axes on test pixels.

| K | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Mean \|excess kurtosis\|, ICA / PCA | 0.38 / 0.24 | 0.69 / 0.44 | 2.01 / 1.18 | 5.00 / 2.20 | 5.67 / 2.05 |
| Mean \|corr(sᵢ², sⱼ²)\|, ICA / PCA | 0.057 / 0.039 | 0.158 / 0.085 | 0.120 / 0.176 | 0.092 / 0.157 | 0.071 / 0.174 |

- ICA always finds less Gaussian components. That is what it optimises.
- From K = 4 upward it also removes more than half of the non-linear dependence left in PCA's axes.
- At K = 2–3 the sources are actually *more* dependent than PCA's. These display products fit the "independent sources" model poorly when only a few components are kept: clouds, for example, brighten every channel together.

## 5. Takeaways

* For compression, ICA performs the same as PCA, by construction.
* Choose ICA when the stored channels need to be *interpretable, separated signals*. With K = 6 the mixing weights (`graphs/ica/png/ica_mixing_K6.png`) separate roughly into brightness, NIR/vegetation (IC3, NDVI-driven), water (IC4, NDWI-driven) and NIR-only (IC6) sources.
* Use standardisation. Without it, some K values fail to converge.
* Rerun with all five dates for a real date-to-date test, and with the raw bands (`--dataset raw`) for physically meaningful sources.
