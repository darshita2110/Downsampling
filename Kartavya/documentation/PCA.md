# PCA for satellite-image channel reduction

## 1. PCA from the basics

**The problem.** A satellite image has height × width pixels, and every pixel has C numbers
(channels). Here C = 8: three RGB levels, three false-colour levels, NDVI and NDWI. Storing or
sending all 8 numbers per pixel is wasteful if some of them repeat information. For example, red
and green brightness rise and fall together over most land.

**Pixels as points.** Think of every pixel as a point in an 8-dimensional space, one axis per
channel. With two channels you could draw it: a scatter plot of red against green. When the two
are strongly correlated, the points form a thin, tilted cloud. Almost all the spread is along one
diagonal direction, and very little is across it.

**The idea.** Rotate the axes so the first new axis points along the direction of greatest spread,
the second along the greatest remaining spread at a right angle to the first, and so on. These new
axes are the **principal components**. If the cloud is thin, the last few axes carry almost
nothing. Drop them and keep K < C numbers per pixel, losing very little.

**The maths, step by step** (X is the N × C matrix: N pixels, C channels):

1. **Centre** each channel: `Xc = X − mean`. PCA describes spread *around* the mean.
2. **Covariance matrix** `S = Xcᵀ Xc / (N − 1)`, size C × C. `S[i, j]` says how channels i and j vary together.
3. **Eigen-decomposition** `S v = λ v`. Each eigenvector v is a principal axis, and its eigenvalue λ is
   the variance of the data along that axis. Sort by λ, largest first.
4. **Explained variance ratio** of component i = `λᵢ / Σλ`. The running total says how much of the
   total variation the first K components keep.
5. **Reduce:** `Z = Xc Wᵀ`, where W holds the first K eigenvectors as rows. Each pixel now has K numbers.
6. **Reconstruct:** `X̂ = Z W + mean`. This is the best possible rebuild from K numbers by any linear
   projection: PCA minimises the mean squared reconstruction error (Eckart–Young theorem).

**Why one model serves every K.** The components are nested. The best 3-component model is the best
2-component model plus one more axis. So `train.py` fits once, and `test.py` keeps the first K rows for each K.

**Standardise or not?** If channels have very different spreads, PCA spends its first components on
the widest channel. Standardising (subtracting the mean and dividing by the standard deviation of
each channel, using training data only) gives every channel equal weight. Both options are
implemented, and the results below show the trade-off.

## 2. What the data really is (checked before preprocessing)

| Check | Finding |
|---|---|
| Format | All PNGs are RGBA uint8, alpha = 255 everywhere. They are **matplotlib figures** from `scripts/main.py`, with a title, white margins, and a colour bar on NDVI/NDWI. |
| Sizes | RGB and False_Color are 3024 × 2852; NDVI and NDWI are 2962 × 3181. **The same pixel position is not the same ground point across folders.** |
| Map panel | Found automatically as the largest mostly non-white block: 2887 × 2793 (composites) and 2825 × 2732 (indices). All panels are resampled (area average) to one 1058 × 1024 grid. |
| RGB / False Color | 8-bit display composites: (B04, B03, B02) and (B08, B04, B03), each with a 2–98 % stretch and clipping. They are **not original spectral bands**. |
| NDVI / NDWI | Derived indices drawn with the RdYlGn and RdBu colour maps over −1…1. They are decoded by matching every colour to the nearest of the 256 colour-map entries, giving a step of 0.0078. The decoded 2026-09-17 NDVI range is −0.356 to 0.801, against −0.352 to 0.804 in `reports/dataset_statistics`. NDWI is −0.691 to 0.480 against −0.689 to 0.478. The no-data share is 0.015 %, the same as the repo's excluded-pixel count. |
| Alignment | RGB:R and False_Color:G are both B04 with the same stretch, and after resampling they are identical (r = 1.0000, best shift 0). NDVI against a pseudo-NDVI from the False_Color levels: best shift (0, 0), r = 0.891. The grids line up. |
| Redundancy | Because B04 and B03 appear in both composites, the 8 channels contain only **6 distinct signals**. PCA finds exactly this: components 7 and 8 have zero variance. |
| Clouds | 2026-09-17 is heavily clouded. Nothing marks clouds, so they stay in the data, as the repo's guide warns. |

Ordering: within each folder, files are sorted by full filename in ascending lexicographic order.
The index is the position in that list. The date in each filename is checked to agree across the
four folders for each index. The mapping is saved in `metrics/filename_index_mapping.json`.

## 3. Pipeline

1. Sort, map and inspect the files. Crop each figure's map panel and resample it to the common grid.
2. Decode NDVI and NDWI from their colours. Set no-data pixels to invalid.
3. Normalise to [0, 1] using fixed ranges: composites ÷ 255, indices (v + 1) / 2. The same scale applies on every date.
4. Build the feature matrix: valid pixels × 8 channels.
5. Standardise with training statistics, then fit PCA (NumPy `eigh` on the 8 × 8 covariance), with a fixed sign convention.
6. Check against scikit-learn's PCA.
7. For K = 1…8: forward transform, reshape to an H × W × K image, inverse transform, reshape to H × W × 8, clip to [0, 1], compute metrics.

**Split.** With 5 dates, indices 0–2 train and 3–4 test. With only one date, the image is split
spatially: the top half trains and the bottom half tests, so test pixels are never seen during fitting.

**Storage** counts float32 values. Reduced = pixels × K × 4 bytes. The model (PCA mean,
K × 8 projection, standardiser mean and std) is added, because a receiver needs it to reconstruct.

## 4. Results (actual run, 12 Nov 2026)

**Data available for this run:** only the 2026-09-17 image from each of the four folders. The
other four dates were not in the repository or the project files. So this is a spatial split
(fit on 541,696 pixels in the top half, test on 541,531 pixels in the bottom half), not the
planned 3-date / 2-date split. Rerun both scripts once all 20 PNGs are in place.

### Explained variance (standardised, training pixels)

| PC | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Individual % | 73.30 | 24.13 | 1.89 | 0.47 | 0.13 | 0.08 | 0.00 | 0.00 |
| Cumulative % | 73.30 | 97.43 | 99.32 | 99.79 | 99.92 | 100.00 | 100.00 | 100.00 |

2 components reach 90 % and 95 %; 3 reach 99 %. Without standardisation: 86.70, 11.11, 1.89, 0.15, 0.14, 0.01, 0, 0.

### Test-region reconstruction (standardised)

| K | Reduction | MSE | RMSE | MAE | PSNR (dB) | SSIM | Storage (MB) | Ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 87.5 % | 0.00732 | 0.0856 | 0.0539 | 21.36 | 0.813 | 17.33 → 2.17 | 8.0× |
| 2 | 75.0 % | 0.00129 | 0.0360 | 0.0231 | 28.88 | 0.968 | 17.33 → 4.33 | 4.0× |
| 3 | 62.5 % | 0.00030 | 0.0173 | 0.0108 | 35.23 | 0.985 | 17.33 → 6.50 | 2.7× |
| 4 | 50.0 % | 0.00010 | 0.0101 | 0.0064 | 39.91 | 0.995 | 17.33 → 8.66 | 2.0× |
| 5 | 37.5 % | 0.00003 | 0.0054 | 0.0036 | 45.29 | 0.997 | 17.33 → 10.83 | 1.6× |
| 6–8 | 25.0 % or less | ~1e-33 | 0 | 0 | >300 (lossless) | 1.000 | 13.00 or more | 1.3× or less |

Errors are on the 0–1 scale; an RMSE of 0.036 is about 9 of 255 grey levels. "Lossless" means equal to floating-point precision.

### Per product, PSNR (dB): standardised / not standardised

| K | RGB | False_Color | NDVI | NDWI |
|---|---|---|---|---|
| 1 | 25.6 / 25.8 | 18.3 / 19.8 | 25.4 / 24.6 | 25.4 / 25.0 |
| 2 | 26.6 / 26.8 | 29.3 / 31.1 | 37.2 / 33.8 | 39.2 / 34.5 |
| 3 | 35.7 / 38.3 | 33.0 / 40.4 | 44.0 / 35.8 | 42.8 / 35.1 |
| 4 | 37.4 / 46.0 | 41.0 / 46.1 | 46.2 / 36.8 | 51.0 / 35.8 |

Overall, the version without standardisation scores higher (K = 3: 37.9 dB against 35.2 dB),
because plain PCA minimises exactly the overall MSE. Standardisation moves accuracy to the
low-variance index channels. At K = 3, NDVI improves from 35.8 dB to 44.0 dB and NDWI from
35.1 dB to 42.8 dB. Choose based on which products matter downstream.

### Time and memory (2-core cloud VM, 541k test pixels, median of 5 runs)

* PCA fit: 0.02–0.03 s, peak 35 MB. Loading and preprocessing the four figures takes most of the time: 6.5 s, peak 523 MB.
* Forward transform: 9–15 ms. Inverse: 21–29 ms. Inverse peak memory: 104 MB. Process RSS: 0.44–0.65 GB.

### Takeaways

* **K = 2 keeps 97 % of the variance and gives SSIM 0.97 at a 4× size reduction. K = 3 gives 99 % and SSIM 0.985.**
* The 8 PNG channels have an intrinsic rank of 6: two channels are duplicates. A fairer baseline would drop the duplicates first (6 channels).
* PCA here works on display products: stretched, clipped 8-bit composites and colour-quantised indices. Use the `raw` experiment on the original reflectance bands for physically meaningful conclusions.
