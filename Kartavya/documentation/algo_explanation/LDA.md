# LDA for satellite-image channel reduction: implementation and suitability

## 1. LDA from the basics

**Supervised or unsupervised.** PCA and ICA only look at pixel values. LDA (Linear Discriminant
Analysis) is **supervised**: every training pixel needs a known **class label**, such as water,
crop, built-up or bare soil. LDA then looks for the few directions in channel space along which
the classes are easiest to tell apart.

**Fisher's idea.** Project the data onto a line. A good line places the class means far apart
(large between-class spread) while keeping each class tight (small within-class spread):

```
S_W = Σ_k Σ_{x∈k} (x − μ_k)(x − μ_k)ᵀ          within-class scatter
S_B = Σ_k N_k (μ_k − μ)(μ_k − μ)ᵀ              between-class scatter
maximise  J(w) = wᵀ S_B w / wᵀ S_W w   →   S_B w = λ S_W w   (keep the largest λ)
```

**The hard limit on components.** S_B is built from c class means that average to the overall
mean, so its rank is at most c − 1. LDA can therefore produce at most **min(c − 1, D)** components.
With 2 classes you get 1 component; with 4 classes, 3. The code refuses any larger K.

**What it is good and bad at.**
- LDA keeps only what separates the given classes. It is excellent for classification, and its
  components are the inputs a classifier needs.
- It throws away everything else, even high-variance information. There is **no inverse
  transform** and no image reconstruction, so PSNR and SSIM do not apply.
- Its quality depends entirely on the labels. Wrong or circular labels give a reduction that is
  confidently wrong.

**Implementation** (`lda.py`):
1. Whiten S_W with a symmetric eigensolver. A small shrinkage term (1e-4) handles the duplicated B04 and B03 channels, which make S_W singular.
2. Eigen-decompose the whitened S_B.
3. Fix the signs so results are reproducible.
4. Classify each pixel to the class whose mean is nearest in LDA space, with a log-prior term. With all components kept, this is the standard shared-covariance LDA classifier.

## 2. Dataset and label check (actual run, 13 Nov 2026)

**Images:** the inspection is the same as for PCA and ICA.
- Only 2026-09-17 is available locally (1 of 5 PNGs per folder).
- The PNGs are matplotlib figures. Each figure's map panel is cut out and resampled to a 1058 × 1024 grid.
- NDVI and NDWI are decoded from their colour maps.
- Alignment is confirmed: the duplicated bands match with r = 1.0000, and NDVI against pseudo-NDVI gives r = 0.891.
- There are 8 channels with a numerical rank of 6.

**Labels: none exist.** I searched:
- every file in the repository
- `data/processed/sentinel2/`, which holds only the RGB, False Color, NDVI and NDWI renderings
- the dataset schema in `docs/DATA_ACCESS_GUIDE.md`: `x, y, lat, lon, date, B02–B08, NDVI, NDWI`, and empty `LST`, `population_density`, `infrastructure_density`. There is no class column.
- the project files: the same four PNGs.

The only class-like item is the Sentinel-2 cloud mask (`MSK_CLASSI`) mentioned in `scripts/explore_s2.py`. It needs `.SAFE` products that aren't present, and a cloud / no-cloud mask is not a land-cover ground truth.

**Why supervised LDA cannot be evaluated here:**
1. **No labels means no between-class scatter.** S_B cannot be computed, so LDA has nothing to optimise.
2. **Labels made from these same pixels would be circular.** Thresholding NDVI or NDWI ("NDVI > 0.4 = vegetation") or clustering the pixels would only rebuild that rule. LDA would recover it almost perfectly, and the accuracy would measure agreement with the rule rather than with the ground. Clouds make this worse: in this scene they look like bare land in NDVI and like water in NDWI.
3. **Random or arbitrary labels** produce a model that runs but means nothing.

So `train.py` inspects the images, writes `outputs/metrics/lda/png/label_report.json` with the
reasons above, fits **no** model, and exits with code 3. `test.py` reads that report and refuses
to compute accuracy, precision, recall, F1 or a confusion matrix, also exiting with code 3. No
labels were invented.

## 3. Verification of the implementation

Because the Raichur data has no labels, the code was checked on two real labelled datasets bundled
with scikit-learn. These show only that the code is correct; they say nothing about satellite images.
The split is a stratified 70 / 30 split, seed 42.

| Dataset | K | Test accuracy | Macro F1 | Fisher ratio, original → LDA | Components vs scikit-learn |
|---|---|---|---|---|---|
| Iris (4 features, 3 classes) | 1 | 0.978 | 0.978 | 2.68 → 27.73 | \|r\| = 1.000000 |
| Iris | 2 | 1.000 | 1.000 | 2.68 → 15.75 | \|r\| = 1.000000 |
| Wine (13 features, 3 classes) | 1 | 0.906 | 0.909 | 0.86 → 8.47 | \|r\| = 1.000000 |
| Wine | 2 | 1.000 | 1.000 | 0.86 → 6.11 | \|r\| = 1.000000 |

- On the full datasets, discriminant ratios match scikit-learn to within 1e-12, and predictions agree 100 %.
- K = 3 was correctly refused for 3 classes.
- The Fisher ratio is highest at K = 1, because the first discriminant is the most separating one; adding the second averages in a weaker direction.
- The labelled Raichur code path (label loading, nearest-neighbour resampling, masking, the component limit, metrics and plots) was run once end to end. It used a deliberately synthetic block-pattern raster in a temporary folder outside the repository. Those numbers are not reported because the labels were fake.

## 4. How to make LDA usable on this project

1. Get an **independent** per-pixel land-cover map for the Raichur tile. Options:
   - ESA WorldCover 10 m, reprojected to EPSG:32643 and cropped to the scene bounds
   - Dynamic World
   - manual polygons digitised over the scene and rasterised
2. Save one integer raster per date as `data/labels/sentinel2/raichur_labels_<date>.npy` (or a single-band `.png`). Use 0 for unlabelled pixels. The raster must cover the same area as the images; the code checks the aspect ratio and resamples with nearest-neighbour.
3. Optionally, set the class names in `CLASS_NAMES` in `lda/config.py`.
4. Rerun `train.py` and `test.py`. Results go to `outputs/*/lda/png/`:
   - test accuracy, precision, recall and F1 per K
   - a confusion matrix
   - Fisher ratio before and after reduction
   - reduction %, storage, runtime and memory
   - the LD component images

**Expected component count:** with WorldCover's 11 classes, only those present in the scene count.
At most min(c − 1, 8) components are possible; in practice 5–7.

## 5. Suitability verdict

For this project's main goal, *reducing channels while preserving information for reconstruction*,
**LDA is not the right tool**. It needs labels the dataset doesn't have, it cannot reconstruct the
image, and it keeps at most c − 1 dimensions. Use PCA or ICA for reconstruction-oriented reduction.

LDA becomes the right tool when the goal changes to *reducing channels for a land-cover
classification task* and independent labels have been added.
