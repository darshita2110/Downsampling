# 4. Isomap on real data: results, and a metric fix this run surfaced

Run by you, on your own machine, against the real `raichur_features.parquet`
(not reproduced in this sandbox - no access to the file; see
`documentation/03_real_data_loader.md`). Full console output and both results
JSONs are preserved in this project for the record.

## Run details
- Date: `2026-09-27` ("mostly clear, some small clouds" per
  `docs/DATA_ACCESS_GUIDE.md`); 10,901,746 valid pixels loaded, grid
  `(3358, 3247)` - consistent with the documented full-coverage figure, which
  is itself a useful confirmation that `src/data_loader.py` parses the real
  schema correctly, not just the mock.
- 6 channels used (B02, B03, B04, B08, NDVI, NDWI) - LST still unavailable upstream.
- Subsampled to 8,000 pixels (see `01_Isomap/experiments/05_isomap_on_real_data.py`
  docstring for why exact Isomap cannot run on ~10.9M pixels directly).
- `n_neighbors=12`, `n_components=3`, seed=42 - identical settings to the
  synthetic run, so the two are comparable.

## A bug this run helped surface: `explained_variance_ratio`

Both the synthetic run and this real run originally reported
`explained_variance_ratio: 1.0`. Seeing the exact same value on two very
different datasets was the signal that something was structurally wrong
rather than a genuine finding (a real "100% variance explained in 3 of 7
dimensions" on satellite reflectance data would itself be a surprising claim
to make without scrutiny - and the project brief specifically asks us not to
report metrics that aren't mathematically appropriate). Checked directly:
scikit-learn's `Isomap` fits its internal `KernelPCA` step with
`n_components` already fixed, so `model.kernel_pca_.eigenvalues_` only ever
contains exactly as many eigenvalues as were kept - there are no discarded
eigenvalues available to divide by, so any "kept / total" ratio computed from
it is 1.0 by construction, regardless of the data. Fixed in
`src/evaluation.explained_variance_ratio`: it now detects this truncation and
returns `None` (not computable with scikit-learn's implementation) rather
than a misleading 1.0. Both results JSONs have been corrected to `null` with
a note explaining the original value was a measurement artifact, not deleted
or silently replaced - the original value is preserved in this document and
in git history for transparency.

## Synthetic vs. real: side-by-side

| Metric | Synthetic (1,600 px) | Real (8,000 px subsample) | Note |
|---|---|---|---|
| Trustworthiness | 0.999 | 0.999 | Consistent - embeddings are locally faithful on both |
| Continuity | 0.999 | 0.999 | Consistent |
| Neighbourhood preservation (Jaccard) | 0.592 | 0.427 | **Lower on real data** - real spectral neighbourhoods are less clean-cut than our constructed synthetic manifold; expected, and a useful contrast rather than a problem to fix |
| Reconstruction error | 2.345 | 0.268 | **Not directly comparable across datasets** - this is classical MDS stress on standardised data with different sample sizes, dimensionality of spread, and k-NN graph structure; only comparable across runs on the *same* dataset (e.g. across `n_neighbors` values), which is how we will use it going forward |
| Explained variance ratio | not computable (None) | not computable (None) | See fix above |
| Fit time | 0.44-0.70 s | 14.1-15.2 s | 8,000/1,600 = 5x more pixels; 14/0.5 ~ 28-30x slower - worse than linear, consistent with Isomap's documented super-linear scaling (see `01_Isomap/documentation/02_isomap_stage4_5.md`) |
| Peak memory | 63 MB | 1,541 MB | ~24x more memory for 5x more pixels - again consistent with super-linear (here closer to quadratic, from the dense pairwise-distance step) scaling, not a bug |

**Reading on neighbourhood preservation:** the drop from 0.59 to 0.43 is the
most informative single number here. It says real Sentinel-2 spectral
neighbourhoods are messier than our synthetic Swiss-roll-style construction -
plausible, since real reflectance includes sensor noise, mixed pixels, and
(per `docs/DATA_ACCESS_GUIDE.md`) unmasked cloud/shadow pixels that can
masquerade as unrelated land cover. This is exactly the kind of real-data
behaviour synthetic validation alone cannot show us, and is worth carrying
into the eventual Section 5 discussion rather than treating 0.43 as "worse,
therefore wrong."

**Reading on fit time / memory scaling:** both scale noticeably worse than
linearly with pixel count, confirming (not just asserting) the complexity
concern already raised in `01_Isomap/documentation/02_isomap_stage4_5.md`. At 8,000
pixels we are already at ~14s and ~1.5GB; this makes the random-subsampling
choice in `01_Isomap/experiments/05_isomap_on_real_data.py` a practical necessity, not
a convenience, and strengthens the case for flagging landmark/approximate
Isomap variants as a research gap for the paper rather than a nice-to-have.

## Status
Isomap is now validated end-to-end: maths (Stage 1) -> basic example (Stage 2)
-> modular code (Stage 3) -> synthetic cube (Stage 4) -> evaluation (Stage 5)
-> **real Raichur data, loader confirmed working against the actual file**.
`01_Isomap/results/isomap_real_raichur.json` and `01_Isomap/results/isomap_synthetic_v1.json` are
both corrected and ready to feed into the eventual Section-5 comparison table
once the other five techniques reach the same point.
