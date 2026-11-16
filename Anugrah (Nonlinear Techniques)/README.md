# Anugrah: Non-linear Feature Extraction for Spatial-Spectral Dimensionality Reduction

My part of the project: reduce the 7 Sentinel-2-derived channels (B02, B03, B04, B08,
NDVI, NDWI, LST) to a small number of non-linear latent features, using manifold
learning and non-linear PCA-family methods, and evaluate/compare them.

Reference literature: Vaddi et al., *Egyptian Journal of Remote Sensing and Space
Sciences*, Vol. 27 (2024), pp. 82-92.

Assigned techniques, in the order we will build them:

| # | Technique | Status |
|---|---|---|
| 1 | Isomap | Stages 1-5 done on synthetic data; real-data run pending |
| 2 | Isomap with SMACOF | not started |
| 3 | Kernel PCA | not started |
| 4 | Locally Linear Embedding (LLE) | not started |
| 5 | Constrained Laplacian Eigenmaps | not started |
| 6 | FHNPW (Fuzzy-embedded Hyperbolic Sigmoid Nonlinear PCA, Weighted LS) | not started - see note below |

## Relationship to the rest of the repository

This folder sits alongside `darshita/` (band-selection by ranking/search/clustering)
and the root `scripts/` (Sentinel-2 download + NDVI/NDWI/LST feature table for the
Raichur study area). We reuse that root pipeline's channel conventions
(`B02, B03, B04, B08, NDVI, NDWI`, with `LST` reserved but not yet integrated there
either) so that, eventually, all parts of the project can be run on the same
feature table. We do not modify `darshita/`, `scripts/`, or root files.

## Folder structure

```
Anugrah (Nonlinear Techniques)/
├── README.md                 <- this file, updated as we progress
├── requirements.txt          <- pinned dependencies for this subproject only
├── data/                     <- generated/downloaded data (gitignored; see below)
│   └── synthetic_v1/         <- our first, synthetic 7-channel dataset
├── notebooks/                <- exploratory notebooks (optional, not required)
├── src/                      <- the actual importable implementations
│   ├── data_loader.py        <- (not yet created) loading real/synthetic cubes
│   ├── preprocessing.py      <- (not yet created) scaling, NaN handling, flatten/reshape
│   ├── isomap.py             <- (not yet created) Technique 1
│   ├── kernel_pca.py         <- (not yet created) Technique 3
│   ├── lle.py                <- (not yet created) Technique 4
│   ├── laplacian_eigenmaps.py<- (not yet created) Technique 5
│   ├── fhnpw.py              <- (not yet created) Technique 6
│   └── evaluation.py         <- (not yet created) shared metrics
├── experiments/              <- runnable scripts, numbered in the order we build them
│   ├── 00_make_synthetic_dataset.py
│   └── 01_pca_sanity_check.py
├── results/                  <- saved metrics tables (CSV/JSON), one per experiment
├── visualisations/           <- saved figures (PNG)
└── documentation/            <- one write-up per technique, for the paper
    └── 01_concepts.md        <- non-linear DR intuition, PCA's limits, method comparison
```

`data/` is where generated/downloaded arrays live. Treat it like the rest of the
repo's `data/` folders: do not commit large files; keep only small synthetic data
or a `.gitignore` entry for anything derived from real Sentinel-2 downloads.

## Status: Stage 0 complete (environment + synthetic dataset)

### What exists right now

- `requirements.txt` - minimal pinned dependencies (numpy, scipy, scikit-learn,
  matplotlib, pandas). `scikit-fuzzy` is commented out until we reach FHNPW.
- `experiments/00_make_synthetic_dataset.py` - generates a 40x40x7 synthetic cube
  (`data/synthetic_v1/cube.npy`) whose 7 channels are known, by construction, to come
  from only 2 independent latent drivers wrapped through a non-linear ("Swiss roll")
  surface. This gives us a dataset where we *know* the ground-truth intrinsic
  dimensionality, so every technique's output can be sanity-checked against it before
  we ever touch real imagery.
- `experiments/01_pca_sanity_check.py` - runs plain PCA on that cube and saves
  `visualisations/01_pca_vs_truth.png`, showing concretely that PCA keeps 85% of the
  variance in 2 components yet does not recover the true roll structure (colours
  representing true position are scrambled in the PCA scatter). This is the evidence
  behind "why PCA is not enough", not just an assertion.
- `documentation/01_concepts.md` - the conceptual write-up: non-linear DR intuition,
  why PCA is insufficient here, and how Isomap / Kernel PCA / LLE / Laplacian
  Eigenmaps differ from each other.

### How to run what exists so far

```bash
cd "Anugrah (Nonlinear Techniques)"
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python experiments/00_make_synthetic_dataset.py
python experiments/01_pca_sanity_check.py
```

Expected output of the first command: a printed per-channel value summary and
`data/synthetic_v1/cube.npy` (shape `(40, 40, 7)`) plus ground-truth latents.
Expected output of the second: `visualisations/01_pca_vs_truth.png` and printed
PCA explained-variance numbers.

### A note on FHNPW

"Fuzzy-embedded Hyperbolic Sigmoid Nonlinear PCA with Weighted Least Squares" is a
specific, named algorithm from our reference paper's own citation trail, not a
standard scikit-learn method. Before writing any code for it we will:
1. Track down and read its original source publication (not just the name).
2. Write out its actual objective function and update rules as published.
3. Only then implement it - explicitly labelled as "reproduced from [citation]"
   or "our approximation of [citation], differing in X because Y", never as an
   invented algorithm that merely matches the name.

If, after a genuine search, the exact formulation cannot be verified from a
reliable source, we will document that as a limitation rather than fabricate a
plausible-looking substitute. This has not been investigated yet - it is queued
after Isomap, Kernel PCA, LLE and Laplacian Eigenmaps are done.

## Commitment: real data, not just synthetic

Synthetic data (`data/synthetic_v1/`) is only for building intuition and sanity-checking
each technique against a known ground truth. It is not a substitute for the real
comparison. Once each technique is implemented and validated on synthetic data, every
technique will also be run on real Sentinel-2-derived data pulled from this repo's own
pipeline (`scripts/main.py` output: `data/processed/ML_Dataset/combined/raichur_features.parquet`,
same 7 channels: B02, B03, B04, B08, NDVI, NDWI, LST), reshaped into `(Height, Width, 3)`
cubes for a chosen tile/date, and that real-data run is what will appear in the final
Section 5 comparison table across all six techniques - not the synthetic set. This will
need a `src/data_loader.py` that reads that Parquet file and reconstructs the
`(H, W, 7)` cube for a given date (LST is currently all-null upstream, so we will flag
that explicitly rather than silently drop or fabricate it, until it is integrated there).

### Isomap: Stage 1 (maths) and Stage 2 (basic example) — done

- `documentation/02_isomap.md` - full write-up: algorithm steps, objective
  function, hyperparameters, advantages/limitations, citation.
- `experiments/02_isomap_basic_example.py` - runs on the textbook Swiss roll
  (`sklearn.datasets.make_swiss_roll`), producing:
  - `visualisations/02a_isomap_vs_pca_swissroll.png` - Isomap unrolls the roll
    cleanly (Spearman correlation with true position: 1.000) where PCA loops it
    into a spiral (0.221).
  - `visualisations/02b_neighbor_graph.png` - the actual k-NN graph Isomap builds,
    shown hugging the roll's surface.
  - `visualisations/02c_short_circuit_failure.png` - `n_neighbors` swept over
    6/10/30/80, concretely reproducing the "short-circuit" failure mode (the
    embedding curls back into a loop once neighbours start jumping across
    windings of the roll).

```bash
python experiments/02_isomap_basic_example.py
```

### Isomap: Stage 3 (module), Stage 4 (applied to synthetic cube), Stage 5 (evaluation) — done

- `src/preprocessing.py` - shared cube<->pixel flattening, validation, standardising;
  used by every technique module so preprocessing stays identical across the
  whole comparison (Section 5 requirement).
- `src/isomap.py` - `fit_isomap_pixels` / `fit_isomap_cube`: a thin, validated,
  documented wrapper around `sklearn.manifold.Isomap` (exact reproduction, not a
  reimplementation - see module docstring), with timing/memory diagnostics and
  an explicit warning if the k-NN graph is disconnected.
- `src/evaluation.py` - shared metrics for all techniques: trustworthiness,
  continuity, neighbourhood preservation (Jaccard), reconstruction error (only
  when defined), explained-variance ratio (only when defined), timing, memory.
- `experiments/03_isomap_on_synthetic_cube.py` - applies Isomap to
  `data/synthetic_v1/cube.npy` (7 -> 3 components, `n_neighbors=12`, seed=42) and:
  - **Proves, not just asserts, that pixel-wise Isomap is spatially blind:**
    `visualisations/03c_spectral_vs_spatial_neighbors.png` shows a concrete
    example pixel whose 10 spectral nearest neighbours (what Isomap's graph
    actually uses) share **zero** overlap with its immediate 3x3 spatial
    neighbourhood.
  - Checks recovery of the known-ground-truth latents
    (`visualisations/03b_latent_recovery.png`) and catches a real subtlety: one
    true latent is cyclic (an angle), so it only shows up correctly across two
    embedding dimensions *jointly* (`cos`/`sin` correlation 0.94), not in either
    dimension's raw Spearman correlation (0.25) - documented honestly in
    `documentation/02_isomap_stage4_5.md` rather than reported as a failure.
  - Runs the full evaluation suite and saves `results/isomap_synthetic_v1.json`.
- `documentation/02_isomap_stage4_5.md` - full write-up of the above, plus
  computational-complexity/scalability discussion for a real satellite tile and
  a flagged research-gap note (spatially-weighted k-NN graph as a possible
  modified-Isomap extension, to be clearly labelled as a modification if built).

```bash
python experiments/03_isomap_on_synthetic_cube.py
```

**Isomap is now fully validated on synthetic data.** Per the "Commitment: real
data" section above, the version of this result that will actually go in the
Section-5 comparison table is the one run on real Raichur Sentinel-2 data - not
this synthetic run, which exists only to build and check our understanding.

### Real-data loader — built and schema-tested (not yet run on the real file)

- `src/data_loader.py` - reads the real Raichur Parquet dataset per
  `docs/DATA_ACCESS_GUIDE.md`'s documented schema. Returns valid pixels as a
  flat array (real tiles have partial, irregular coverage - not a clean
  rectangle like our synthetic cube) plus `(row, col)` grid positions;
  `reconstruct_grid()` scatters any per-pixel array back into a grid (NaN
  where uncovered) for visualisation. `get_grid_reference()` fixes a real bug
  our own test caught: deriving grid shape from a partial-coverage date's own
  extent silently under-sizes it - full details in
  `documentation/03_real_data_loader.md`.
- **The real 808 MB Parquet file is not available in this development
  environment** (gitignored; shared-folder link in `DATA_ACCESS_GUIDE.md` not
  yet filled in). So `data_loader.py` is verified here only against a mock
  Parquet file built to match the documented schema exactly
  (`experiments/04_test_data_loader_with_mock.py`, 6/6 checks pass) - running
  it against the real file is still an open step, to be done on a machine
  that has the data.
- `experiments/05_isomap_on_real_data.py` - ready to run once the data is in
  place (loads `2026-09-27`, subsamples 8,000 pixels - exact Isomap cannot run
  on a full ~10.9M-pixel date, see the complexity note above - and runs the
  same Isomap/evaluation code used throughout). Confirmed in this environment
  only that it fails gracefully with clear instructions when the data file is
  absent.
- Known, explicitly-flagged limitations carried into this first real-data
  pass: no cloud masking (none exists in the dataset), LST unavailable (6 of
  7 channels used), and random (not spatially stratified) subsampling.

```bash
# once docs/DATA_ACCESS_GUIDE.md's data is downloaded and in place:
python experiments/04_test_data_loader_with_mock.py   # schema self-test (no real data needed)
python experiments/05_isomap_on_real_data.py           # the real run
```

## Next step

Run `experiments/05_isomap_on_real_data.py` on a machine with the real Parquet
file in place, confirm the results look sane, then move to Technique 2 (Isomap
with SMACOF) using the same `src/` structure and evaluation suite.
