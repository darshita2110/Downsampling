# Anugrah: Non-linear Feature Extraction for Spatial-Spectral Dimensionality Reduction

My part of the project: reduce the 7 Sentinel-2-derived channels (B02, B03, B04, B08,
NDVI, NDWI, LST) to a small number of non-linear latent features, using manifold
learning and non-linear PCA-family methods, and evaluate/compare them.

Reference literature: Vaddi et al., *Egyptian Journal of Remote Sensing and Space
Sciences*, Vol. 27 (2024), pp. 82-92.

Assigned techniques, in the order we will build them:

| # | Technique | Status |
|---|---|---|
| 1 | Isomap | Stage 1-2 done (maths + basic example); Stage 3 (module) next |
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

## Next step

Stage 3 of **Isomap**: a modular `src/isomap.py` (thin, documented wrapper around
`sklearn.manifold.Isomap` with input validation, fixed seeds, and docstrings tying
each parameter back to the maths in `documentation/02_isomap.md`), then Stage 4 -
applying it to our synthetic 7-channel cube (`data/synthetic_v1/`) with an
explicit check of whether/how spatial structure is or isn't preserved.
