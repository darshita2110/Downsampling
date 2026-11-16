# Anugrah: Non-linear Feature Extraction for Spatial-Spectral Dimensionality Reduction

My part of the project: reduce the 7 Sentinel-2-derived channels (B02, B03, B04, B08,
NDVI, NDWI, LST) to a small number of non-linear latent features, using manifold
learning and non-linear PCA-family methods, and evaluate/compare them.

Reference literature: Vaddi et al., *Egyptian Journal of Remote Sensing and Space
Sciences*, Vol. 27 (2024), pp. 82-92.

Assigned techniques, in the order we will build them:

| # | Technique | Status |
|---|---|---|
| 1 | Isomap | **Done**: Stages 1-5 on synthetic AND real Raichur data |
| 2 | Isomap with SMACOF | **Done**: Stages 1-5 on synthetic AND real Raichur data |
| 3 | Kernel PCA | **Done**: Stages 1-5 on synthetic AND real Raichur data |
| 4 | Locally Linear Embedding (LLE) | **Done**: Stages 1-5 on synthetic AND real Raichur data (see 04_LLE/documentation/04_real_data_results.md) |
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

Each technique gets its own numbered top-level folder (code, docs, experiments,
results, visuals all together) - easier to navigate once there are six of
these than one giant shared `experiments/`/`results`/`visualisations/`. Only
code that is genuinely identical across every technique (preprocessing,
evaluation metrics, the real-data loader) stays shared, in `src/`.

```
Anugrah (Nonlinear Techniques)/
├── README.md                  <- this file, updated as we progress
├── requirements.txt           <- pinned dependencies for this subproject only
├── data/                      <- shared across all techniques (gitignored; see below)
│   └── synthetic_v1/          <- our synthetic 7-channel dataset
├── notebooks/                 <- exploratory notebooks (optional, not required)
├── src/                       <- SHARED code, used identically by every technique
│   ├── preprocessing.py       <- cube<->pixel flattening, standardising, validation
│   ├── evaluation.py          <- trustworthiness, continuity, reconstruction error, etc.
│   └── data_loader.py         <- real Raichur Parquet -> pixels/grid
├── experiments/                <- scripts that aren't specific to one technique
│   ├── 00_make_synthetic_dataset.py
│   ├── 01_pca_sanity_check.py
│   └── 04_test_data_loader_with_mock.py
├── documentation/               <- write-ups that aren't specific to one technique
│   ├── 01_concepts.md            <- non-linear DR intuition, PCA's limits, method comparison
│   └── 03_real_data_loader.md
│
├── 01_Isomap/                   <- Technique 1, DONE (Stages 1-5, synthetic + real)
│   ├── isomap.py                  <- the technique's own module (imports src/ as shared code)
│   ├── experiments/                <- this technique's runnable scripts
│   ├── results/                    <- this technique's saved metrics (JSON)
│   ├── visualisations/             <- this technique's saved figures (PNG)
│   └── documentation/              <- this technique's write-ups
│
├── 02_IsomapSMACOF/              <- Technique 2, same layout, not started yet
├── 03_KernelPCA/                 <- Technique 3, same layout, not started yet
├── 04_LLE/                       <- Technique 4, same layout, not started yet
├── 05_LaplacianEigenmaps/        <- Technique 5, same layout, not started yet
└── 06_FHNPW/                     <- Technique 6, same layout, not started yet
```

Each technique folder is self-contained and runnable from the project root,
e.g. `python 01_Isomap/experiments/03_isomap_on_synthetic_cube.py` - see that
folder's own scripts for the exact `sys.path` pattern used to reach the
shared `src/` modules.

`data/` is where generated/downloaded arrays live, shared by every technique.
Treat it like the rest of the repo's `data/` folders: do not commit large
files; keep only small synthetic data or a `.gitignore` entry for anything
derived from real Sentinel-2 downloads.


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

- `01_Isomap/documentation/02_isomap.md` - full write-up: algorithm steps, objective
  function, hyperparameters, advantages/limitations, citation.
- `01_Isomap/experiments/02_isomap_basic_example.py` - runs on the textbook Swiss roll
  (`sklearn.datasets.make_swiss_roll`), producing:
  - `01_Isomap/visualisations/02a_isomap_vs_pca_swissroll.png` - Isomap unrolls the roll
    cleanly (Spearman correlation with true position: 1.000) where PCA loops it
    into a spiral (0.221).
  - `01_Isomap/visualisations/02b_neighbor_graph.png` - the actual k-NN graph Isomap builds,
    shown hugging the roll's surface.
  - `01_Isomap/visualisations/02c_short_circuit_failure.png` - `n_neighbors` swept over
    6/10/30/80, concretely reproducing the "short-circuit" failure mode (the
    embedding curls back into a loop once neighbours start jumping across
    windings of the roll).

```bash
python 01_Isomap/experiments/02_isomap_basic_example.py
```

### Isomap: Stage 3 (module), Stage 4 (applied to synthetic cube), Stage 5 (evaluation) — done

- `src/preprocessing.py` - shared cube<->pixel flattening, validation, standardising;
  used by every technique module so preprocessing stays identical across the
  whole comparison (Section 5 requirement).
- `01_Isomap/isomap.py` - `fit_isomap_pixels` / `fit_isomap_cube`: a thin, validated,
  documented wrapper around `sklearn.manifold.Isomap` (exact reproduction, not a
  reimplementation - see module docstring), with timing/memory diagnostics and
  an explicit warning if the k-NN graph is disconnected.
- `src/evaluation.py` - shared metrics for all techniques: trustworthiness,
  continuity, neighbourhood preservation (Jaccard), reconstruction error (only
  when defined), explained-variance ratio (only when defined), timing, memory.
- `01_Isomap/experiments/03_isomap_on_synthetic_cube.py` - applies Isomap to
  `data/synthetic_v1/cube.npy` (7 -> 3 components, `n_neighbors=12`, seed=42) and:
  - **Proves, not just asserts, that pixel-wise Isomap is spatially blind:**
    `01_Isomap/visualisations/03c_spectral_vs_spatial_neighbors.png` shows a concrete
    example pixel whose 10 spectral nearest neighbours (what Isomap's graph
    actually uses) share **zero** overlap with its immediate 3x3 spatial
    neighbourhood.
  - Checks recovery of the known-ground-truth latents
    (`01_Isomap/visualisations/03b_latent_recovery.png`) and catches a real subtlety: one
    true latent is cyclic (an angle), so it only shows up correctly across two
    embedding dimensions *jointly* (`cos`/`sin` correlation 0.94), not in either
    dimension's raw Spearman correlation (0.25) - documented honestly in
    `01_Isomap/documentation/02_isomap_stage4_5.md` rather than reported as a failure.
  - Runs the full evaluation suite and saves `01_Isomap/results/isomap_synthetic_v1.json`.
- `01_Isomap/documentation/02_isomap_stage4_5.md` - full write-up of the above, plus
  computational-complexity/scalability discussion for a real satellite tile and
  a flagged research-gap note (spatially-weighted k-NN graph as a possible
  modified-Isomap extension, to be clearly labelled as a modification if built).

```bash
python 01_Isomap/experiments/03_isomap_on_synthetic_cube.py
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
- `01_Isomap/experiments/05_isomap_on_real_data.py` - ready to run once the data is in
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
python 01_Isomap/experiments/05_isomap_on_real_data.py           # the real run
```

### Isomap on real data — confirmed working, results in

Ran successfully against the real `raichur_features.parquet` (2026-09-27,
10,901,746 valid pixels loaded - matching the documented full-coverage count,
confirming the loader parses the real schema correctly, not just the mock).
Full comparison against the synthetic run, including a metric bug this run
helped surface and fix (`explained_variance_ratio` was a measurement
artifact, not a real 100% - see below), is in
`01_Isomap/documentation/04_isomap_real_data_results.md`.

**Bug found and fixed:** `src/evaluation.explained_variance_ratio` reported
1.0 on both the synthetic and real runs - identical values on very different
datasets was the tell. Root cause: scikit-learn's `Isomap` internally
truncates its eigenvalues to exactly `n_components` before exposing them, so
any "kept/total" ratio computed from it is 1.0 by construction, not a real
measurement. Fixed to return `None` (not computable) instead; both results
JSONs corrected with the original value preserved in
`01_Isomap/documentation/04_isomap_real_data_results.md` for transparency.

**Isomap is now fully complete** - the only technique, so far, validated on
both synthetic and real data end-to-end (Stages 1-5).

### Technique 2: Isomap with SMACOF — Stages 1-5 done on synthetic data

- `02_IsomapSMACOF/isomap_smacof.py` - combines two exact, established
  implementations: `sklearn.manifold.Isomap` for geodesic distances (Steps
  1-2, unchanged from Technique 1) and `sklearn.manifold.MDS` (SMACOF) for
  the embedding step. Handles a real version-compatibility issue directly: the
  installed scikit-learn (1.9.1) uses renamed `MDS` constructor arguments vs.
  this project's pinned 1.5.2 - `_build_mds()` detects which API is available
  via introspection rather than assuming one.
- `02_IsomapSMACOF/experiments/01_basic_example.py` (Stage 1-2) - an
  unplanned finding before the intended comparison: at a smaller sample size
  than Technique 1 used, the SAME `n_neighbors` that worked fine there already
  short-circuits here - confirming short-circuiting depends on sampling
  density relative to k, not k alone. Fixed by using a smaller `n_neighbors`
  for this sample size, then: (a) classical MDS vs. SMACOF embeddings compared
  directly on the same geodesic distances (0.999 vs 0.982 Spearman
  correlation with true position), (b) SMACOF's local-minima behaviour made
  concrete - 2 of 8 random restarts land at 4-5x worse stress than the other 6.
- `02_IsomapSMACOF/experiments/02_on_synthetic_cube.py` (Stage 4-5) - run with
  settings identical to Technique 1's synthetic run for direct comparability.
  Found and resolved an apparent discrepancy rather than reporting it as a
  weakness: one true latent looked poorly recovered in SMACOF's embedding
  compared to Technique 1's, but checking a linear combination of two
  embedding dimensions recovered it just as well (Spearman 0.95) - it was
  rotated across two axes instead of one, which is expected since any MDS
  solution is unique only up to rotation/reflection. Full metrics comparison
  in `02_IsomapSMACOF/documentation/02_stage4_5_synthetic.md`; headline
  finding: SMACOF gives a small, inconclusive edge on trustworthiness/
  neighbourhood-preservation at ~25x the runtime and ~2.5x the memory of
  classical-MDS Isomap, on this dataset.
- `02_IsomapSMACOF/experiments/03_on_real_data.py` - ready to run, same
  pattern as Technique 1's, but deliberately uses a SMALLER subsample (2,000
  vs. 8,000 pixels) given the ~25x per-pixel cost multiplier just measured -
  stated explicitly, not a silent shortcut.

```bash
python 02_IsomapSMACOF/experiments/01_basic_example.py
python 02_IsomapSMACOF/experiments/02_on_synthetic_cube.py
python 02_IsomapSMACOF/experiments/03_on_real_data.py
```

### Isomap+SMACOF on real data — confirmed, and a code bug this run caught

Ran successfully on real data (2,000-pixel subsample, per the cost note
above). Found and fixed a real bug in `isomap_smacof.py`'s `_build_mds()`:
it added the `init` constructor argument unconditionally, but this project's
actual PINNED scikit-learn (1.5.2) has no such argument at all - confirmed
directly via `inspect.signature` on the user's machine, which crashed with
`TypeError: MDS.__init__() got an unexpected keyword argument 'init'`. This
is distinct from the earlier `metric`/`dissimilarity` renaming check - fixed
by checking `"init" in params` the same defensive way as every other kwarg.

**The real finding:** SMACOF's small, "inconclusive" neighbourhood-
preservation edge over Isomap on synthetic data (+0.037) turned out to be
**much larger on real data** (+0.184: 0.611 vs 0.427), even using a quarter
as many pixels - a genuine, cross-validated result, not a toy-data fluke.
Caveat found and stated honestly: the real-data SMACOF run hit its
`max_iter=300` cap without converging (vs. 76 iterations to convergence on
synthetic data), so its 178.73s runtime may understate what full convergence
would need - full write-up in
`02_IsomapSMACOF/documentation/03_real_data_results.md`.

**Technique 2 is now fully complete** - Stages 1-5, synthetic and real data,
matching Technique 1's completion level.

### Technique 3: Kernel PCA — Stages 1-5 done on synthetic data

- `03_KernelPCA/kernel_pca.py` - exact reproduction via
  `sklearn.decomposition.KernelPCA`. Also provides a dedicated
  `explained_variance_ratio()` function: unlike Isomap (whose internal
  eigenvalues are always truncated to `n_components`, making that ratio
  uncomputable - see Technique 1's correction), Kernel PCA genuinely does
  expose its full eigenvalue spectrum if asked, so this metric is real here,
  verified directly rather than assumed.
- `03_KernelPCA/experiments/01_basic_example.py` (Stage 1-2) - tested Stage
  1's prediction directly rather than assuming it: RBF Kernel PCA cannot
  unroll the Swiss roll at ANY `gamma` tested (best correlation 0.42 vs.
  Isomap's 1.000) - a genuine negative result, because RBF similarity has no
  mechanism for "walking around" a curved surface the way geodesic distance
  does.
- `03_KernelPCA/experiments/02_on_synthetic_cube.py` (Stage 4-5) - an
  important contrast with Stage 2's negative result: on our actual
  satellite-like synthetic cube (not a pure Swiss roll), Kernel PCA performs
  close to Isomap (trustworthiness 0.992 vs 0.999) and runs **~5x faster
  with less memory** - the Swiss-roll weakness didn't carry over to this
  data. But a real, different limitation was found and verified: one of the
  two true latent variables is not recovered at all here (checked a linear
  combination of all 3 output dimensions - correlation 0.08, no better than
  chance), unlike Isomap/SMACOF where a similarly weak single-dimension
  correlation turned out to be a rotation artifact. Full writeup in
  `03_KernelPCA/documentation/03_stage4_5_synthetic.md`.
- `03_KernelPCA/experiments/03_on_real_data.py` - ready to run, using
  Technique 1's full 8,000-pixel sample size (no cost-driven reason to
  shrink it, unlike Technique 2).

```bash
python 03_KernelPCA/experiments/01_basic_example.py
python 03_KernelPCA/experiments/02_on_synthetic_cube.py
python 03_KernelPCA/experiments/03_on_real_data.py
```

### Kernel PCA on real data — confirmed, trade-off gets MORE pronounced at scale

Ran clean on the first attempt (no code bugs this time). The synthetic-data
trade-off (Kernel PCA: faster, less memory, worse neighbourhood
preservation than Isomap) holds up on real data and sharpens in an
interesting, asymmetric way:
- **Speed advantage grows**: 6.5x faster on synthetic data -> **8.1x faster**
  on real data (1.75s vs. Isomap's 14.13s, both at 8,000 pixels).
- **Memory advantage shrinks**: used 35% of Isomap's memory on synthetic
  data, but 67% on real data - both methods' dominant memory cost is an
  n x n matrix, so this gap is expected to keep narrowing, not widening, at
  full-tile pixel counts.
- **Neighbourhood-preservation gap stays stable** (~0.13 either way) -
  looking like a genuine, consistent property of the method, not a
  synthetic-data artifact.
- Flagged what real data structurally CANNOT tell us: Stage 4's finding that
  Kernel PCA fails to recover one true latent variable (checked against
  KNOWN ground truth) has no real-data equivalent check, since real data has
  no known ground truth to check against - stated as an open question, not
  guessed at. Full write-up in `03_KernelPCA/documentation/04_real_data_results.md`.

**Technique 3 is now fully complete** - Stages 1-5, synthetic and real data.

### Housekeeping: PARQUET_PATH now defaults correctly everywhere

All three real-data scripts' `PARQUET_PATH` default now includes
`DIP_project` (the confirmed real folder layout on this project) out of the
box - the manual edit needed twice before should not be needed again for
Technique 4 onward.

### Technique 4: LLE — Stages 1-5 done on synthetic data

- `04_LLE/lle.py` - exact reproduction via
  `sklearn.manifold.LocallyLinearEmbedding(method="standard")`.
- **Found and fixed a real bug in SHARED code**: `src/evaluation.py`'s
  `reconstruction_error()` only recognised Isomap's `.reconstruction_error()`
  METHOD, not LLE's `.reconstruction_error_` ATTRIBUTE (different shape,
  found by checking directly) - would have silently returned `None` for LLE,
  hiding a real number. Fixed to recognise both, verified Isomap's existing
  results are unaffected.
- `04_LLE/experiments/01_basic_example.py` (Stage 1-2) - tested Stage 1's
  specific claim that LLE's lack of a chained-distance graph means it fails
  differently from Isomap, not just "better" or "worse": at high
  `n_neighbors`, LLE's embedding does NOT curl into Isomap's signature closed
  loop - instead it pinches toward a corner (k=30) then flattens into
  interleaved stripes (k=80). A genuinely different failure mechanism, as
  predicted, not immunity to large k.
- `04_LLE/experiments/02_on_synthetic_cube.py` (Stage 4-5) - the weakest
  result of any technique so far on strict metrics (trustworthiness 0.976,
  neighbourhood preservation 0.259 - worse than Kernel PCA's 0.464), despite
  LLE's OWN reconstruction error being tiny (1.14e-07). This disconnect is
  itself the finding: LLE only optimises a LOCAL criterion with no mechanism
  enforcing sensible GLOBAL layout, so it can satisfy its own objective
  nearly perfectly while still being noisier globally than every other
  technique tested. Full write-up, including a latent-recovery result
  distinct from both prior patterns (visible ring shape for the cyclic
  latent, but weak joint-linear recovery, unlike Isomap/SMACOF's clean
  rescue or Kernel PCA's total miss), in
  `04_LLE/documentation/03_stage4_5_synthetic.md`.
- `04_LLE/experiments/03_on_real_data.py` - ready to run, using the full
  8,000-pixel sample (LLE is the lightest-memory technique so far, no
  cost-driven reason to shrink it).

```bash
python 04_LLE/experiments/01_basic_example.py
python 04_LLE/experiments/02_on_synthetic_cube.py
python 04_LLE/experiments/03_on_real_data.py   # once the data is in place
```

## Additional benchmark scenes (hyperspectral, with ground truth)

Techniques 1-4 (plus a PCA baseline) were also run on public hyperspectral
benchmarks, with identical settings (k=12, 3 components, seed 42, up to 8,000
pixels): Indian Pines, Salinas, Kennedy Space Center (KSC), and Botswana
(no labels supplied, neighbourhood metrics only). Pavia University could NOT be
run: only its ground-truth file was supplied, not the image cube (`PaviaU.mat`).
Isomap with SMACOF is OFF by default for these scenes (too slow in the
development sandbox); run with `--with-smacof` locally.

```powershell
# put the .mat files in data/hsi/ (or pass --data-dir), then:
python experiments/07_benchmarks_all_techniques.py            # add --with-smacof for SMACOF
python experiments/08_benchmark_summary.py
```
Outputs: `results/benchmarks/<scene>_all.json`, `results/benchmarks_summary.md/.csv`,
`visualisations/07_<scene>_embeddings.png`. Full write-up:
`documentation/04_hsi_benchmarks.md`. New shared code: `src/hsi_loader.py`,
`src/classification.py`. (`experiments/05_` and `06_` are the earlier Indian-Pines-only
script and the Raichur-vs-Indian-Pines table; `07_`/`08_` supersede them for the
multi-scene comparison.) The large .mat files are not included in the zip.

## Next step

Technique 5: **Constrained Laplacian Eigenmaps** (a modification of the base
algorithm; Stage 4 must re-run the spectral-vs-spatial neighbour check), then
Technique 6 (FHNPW, only after its original publication is verified).
