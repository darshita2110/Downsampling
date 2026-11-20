# 3. Real-data loader (`src/data_loader.py`) and Isomap on real data

## Why this exists now, not later

The project brief requires the final Section-5 comparison to run on real
Raichur Sentinel-2 data, not synthetic data (see README "Commitment: real
data"). Before moving to Technique 2, we built and validated the loader so
Isomap - and every later technique - can be run on real data as soon as it's
implemented, rather than leaving real-data integration as a rushed step at
the end.

## What we could and couldn't verify here

**The real Parquet file (`raichur_features.parquet`, 808 MB) is not available
in this development environment.** It is gitignored in the root repo and
distributed separately (see `docs/DATA_ACCESS_GUIDE.md`); the shared-folder
link in that guide is not filled in yet either. So `src/data_loader.py` could
not be tested against the real file here - it was written strictly from the
documented schema (columns, dtypes, coverage behaviour, the LST/empty-column
note) in `docs/DATA_ACCESS_GUIDE.md`, and verified against a small **mock**
Parquet file built to match that exact schema
(`experiments/04_test_data_loader_with_mock.py`). All 6 schema-level checks
pass (full-coverage load, partial-coverage grid sizing, zero-row date,
all-null-LST handling, missing-file handling). This confirms the logic is
correct against the documented schema - **confirming it against the real file
is still an open step**, to be done on a machine that has the data.

## A real bug the mock test caught before real data ever touched it

The first version of the loader derived the pixel grid's shape from each
date's *own* x/y extent. The mock test's own partial-coverage case ("northern
part only", matching the documented 2026-10-02/2026-10-07 behaviour) caught
this immediately: a half-covered date produced a grid half the true size,
silently - not an error, just a wrong, undersized grid. Fixed by adding
`get_grid_reference()`, which reads one known full-coverage date's coordinate
axes (2026-09-27 is documented at 100% coverage) and lets every other date
reuse those same axes, so row/col positions - and therefore where NaNs end up
in `reconstruct_grid()` - are consistent across dates. This is exactly the
value of testing loader logic against a schema-accurate mock before real data
is in hand: this bug would otherwise have surfaced as a subtly wrong figure
sometime after Section 5 was already underway.

## Design decisions, stated explicitly

- **No dense `(H, W, 7)` cube for real data.** Real tiles have irregular,
  partial coverage (0% to 100% depending on cloud/shadow per date). The
  loader returns valid pixels as a flat array plus `(row, col)` positions,
  and `reconstruct_grid()` scatters any per-pixel array (raw channels, or
  later an embedding) back into a grid for visualisation, filling gaps with
  NaN. Dimensionality-reduction code (`src/isomap.fit_isomap_pixels`) already
  worked on flat pixel arrays and needed no changes for this.
- **LST handling.** `lst_mode="drop"` (default): real-data runs use 6
  channels, not the assigned 7, because LST is documented as entirely empty
  upstream. `lst_mode="require"` exists so that once LST IS integrated
  upstream, forgetting to update this loader produces a loud error instead of
  silently continuing to drop a now-available channel.
- **Reflectance scaling.** Raw bands are divided by 10000 by default
  (`reflectance_scale=True`), per the documented convention, so they are on a
  comparable numeric scale to NDVI/NDWI before standardisation.
- **Clouds are not masked.** The real dataset has no cloud mask; cloud/shadow
  pixels can look like bare land (NDVI) or water (NDWI), per
  `docs/DATA_ACCESS_GUIDE.md`. This loader does not add masking - a known,
  stated limitation of this first real-data pass, carried through into
  `01_Isomap/experiments/05_isomap_on_real_data.py`'s results file under
  `known_limitations`.

## Isomap on real data (`01_Isomap/experiments/05_isomap_on_real_data.py`)

Loads `2026-09-27` ("mostly clear" per the docs), **randomly subsamples 8,000
of the available valid pixels** (a full date can have up to ~10.9 million
pixels - see `01_Isomap/documentation/02_isomap_stage4_5.md`'s complexity note for why
exact Isomap cannot run on that directly), runs the same `fit_isomap_pixels`
used throughout this project, and reconstructs a false-colour composite back
into the real (gappy) tile grid.

**Status: written and schema-tested, not yet run against real data** (same
constraint as the loader itself - no file available here). When you run this
on a machine with the data in place, it will produce
`01_Isomap/visualisations/05a_isomap_real_data_composite.png` and
`01_Isomap/results/isomap_real_raichur.json`. Verified in this environment only that it
fails gracefully and informatively when the data file is absent (confirmed:
prints the expected path and a pointer to `docs/DATA_ACCESS_GUIDE.md`, then
exits cleanly rather than crashing).

## What still needs to happen on real data (flagged, not silently skipped)

- Random subsampling is a real limitation, stated in the results file itself
  (`known_limitations`) - a block/stratified spatial subsample, or an
  approximate/landmark-Isomap variant, would be a better long-term answer and
  is noted as a research gap in `01_Isomap/documentation/02_isomap_stage4_5.md`.
- No cloud masking - results on 2026-09-27 ("mostly clear, some small
  clouds") will include some cloud-contaminated pixels.
- Only 6 of the assigned 7 channels are usable until LST is integrated
  upstream - this affects every technique equally, not just Isomap, so it
  should be noted once in the final paper rather than repeated per technique.
