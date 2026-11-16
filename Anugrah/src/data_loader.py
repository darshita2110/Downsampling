"""
Loader for the real Raichur Sentinel-2 dataset produced by the repo root's
`scripts/main.py` pipeline (see docs/DATA_ACCESS_GUIDE.md). NOT tested against
the real file in this environment - the Parquet file (808 MB) is gitignored and
downloaded separately; the shared-folder link in DATA_ACCESS_GUIDE.md is not
filled in yet, so this environment has no access to it. Instead, this module is
verified here against a small synthetic mock built to the *documented* schema
(see tests/test_data_loader_schema.py) - correctness against the real file
still needs to be confirmed once you run it locally.

Design note: unlike our synthetic data/synthetic_v1/cube.npy (a clean, fully
dense (H, W, 7) rectangle), real tiles have PARTIAL, IRREGULAR coverage - only
pixels where all 4 raw bands are valid are included, and coverage ranges from
0% to 100% depending on cloud/shadow on that date (see DATA_ACCESS_GUIDE.md
section "Dates"). Forcing that into a dense cube would mean inventing values
for missing pixels. Instead:
  - `load_raichur_tile()` returns the VALID pixels only, as a flat
    (n_valid_pixels, n_channels) array, plus each pixel's (row, col) position
    in the full grid and the full grid's shape.
  - `reconstruct_grid()` scatters any per-pixel array (the raw channels, or
    later a technique's embedding) back into a (n_rows, n_cols[, k]) array for
    visualisation, filling everywhere-else with NaN - this is for PLOTTING
    only; dimensionality-reduction code should keep operating on the flat
    valid-pixels array via src.isomap.fit_isomap_pixels (already generic, no
    changes needed there).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_BAND_NAMES = ["B02", "B03", "B04", "B08"]
DERIVED_NAMES = ["NDVI", "NDWI"]
RESERVED_NAMES = ["LST"]  # documented as "Empty, not yet integrated" upstream
ALL_CHANNEL_NAMES = RAW_BAND_NAMES + DERIVED_NAMES + RESERVED_NAMES  # matches our assigned 7 channels


def get_grid_reference(parquet_path, reference_date):
    """Read just x, y for one date and return its sorted unique coordinate axes,
    to be reused (via `grid_reference=`) when loading OTHER dates.

    Why this exists: docs/DATA_ACCESS_GUIDE.md states "the grid is the same on
    every date" (a fixed 3247x3358, 10 m study-area grid) - x, y identify the
    same physical pixel across all dates. But most dates have PARTIAL coverage
    (see the 'Dates' table), so deriving row/col axes from a partial date's own
    x, y extent under-covers the true grid (we found this directly: it failed
    experiments/04_test_data_loader_with_mock.py's own consistency check before
    this function existed). Pass a known FULL-coverage date here (per
    DATA_ACCESS_GUIDE.md, 2026-09-27 is documented at 100% coverage) and reuse
    its axes for every other date, so row/col positions - and hence NaN
    locations in `reconstruct_grid` - are consistent and comparable across dates.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found - see docs/DATA_ACCESS_GUIDE.md.")
    df = pd.read_parquet(parquet_path, columns=["x", "y"],
                          filters=[("date", "==", pd.Timestamp(reference_date))])
    if len(df) == 0:
        raise ValueError(f"No rows for reference_date={reference_date}; pick a date with coverage.")
    return {
        "unique_x": np.sort(df["x"].unique()),
        "unique_y": np.sort(df["y"].unique())[::-1],
        "reference_date": reference_date,
    }


def load_raichur_tile(parquet_path, date, lst_mode="drop", reflectance_scale=True, grid_reference=None):
    """Load one date's valid pixels from the Raichur Parquet dataset.

    Parameters
    ----------
    parquet_path : str or Path
        Path to raichur_features.parquet (see docs/DATA_ACCESS_GUIDE.md).
    date : str or datetime.date
        One of the dataset's acquisition dates, e.g. "2026-09-27". The dataset
        documents one date (2026-09-29) with ZERO valid pixels - this function
        raises a clear error for that case rather than returning an empty,
        silently-broken result.
    lst_mode : {"drop", "require", "keep_nan"}
        How to handle LST, which is documented as entirely empty upstream:
          "drop"     - (default) exclude LST, return the other 6 channels.
                       This is a real, documented limitation of the current
                       pipeline, not a choice we are hiding.
          "require"  - raise an error (use once LST is actually integrated
                       upstream, as a safety check that it's no longer empty).
          "keep_nan" - include an all-NaN LST column anyway (only useful if a
                       later step will impute it; dimensionality-reduction
                       code elsewhere in this project rejects NaNs on purpose,
                       see src/preprocessing.validate_cube, so this mode alone
                       will not make the data usable downstream).
    grid_reference : dict or None
        Output of `get_grid_reference()` on a known FULL-coverage date. If
        given, row/col positions are computed against THAT date's coordinate
        axes rather than this date's own (necessarily partial-coverage-sized)
        extent - use this whenever comparing/visualising more than one date so
        grid positions and NaN locations line up. If None (default), axes are
        derived from this call's own data only, which under-sizes the grid for
        any date that is not full-coverage (documented explicitly here rather
        than silently producing an undersized grid with no warning).
    reflectance_scale : bool
        If True (default), divide B02/B03/B04/B08 by 10000 to convert to
        reflectance (0-1-ish range), per DATA_ACCESS_GUIDE.md section 3. This
        matters for manifold-learning methods: without it, the raw bands
        (~0-3000+) would dominate Euclidean distance over NDVI/NDWI (~-1 to 1)
        purely due to scale, before src.preprocessing.standardize even runs.

    Returns
    -------
    dict with:
        pixels         : (n_valid, n_channels) float64 array
        channel_names  : list[str], same order as pixels' columns
        row, col       : (n_valid,) int arrays - this pixel's position in the
                         full grid (0-indexed, row 0 = northernmost / largest y)
        grid_shape     : (n_rows, n_cols) - the full grid's shape, as derived
                         from this date's own x, y extent (not hardcoded)
        x, y           : (n_valid,) the original UTM coordinates (kept for
                         joining back to latitude/longitude if needed later)
        date           : the date requested
        lst_available  : bool - whether LST actually had non-null values
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. This file is not stored in the "
            f"repository (see docs/DATA_ACCESS_GUIDE.md) - download it "
            f"separately and place it at data/processed/ML_Dataset/combined/ "
            f"relative to the repo root, or pass the correct path."
        )
    if lst_mode not in ("drop", "require", "keep_nan"):
        raise ValueError(f"lst_mode must be 'drop', 'require' or 'keep_nan', got {lst_mode!r}")

    columns = ["x", "y"] + RAW_BAND_NAMES + DERIVED_NAMES + RESERVED_NAMES
    df = pd.read_parquet(parquet_path, columns=columns, filters=[("date", "==", pd.Timestamp(date))])
    if len(df) == 0:
        raise ValueError(
            f"No rows found for date={date}. Either the date is wrong, or - as "
            f"documented for 2026-09-29 - this date genuinely has zero valid "
            f"pixels for every band (see docs/DATA_ACCESS_GUIDE.md, 'Dates' table)."
        )

    lst_available = df["LST"].notna().any()
    if lst_mode == "require" and not lst_available:
        raise ValueError(
            "lst_mode='require' but LST is entirely null for this date - LST "
            "is documented as 'not yet integrated' upstream (see "
            "docs/DATA_ACCESS_GUIDE.md section 5, 'Future work'). Use "
            "lst_mode='drop' (6 channels) until it is, or supply LST from "
            "elsewhere and merge it in before calling this function."
        )
    channel_names = list(RAW_BAND_NAMES + DERIVED_NAMES)
    if lst_mode == "keep_nan":
        channel_names.append("LST")

    # Pixel grid axes: sorted unique x -> columns (west to east), sorted unique
    # y DESCENDING -> rows (north to south, standard image row order). If a
    # grid_reference is given (from a known full-coverage date), reuse ITS
    # axes so row/col positions are consistent across dates; otherwise derive
    # axes from this date's own data, which under-sizes the grid for any
    # partial-coverage date (see the grid_reference docstring above).
    if grid_reference is not None:
        unique_x = grid_reference["unique_x"]
        unique_y = grid_reference["unique_y"]
    else:
        unique_x = np.sort(df["x"].unique())
        unique_y = np.sort(df["y"].unique())[::-1]
    col = np.searchsorted(unique_x, df["x"].to_numpy())
    row = np.searchsorted(unique_y[::-1], df["y"].to_numpy())
    row = len(unique_y) - 1 - row  # correct for the reversal above
    grid_shape = (len(unique_y), len(unique_x))

    pixels = df[channel_names].to_numpy(dtype="float64")
    if reflectance_scale:
        for i, name in enumerate(channel_names):
            if name in RAW_BAND_NAMES:
                pixels[:, i] = pixels[:, i] / 10000.0

    return {
        "pixels": pixels,
        "channel_names": channel_names,
        "row": row,
        "col": col,
        "grid_shape": grid_shape,
        "x": df["x"].to_numpy(),
        "y": df["y"].to_numpy(),
        "date": date,
        "lst_available": bool(lst_available),
    }


def reconstruct_grid(values, row, col, grid_shape, fill_value=np.nan):
    """Scatter a (n_valid,) or (n_valid, k) array of per-pixel values back into
    a full (n_rows, n_cols) or (n_rows, n_cols, k) grid, for visualisation.
    Positions not covered by any valid pixel are filled with `fill_value`
    (default NaN) - this is the honest representation of real, partial
    coverage; do not interpolate or fill with zeros without saying so.
    """
    values = np.asarray(values)
    n_rows, n_cols = grid_shape
    if values.ndim == 1:
        grid = np.full((n_rows, n_cols), fill_value, dtype="float64")
        grid[row, col] = values
    else:
        k = values.shape[1]
        grid = np.full((n_rows, n_cols, k), fill_value, dtype="float64")
        grid[row, col, :] = values
    return grid
