"""
Verifies src/data_loader.py's LOGIC against a small mock Parquet file built to
match the documented real-data schema exactly (docs/DATA_ACCESS_GUIDE.md):
columns x, y, latitude, longitude, date, B02, B03, B04, B08, NDVI, NDWI, LST;
LST entirely null; partial (irregular) spatial coverage; one date with zero
rows. This is NOT a test against the real file (not available in this
environment - see src/data_loader.py docstring) - it only confirms the loader
behaves correctly against the schema as documented, so that running it against
the real file later is a matter of pointing at the right path, not debugging
loader logic blind.

Run:
    python experiments/04_test_data_loader_with_mock.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import get_grid_reference, load_raichur_tile, reconstruct_grid

MOCK_PATH = ROOT / "data" / "mock_raichur_schema.parquet"


def make_mock_parquet(rng):
    """Build a tiny mock matching the real schema: a 12x10 grid, two dates
    (one with full coverage, one with partial/irregular coverage - e.g. clouds
    covering the northern half), LST entirely null (as documented), raw bands
    on the real reflectance x10000 scale, NDVI/NDWI already in [-1, 1].
    """
    n_rows, n_cols = 12, 10
    x0, y0, spacing = 735000, 1815000, 10  # arbitrary but realistic UTM-looking origin
    xs = x0 + np.arange(n_cols) * spacing
    ys = y0 - np.arange(n_rows) * spacing  # y decreases going south/down, as in real UTM north-up grids

    rows_out = []
    for date, coverage in [("2026-09-27", "full"), ("2026-10-02", "partial"), ("2026-09-29", "none")]:
        for r in range(n_rows):
            for c in range(n_cols):
                if coverage == "none":
                    continue
                if coverage == "partial" and r < n_rows // 2:
                    continue  # simulate the documented "northern part only" case
                b02 = rng.integers(200, 2500)
                b03 = rng.integers(200, 2500)
                b04 = rng.integers(200, 2500)
                b08 = rng.integers(200, 4000)
                ndvi = (b08 - b04) / (b08 + b04)
                ndwi = (b03 - b08) / (b03 + b08)
                rows_out.append({
                    "x": xs[c], "y": ys[r],
                    "latitude": 16.1 + r * 1e-4, "longitude": 77.2 + c * 1e-4,
                    "date": pd.Timestamp(date),
                    "B02": b02, "B03": b03, "B04": b04, "B08": b08,
                    "NDVI": ndvi, "NDWI": ndwi, "LST": np.nan,
                })
    df = pd.DataFrame(rows_out)
    df.to_parquet(MOCK_PATH)
    return df


def main():
    rng = np.random.default_rng(42)
    MOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = make_mock_parquet(rng)
    print(f"Built mock Parquet: {len(df)} rows, dates={sorted(df['date'].dt.date.unique())}")

    # --- Test 1: full-coverage date loads correctly, LST dropped by default
    result = load_raichur_tile(MOCK_PATH, date="2026-09-27", lst_mode="drop")
    assert result["pixels"].shape == (120, 6), result["pixels"].shape  # 12x10 grid, 6 channels (LST dropped)
    assert result["channel_names"] == ["B02", "B03", "B04", "B08", "NDVI", "NDWI"]
    assert result["grid_shape"] == (12, 10)
    assert result["lst_available"] is False
    # reflectance scaling: raw B02 was in [200, 2500); scaled should be in [0.02, 0.25)
    b02_col = result["channel_names"].index("B02")
    assert result["pixels"][:, b02_col].max() < 1.0, "reflectance_scale did not divide by 10000"
    print("Test 1 passed: full-coverage date, LST dropped, reflectance scaled correctly.")

    # --- Test 2: partial coverage, WITHOUT a grid_reference, under-sizes the
    # grid to just its own extent - demonstrating why grid_reference exists.
    result_partial_naive = load_raichur_tile(MOCK_PATH, date="2026-10-02", lst_mode="drop")
    assert result_partial_naive["pixels"].shape[0] == 60, result_partial_naive["pixels"].shape
    assert result_partial_naive["grid_shape"] == (6, 10), (
        f"expected an UNDER-sized (6, 10) grid without grid_reference, got "
        f"{result_partial_naive['grid_shape']} - this is the exact failure mode "
        f"grid_reference is designed to prevent."
    )
    print("Test 2 passed: confirmed partial-coverage dates under-size their own grid "
          "when loaded without grid_reference (this is why grid_reference exists).")

    # --- Test 3: partial coverage WITH a grid_reference from the full-coverage
    # date reconstructs into the correct full-sized grid, NaN exactly in the gap
    reference = get_grid_reference(MOCK_PATH, reference_date="2026-09-27")
    result_partial = load_raichur_tile(MOCK_PATH, date="2026-10-02", lst_mode="drop",
                                        grid_reference=reference)
    assert result_partial["grid_shape"] == (12, 10), result_partial["grid_shape"]
    grid = reconstruct_grid(result_partial["pixels"][:, 0], result_partial["row"],
                             result_partial["col"], result_partial["grid_shape"])
    assert grid.shape == (12, 10)
    assert np.isnan(grid[:6, :]).all(), "expected the uncovered northern half to be NaN"
    assert not np.isnan(grid[6:, :]).any(), "expected the covered southern half to have no NaN"
    print("Test 3 passed: with grid_reference, partial coverage reconstructs at full size, "
          "NaN exactly where pixels are missing.")

    # --- Test 4: a date with zero rows raises a clear error, not a silent empty result
    try:
        load_raichur_tile(MOCK_PATH, date="2026-09-29", lst_mode="drop")
        raise AssertionError("expected a ValueError for the zero-row date")
    except ValueError as e:
        assert "zero valid pixels" in str(e) or "No rows found" in str(e)
        print(f"Test 4 passed: zero-row date raises clearly: {e}")

    # --- Test 5: lst_mode='require' raises, since LST is genuinely all-null here
    try:
        load_raichur_tile(MOCK_PATH, date="2026-09-27", lst_mode="require")
        raise AssertionError("expected a ValueError since LST is all-null in the mock")
    except ValueError as e:
        print(f"Test 5 passed: lst_mode='require' correctly refuses all-null LST: {e}")

    # --- Test 6: a missing file raises a clear, actionable error
    try:
        load_raichur_tile(ROOT / "data" / "does_not_exist.parquet", date="2026-09-27")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        print(f"Test 6 passed: missing file raises clearly: {e}")

    print("\nAll schema-level tests passed. This confirms the LOGIC is correct against "
          "the documented schema - running against the real Parquet file still needs to "
          "be done on a machine that has it (see docs/DATA_ACCESS_GUIDE.md).")


if __name__ == "__main__":
    main()
