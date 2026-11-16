"""Raichur Sentinel-2: RGB, false colour, NDVI and NDWI images plus one combined ML dataset.

Run from the project folder:  python scripts/main.py

Each date is loaded on its own and rows are written in blocks, so memory use stays low.
Re-running overwrites the previous outputs. The raw NetCDF files are only ever read.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import xarray as xr
from PIL import Image
from pyproj import Transformer

# ------------------------------------------------------------------- settings
DATES = ["2026-09-17", "2026-09-27", "2026-09-29", "2026-10-02", "2026-10-04", "2026-10-07"]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sentinel2_l1c"
IMAGE_DIR = ROOT / "data" / "processed" / "sentinel2"
DATASET_DIR = ROOT / "data" / "processed" / "ML_Dataset" / "combined"
PARQUET_PATH = DATASET_DIR / "raichur_features.parquet"
CSV_PATH = DATASET_DIR / "raichur_features.csv"
STATS_DIR = ROOT / "reports" / "dataset_statistics"

# The full table is too large for a useful CSV, so the CSV keeps the same randomly chosen pixel
# locations on every date. Set this to None to export every row instead.
CSV_SAMPLE_PIXELS = 100_000
RANDOM_SEED = 42
BLOCK_ROWS = 256

BANDS = ["B02", "B03", "B04", "B08"]
EMPTY_FEATURES = ["LST", "population_density", "infrastructure_density"]  # not yet integrated

# Only used to set image contrast, so that bright clouds do not darken the land.
# No pixel is removed: the files have no cloud band, and clouds remain in images and dataset.
DISPLAY_BRIGHT_B02 = 2500

SCHEMA = pa.schema(
    [("x", pa.int32()), ("y", pa.int32()), ("latitude", pa.float64()), ("longitude", pa.float64()),
     ("date", pa.date32())]
    + [(b, pa.int16()) for b in BANDS]
    + [("NDVI", pa.float32()), ("NDWI", pa.float32())]
    + [(name, pa.float32()) for name in EMPTY_FEATURES],
    metadata={"values": "B02-B08 are L1C top-of-atmosphere reflectance x 10000; x/y are EPSG:32643 metres"},
)


# --------------------------------------------------------------------- reading
def load_scene(date):
    """Read one date as int16 arrays. The raw values are used directly, with no scaling applied."""
    path = RAW_DIR / f"raichur_s2_{date}.nc"
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        file_date = str(ds["t"].values)[:10]
        if file_date != date:
            raise ValueError(f"{path.name} contains data for {file_date}")
        bands = {b: ds[b].values for b in BANDS}
        fill = {b: ds[b].attrs["_FillValue"] for b in BANDS}
        scene = {"bands": bands, "x": ds["x"].values.astype("i4"), "y": ds["y"].values.astype("i4"),
                 "crs": ds["crs"].attrs["crs_wkt"]}
    # A pixel is usable only when every band has a real, positive value.
    scene["valid"] = np.logical_and.reduce([(bands[b] != fill[b]) & (bands[b] > 0) for b in BANDS])
    return scene


def normalized_difference(a, b, valid):
    """(a - b) / (a + b); NaN wherever a pixel is not usable. a + b > 0 on all valid pixels."""
    out = np.full(a.shape, np.nan, "f4")
    a, b = a[valid].astype("f4"), b[valid].astype("f4")
    out[valid] = (a - b) / (a + b)
    return out


# ---------------------------------------------------------------------- images
def save_figure(image, title, path, colorbar=None):
    fig, ax = plt.subplots(figsize=(12.5, 12.5))
    if colorbar:
        shown = ax.imshow(image, cmap=colorbar["cmap"], vmin=-1, vmax=1, interpolation="nearest")
        fig.colorbar(shown, ax=ax, fraction=0.04, pad=0.02, label=colorbar["label"])
    else:
        ax.imshow(image, interpolation="nearest")
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def composite(scene, channels):
    """Stack three bands with a 2-98 percentile stretch; no-data pixels become transparent."""
    valid = scene["valid"]
    not_bright = valid & (scene["bands"]["B02"] < DISPLAY_BRIGHT_B02)
    use = not_bright if not_bright.any() else valid
    layers = []
    for band in channels:
        values = scene["bands"][band]
        low, high = np.percentile(values[use], [2, 98])
        layers.append((np.clip((values.astype("f4") - low) / (high - low), 0, 1) * 255).astype("u1"))
    return np.dstack(layers + [(valid * 255).astype("u1")])


def make_images(scene, ndvi, ndwi, date):
    outputs = {
        "RGB": IMAGE_DIR / "RGB" / f"raichur_rgb_{date}.png",
        "False_Color": IMAGE_DIR / "False_Color" / f"raichur_false_color_{date}.png",
        "NDVI": IMAGE_DIR / "NDVI" / f"raichur_ndvi_{date}.png",
        "NDWI": IMAGE_DIR / "NDWI" / f"raichur_ndwi_{date}.png",
    }
    save_figure(composite(scene, ["B04", "B03", "B02"]), f"Raichur True Colour (B04, B03, B02), {date}", outputs["RGB"])
    save_figure(composite(scene, ["B08", "B04", "B03"]), f"Raichur False Colour (B08, B04, B03), {date}", outputs["False_Color"])
    save_figure(ndvi, f"Raichur NDVI, {date}", outputs["NDVI"], {"cmap": "RdYlGn", "label": "NDVI = (B08 - B04) / (B08 + B04)"})
    save_figure(ndwi, f"Raichur NDWI (McFeeters), {date}", outputs["NDWI"], {"cmap": "RdBu", "label": "NDWI = (B03 - B08) / (B03 + B08)"})
    return list(outputs.values())


# --------------------------------------------------------------------- dataset
def write_rows(scene, ndvi, ndwi, date, parquet_writer, csv_writer, chosen):
    """Write one date to the Parquet file (all rows) and CSV (chosen pixels), block by block."""
    transformer = Transformer.from_crs(scene["crs"], "EPSG:4326", always_xy=True)
    day_number = int(np.datetime64(date, "D").astype("i8"))
    height = scene["valid"].shape[0]
    rows_written = csv_rows = 0
    for start in range(0, height, BLOCK_ROWS):
        block = slice(start, start + BLOCK_ROWS)
        keep = scene["valid"][block]
        if not keep.any():
            continue
        row, col = np.nonzero(keep)
        x, y = scene["x"][col], scene["y"][row + start]
        lon, lat = transformer.transform(x.astype("f8"), y.astype("f8"))  # UTM metres to degrees
        columns = {"x": pa.array(x), "y": pa.array(y), "latitude": pa.array(lat), "longitude": pa.array(lon),
                   "date": pa.array(np.full(x.size, day_number, "i4")).cast(pa.date32()),
                   "NDVI": pa.array(ndvi[block][keep]), "NDWI": pa.array(ndwi[block][keep])}
        for band in BANDS:
            columns[band] = pa.array(scene["bands"][band][block][keep])
        for name in EMPTY_FEATURES:
            columns[name] = pa.nulls(x.size, pa.float32())  # empty until a real dataset is added
        table = pa.Table.from_arrays([columns[f.name] for f in SCHEMA], schema=SCHEMA)
        parquet_writer.write_table(table)
        rows_written += table.num_rows
        selected = table if chosen is None else table.filter(pa.array(chosen[row + start, col]))
        if selected.num_rows:
            csv_writer.write_table(selected.replace_schema_metadata(None))
            csv_rows += selected.num_rows
    return rows_written, csv_rows


def feature_statistics():
    """Reopen the finished Parquet file and calculate statistics per date and feature."""
    totals = {}
    parquet = pq.ParquetFile(PARQUET_PATH)
    names = [f.name for f in SCHEMA if f.name not in ("x", "y", "date")]
    for i in range(parquet.num_row_groups):
        table = parquet.read_row_group(i)
        date = str(table["date"][0].as_py())
        for name in names:
            column = table[name]
            entry = totals.setdefault((date, name), {"rows": 0, "missing": 0, "min": None, "max": None, "sum": 0.0})
            entry["rows"] += len(column)
            entry["missing"] += column.null_count
            if column.null_count < len(column):
                limits = pc.min_max(column).as_py()
                entry["min"] = limits["min"] if entry["min"] is None else min(entry["min"], limits["min"])
                entry["max"] = limits["max"] if entry["max"] is None else max(entry["max"], limits["max"])
                entry["sum"] += pc.sum(column.cast(pa.float64())).as_py()
    records = []
    for (date, name), e in sorted(totals.items()):
        present = e["rows"] - e["missing"]
        records.append({"date": date, "feature": name, "rows": e["rows"], "missing_pct": round(100 * e["missing"] / e["rows"], 3),
                        "min": e["min"], "max": e["max"], "mean": e["sum"] / present if present else None})
    return pd.DataFrame(records)


# ---------------------------------------------------------------- verification
def verify(summary, stats, image_paths):
    """Check the saved files by reopening them. Returns a list of problems."""
    problems = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if np.asarray(image.convert("L").resize((200, 200))).std() < 1:
                    problems.append(f"{path.name} looks blank")
        except Exception as exc:
            problems.append(f"{path.name}: {exc}")

    expected = summary.set_index("date")["rows_in_dataset"]
    expected = expected[expected > 0].to_dict()
    found = stats[stats["feature"] == "NDVI"].set_index("date")["rows"].to_dict()
    if found != expected:
        problems.append(f"rows per date differ: expected {expected}, found {found}")
    for index in ("NDVI", "NDWI"):
        part = stats[stats["feature"] == index]
        if part["min"].min() < -1 or part["max"].max() > 1:
            problems.append(f"{index} is outside the range -1 to +1")
    for name, low, high in (("latitude", 16.0, 16.5), ("longitude", 77.1, 77.6)):
        part = stats[stats["feature"] == name]
        if part["min"].min() < low or part["max"].max() > high:
            problems.append(f"{name} falls outside the Raichur study area")
    if stats[stats["feature"].isin(EMPTY_FEATURES)]["missing_pct"].ne(100).any():
        problems.append("a not-yet-integrated feature contains values")

    csv_columns = pd.read_csv(CSV_PATH, nrows=5).columns.tolist()
    if csv_columns != [f.name for f in SCHEMA]:
        problems.append(f"unexpected CSV columns: {csv_columns}")
    return problems


# ------------------------------------------------------------------------- run
def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_tmp = PARQUET_PATH.with_name(PARQUET_PATH.name + ".tmp")
    csv_tmp = CSV_PATH.with_name(CSV_PATH.name + ".tmp")
    records, image_paths, failed, chosen = [], [], [], None
    with pq.ParquetWriter(parquet_tmp, SCHEMA, compression="zstd", use_dictionary=["x", "y", "date"],
                          use_byte_stream_split=["latitude", "longitude", "NDVI", "NDWI"]) as parquet_writer, \
            open(csv_tmp, "wb") as csv_sink, pacsv.CSVWriter(csv_sink, SCHEMA.remove_metadata()) as csv_writer:
        for number, date in enumerate(DATES, 1):
            print(f"[{number}/{len(DATES)}] {date}", flush=True)
            record = {"date": date, "status": "", "total_pixels": 0, "rows_in_dataset": 0, "rows_in_csv": 0, "pixels_excluded": 0}
            records.append(record)
            try:
                scene = load_scene(date)
                valid = scene["valid"]
                record["total_pixels"] = int(valid.size)
                record["pixels_excluded"] = int(valid.size - valid.sum())
                if not valid.any():
                    record["status"] = "no valid pixels: no images or rows"
                    print("  no valid pixels in this file, nothing to generate", flush=True)
                    continue
                if CSV_SAMPLE_PIXELS is not None and chosen is None:
                    flat = np.zeros(valid.size, bool)
                    flat[np.random.default_rng(RANDOM_SEED).choice(valid.size, CSV_SAMPLE_PIXELS, replace=False)] = True
                    chosen = flat.reshape(valid.shape)
                bands = scene["bands"]
                ndvi = normalized_difference(bands["B08"], bands["B04"], valid)
                ndwi = normalized_difference(bands["B03"], bands["B08"], valid)
                image_paths += make_images(scene, ndvi, ndwi, date)
                print("  images saved", flush=True)
            except Exception as exc:  # report this date, carry on with the others
                record["status"] = f"failed: {type(exc).__name__}: {exc}"
                failed.append(date)
                print(f"  FAILED: {exc}", flush=True)
                continue
            record["rows_in_dataset"], record["rows_in_csv"] = write_rows(scene, ndvi, ndwi, date, parquet_writer, csv_writer, chosen)
            record["status"] = "processed"
            print(f"  {record['rows_in_dataset']:,} rows written", flush=True)
            del scene, bands, ndvi, ndwi, valid
    parquet_tmp.replace(PARQUET_PATH)
    csv_tmp.replace(CSV_PATH)

    summary = pd.DataFrame(records)
    summary["excluded_pct"] = (100 * summary["pixels_excluded"] / summary["total_pixels"].replace(0, np.nan)).round(3)
    summary.to_csv(STATS_DIR / "date_summary.csv", index=False)
    stats = feature_statistics()
    stats.to_csv(STATS_DIR / "feature_statistics.csv", index=False)

    problems = verify(summary, stats, image_paths)
    print(f"\nDates processed: {(summary['status'] == 'processed').sum()} of {len(DATES)}")
    print(summary[["date", "status", "rows_in_dataset", "rows_in_csv", "excluded_pct"]].to_string(index=False))
    print(f"Images: {len(image_paths)}")
    print(f"Parquet: {PARQUET_PATH.stat().st_size / 1e6:,.0f} MB, CSV: {CSV_PATH.stat().st_size / 1e6:,.0f} MB")
    for problem in problems:
        print("PROBLEM:", problem)
    print("Verification passed" if not problems and not failed else "Verification found problems")
    return 1 if problems or failed else 0


if __name__ == "__main__":
    sys.exit(main())
