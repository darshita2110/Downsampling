"""
download_s2_many_days.py
Download Sentinel-2 L1C (B02, B03, B04, B08) for MANY days over Raichur in one run.

MODE = "range"  -> ONE batch job for the whole date range, then split into one file per day (recommended)
MODE = "loop"   -> one small request per day (simple, but wastes requests: S2 passes only every ~5 days)

Install : pip install openeo xarray netcdf4
Run     : python scripts/download_s2_many_days.py   (from the project folder)
"""
import os
import time
from datetime import date, timedelta

import openeo
import xarray as xr

# ----------------------------- CONFIG ---------------------------------
MODE      = "range"                      # "range" or "loop"
START     = "2026-08-01"                 # first day wanted (inclusive)
END       = "2026-10-29"                 # day AFTER the last day wanted (openEO end is exclusive)
BBOX      = {"west": 77.2, "south": 16.1, "east": 77.5, "north": 16.4}
BANDS     = ["B02", "B03", "B04", "B08"]
MAX_CLOUD = 20                           # percent
OUT_DIR   = "data/raw/sentinel2_l1c"
# -----------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def connect():
    # opens a browser window the first time to log in; token is cached afterwards
    return openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()


def build_cube(con, start, end):
    return con.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=BBOX,
        temporal_extent=[start, end],
        bands=BANDS,
        max_cloud_cover=MAX_CLOUD,
    )


# ------------------------- MODE 1: one request, split by day -----------
def split_by_day(nc_path, out_dir):
    """Cut a multi-date NetCDF into one file per calendar day. Returns list of saved days."""
    ds = xr.open_dataset(nc_path)
    tdim = "t" if "t" in ds.dims else "time"
    days = ds[tdim].dt.strftime("%Y-%m-%d").values          # date string of every time step
    saved = []
    for day in sorted(set(days)):
        idx = [i for i, d in enumerate(days) if d == day]
        sub = ds.isel({tdim: idx})
        if len(idx) > 1:                                     # 2+ acquisitions same day -> merge
            sub = sub.mean(dim=tdim, skipna=True)
        else:
            sub = sub.isel({tdim: 0})
        out = os.path.join(out_dir, f"raichur_s2_{day}.nc")
        sub.to_netcdf(out)
        saved.append(day)
        print(f"  saved {out}")
    ds.close()
    return saved


def run_range_mode():
    range_file = os.path.join(OUT_DIR, f"raichur_s2_range_{START}_to_{END}.nc")
    if not os.path.exists(range_file):                       # skip re-download if already there
        con = connect()
        cube = build_cube(con, START, END)
        print("Submitting batch job ... (waits until finished, then downloads)")
        cube.execute_batch(range_file, title="S2 Raichur many days")
    else:
        print("Range file already exists, skipping download.")
    days = split_by_day(range_file, OUT_DIR)
    print(f"\nDone. {len(days)} days with data: {days}")


# ------------------------- MODE 2: loop over days ----------------------
def daterange(start, end):
    d = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while d < stop:
        yield d
        d += timedelta(days=1)


def run_loop_mode():
    con = connect()
    got, empty, failed = [], [], []
    for d in daterange(START, END):
        day = d.isoformat()
        out = os.path.join(OUT_DIR, f"raichur_s2_{day}.nc")
        if os.path.exists(out):                              # resume support
            got.append(day)
            continue
        nxt = (d + timedelta(days=1)).isoformat()
        for attempt in range(3):                             # retry on temporary errors
            try:
                build_cube(con, day, nxt).download(out)
                got.append(day)
                print(f"{day}: downloaded")
                break
            except Exception as e:
                msg = str(e)
                if "NoDataAvailable" in msg or "no data" in msg.lower():
                    empty.append(day)
                    print(f"{day}: no scene (no pass / too cloudy)")
                    break
                print(f"{day}: attempt {attempt + 1} failed: {msg[:120]}")
                time.sleep(10)
        else:
            failed.append(day)
        time.sleep(6)                                        # stay under request-rate limits
    print(f"\nDownloaded {len(got)} | empty {len(empty)} | failed {len(failed)}")
    if failed:
        print("Re-run the script to retry:", failed)


if __name__ == "__main__":
    run_range_mode() if MODE == "range" else run_loop_mode()
