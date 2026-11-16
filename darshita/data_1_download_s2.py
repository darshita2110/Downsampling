"""
DATA STEP 1 - Download ALL 12 Sentinel-2 bands over Raichur (real satellite data).

Why: the team's dataset has only 4 bands (B02, B03, B04, B08). Band selection needs the
full set of real bands to choose from. Sentinel-2 Level-2A has 12 spectral bands:

    B01 coastal  443 nm   B05 red-edge 705 nm   B8A narrow NIR 865 nm
    B02 blue     490 nm   B06 red-edge 740 nm   B09 water vapour 945 nm
    B03 green    560 nm   B07 red-edge 783 nm   B11 SWIR-1 1610 nm
    B04 red      665 nm   B08 NIR      842 nm   B12 SWIR-2 2190 nm

(B10 exists only in Level-1C, so L2A has 12.)  We also download SCL, the "scene
classification" layer that marks clouds, shadows and no-data pixels.

Same area and same Copernicus login as scripts/download_s2_many_days.py.

Install : pip install openeo xarray netcdf4
Run     : python darshita/data_1_download_s2.py        (from the project folder)
Output  : data/darshita/raichur_<date>.npy       image, shape (height, width, 12), reflectance 0..1
          data/darshita/raichur_<date>_scl.npy   cloud/shadow class of every pixel
          data/darshita/raichur_bands.json       band names and wavelengths
"""
import json
import os
from datetime import date, timedelta

import numpy as np

# ----------------------------- CONFIG ---------------------------------
# Clear, full-coverage dates (see docs/DATA_ACCESS_GUIDE.md, section 4). Add more if you like.
DATES      = ["2026-09-27", "2026-10-09", "2026-10-12", "2026-10-27"]
BBOX       = {"west": 77.2, "south": 16.1, "east": 77.5, "north": 16.4}   # same as the team
BANDS      = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
WAVELENGTH = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]   # nm, centre
RESOLUTION = 20            # metres per pixel. 20 m keeps one date at about 1650 x 1650 pixels
OUT_DIR    = "data/darshita"
RAW_DIR    = "data/darshita/raw_netcdf"
# -----------------------------------------------------------------------


def download_one_date(con, day):
    """Ask Copernicus for one date and save the result as a NetCDF file. Returns its path."""
    nc_path = os.path.join(RAW_DIR, f"raichur_s2_l2a_{day}.nc")
    if os.path.exists(nc_path):
        print(f"{day}: NetCDF already downloaded")
        return nc_path
    next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()   # end date is exclusive
    cube = con.load_collection(
        "SENTINEL2_L2A",                       # L2A = surface reflectance (atmosphere removed)
        spatial_extent=BBOX,
        temporal_extent=[day, next_day],
        bands=BANDS + ["SCL"],
    )
    # 10 m, 20 m and 60 m bands all put on one 20 m grid.
    # "near" (nearest neighbour) keeps the SCL cloud classes as whole numbers.
    cube = cube.resample_spatial(resolution=RESOLUTION, method="near")
    print(f"{day}: submitting job to Copernicus (usually 2-10 minutes) ...")
    try:
        cube.execute_batch(nc_path, out_format="netCDF", title=f"S2 L2A 12 bands Raichur {day}")
    except Exception as e:
        print(f"{day}: FAILED - {str(e)[:200]}")
        return None
    return nc_path


def netcdf_to_numpy(nc_path, day):
    """Turn the NetCDF file into a plain numpy cube (height, width, 12) that any script can load."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    tdim = "t" if "t" in ds.dims else ("time" if "time" in ds.dims else None)
    if tdim is not None and ds.sizes[tdim] > 1:
        # two satellite tiles can cover Raichur on the same day: keep the first valid value
        ds_first = ds.isel({tdim: 0})
        ds = ds_first.fillna(ds.isel({tdim: 1}))
    elif tdim is not None:
        ds = ds.isel({tdim: 0})

    bands = []
    for b in BANDS:
        values = ds[b].values.astype(np.float32)
        bands.append(values)
    cube = np.stack(bands, axis=-1)                       # (height, width, 12)
    nodata = ~np.isfinite(cube).all(axis=-1) | (cube == 0).all(axis=-1)
    cube = np.nan_to_num(cube, nan=0.0) / 10000.0         # stored as reflectance x 10000
    scl = np.nan_to_num(ds["SCL"].values, nan=0).astype(np.uint8)
    scl[nodata] = 0                                       # 0 = no data in SCL
    ds.close()

    np.save(os.path.join(OUT_DIR, f"raichur_{day}.npy"), cube)
    np.save(os.path.join(OUT_DIR, f"raichur_{day}_scl.npy"), scl)
    print(f"{day}: saved raichur_{day}.npy  shape {cube.shape}  ({cube.nbytes / 1e6:.0f} MB)")


def main():
    import openeo

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "raichur_bands.json"), "w") as f:
        json.dump({"bands": BANDS, "wavelength_nm": WAVELENGTH, "resolution_m": RESOLUTION,
                   "units": "surface reflectance, 0..1", "collection": "SENTINEL2_L2A"}, f, indent=1)

    # Opens a browser to log in the first time (same account as the team's script).
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    for day in DATES:
        if os.path.exists(os.path.join(OUT_DIR, f"raichur_{day}.npy")):
            print(f"{day}: already done")
            continue
        nc_path = download_one_date(con, day)
        if nc_path:
            netcdf_to_numpy(nc_path, day)
    print("\nNext: python darshita/data_2_check_s2.py")


if __name__ == "__main__":
    main()
