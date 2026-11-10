# Raichur Sentinel-2 Data: Access Guide

This guide explains how to get the Raichur heat and red-zone project's satellite data, what the dataset contains, and what to watch for before modelling with it.

GitHub holds only the code and small statistics files. The data is too large for GitHub, which rejects any file over 100 MB, so you need to get it separately.

## 1. Get the data

Set up the code first (Python 3.14 was used):

```bash
git clone <repository-url>
cd DIP_project
pip install -r requirements.txt
```

Then use one of these two options.

### Option A: Download the ready-made files (recommended)

**Shared data folder:** `[PROJECT OWNER: PASTE THE SHARED FOLDER LINK HERE]`

Put the downloaded folders in these locations, keeping the paths exactly as shown:

| Item | Location in the project | Size |
|---|---|---|
| Combined dataset, Parquet and CSV | `data/processed/ML_Dataset/combined/` | 808 MB |
| RGB, False Colour, NDVI and NDWI images | `data/processed/sentinel2/` | 192 MB |
| Raw NetCDF files (only needed to re-run the pipeline) | `data/raw/sentinel2_l1c/` | 213 MB |

If you only need a quick look, `raichur_features.csv` (37 MB) is enough.

### Option B: Regenerate everything from Copernicus

This needs a free [Copernicus Data Space](https://dataspace.copernicus.eu/) account. Run both commands from the project folder:

```bash
python scripts/download_s2_many_days.py
python scripts/main.py
```

The first command opens a browser to log in, waits for a batch job, and saves one NetCDF file per date, about 800 MB in total. It also downloads dates that the pipeline doesn't use. Copernicus may reprocess its archive, so regenerated values aren't guaranteed to match the shared files exactly.

`main.py` overwrites its previous outputs and never changes the raw files. It ends by checking its own outputs and prints `Verification passed` if they are all fine.

## 2. Repository layout

```text
DIP_project/
├── docs/DATA_ACCESS_GUIDE.md      this guide
├── requirements.txt
├── scripts/
│   ├── main.py                    images, dataset and statistics
│   ├── download_s2_many_days.py   downloads Sentinel-2 data from Copernicus
│   └── explore_s2.py              optional viewer for a raw .SAFE product
├── reports/dataset_statistics/
│   ├── date_summary.csv           rows and excluded pixels per date
│   └── feature_statistics.csv     min, max, mean and missing % per date
└── data/                          not on GitHub
    ├── raw/sentinel2_l1c/
    └── processed/
        ├── sentinel2/{RGB,False_Color,NDVI,NDWI}/
        └── ML_Dataset/combined/
```

## 3. The dataset

| File | Contents |
|---|---|
| `raichur_features.parquet` | Every usable pixel on every date: 37,927,814 rows |
| `raichur_features.csv` | A sample of 347,944 rows, covering the same pixel locations on each date |

The full table would make a very large CSV that Excel couldn't open, because Excel stops at about 1 million rows. **Use the Parquet file for training and the CSV for quick exploration.** To change the sample size, or export every row, edit `CSV_SAMPLE_PIXELS` in `scripts/main.py`.

### Columns

| Column | Meaning |
|---|---|
| `x`, `y` | Pixel-centre coordinates in metres, EPSG:32643 (UTM zone 43N) |
| `latitude`, `longitude` | The same position in degrees, WGS84 |
| `date` | Acquisition date |
| `B02`, `B03`, `B04`, `B08` | Blue, Green, Red and Near Infrared, stored as reflectance × 10000 |
| `NDVI` | `(B08 - B04) / (B08 + B04)` |
| `NDWI` | `(B03 - B08) / (B03 + B08)` (McFeeters) |
| `LST`, `population_density`, `infrastructure_density` | Empty, not yet integrated |

To get reflectance, divide a band value by 10000. Don't subtract the 1000 offset that newer Sentinel-2 products use: comparing the 2026-10-27 file with the original Copernicus product showed that it has already been removed. Every date also has values below 1000, which supports this for the other dates.

### Dates

| Date | Rows | Area covered | Condition |
|---|---:|---:|---|
| 2026-09-17 | 10,901,746 | 100.0% | Heavy cloud and shadow |
| 2026-09-27 | 10,901,746 | 100.0% | Mostly clear, some small clouds |
| 2026-09-29 | 0 | 0% | **No valid pixels, so no images or rows** |
| 2026-10-02 | 7,990,867 | 73.3% | Northern part only, with haze in the south of that area |
| 2026-10-04 | 142,588 | 1.3% | Thin strip at the south-east edge, with cloud |
| 2026-10-07 | 7,990,867 | 73.3% | Northern part only, heavy cloud |

"Condition" comes from looking at the images, not from a measurement. The grid is the same on every date, so `x` and `y` identify the same 10 m pixel across dates.

### Load the data

The full table is too large to load at once on most laptops, so read only what you need. One date with eight columns takes about 400 MB of memory.

```python
import datetime as dt
import pandas as pd

PATH = "data/processed/ML_Dataset/combined/raichur_features.parquet"

# One date, selected columns
one_date = pd.read_parquet(
    PATH,
    columns=["x", "y", "latitude", "longitude", "B04", "B08", "NDVI", "NDWI"],
    filters=[("date", "==", dt.date(2026, 9, 27))],
)

# One pixel through time
pixel = pd.read_parquet(
    PATH,
    columns=["date", "NDVI", "NDWI"],
    filters=[("x", "==", 735275), ("y", "==", 1814825)],
)

# Compare dates using the CSV sample: one row per pixel, one column per date
sample = pd.read_csv("data/processed/ML_Dataset/combined/raichur_features.csv", parse_dates=["date"])
ndvi_by_date = sample.pivot(index=["x", "y"], columns="date", values="NDVI")
```

## 4. Read this before modelling

- **Clouds and shadows are still in the data, and nothing marks them.** The files have no cloud band and no cloud masking was applied. Cloud pixels can look like bare land in NDVI or like water in NDWI, so training on 2026-09-17 or 2026-10-07 without masking will mislead a model.
- **Values are top-of-atmosphere.** These are Level-1C products with no atmospheric correction, so NDVI and NDWI will differ from surface-reflectance values, and published thresholds may not apply directly.
- **Coverage differs by date.** When comparing dates, compare the same pixels by joining on `x` and `y`. Scene-wide averages aren't comparable across dates that cover different areas.
- **Only fully valid pixels are included.** A pixel appears for a date only if all four bands have valid values. `reports/dataset_statistics/date_summary.csv` gives the excluded count for each date.
- **The six dates aren't the cleanest available.** The raw download also has 2026-10-09, 2026-10-12 and 2026-10-27, which have full coverage and look much clearer, and 2026-10-14, which is only a thin strip. To add any of them, edit `DATES` in `scripts/main.py` and re-run it.

## 5. Future work

Sentinel-2 can't supply these features, so they stay empty until real datasets are added.

| Feature | Possible source | Note |
|---|---|---|
| `LST` | Landsat 8/9 Collection 2 Level-2 surface temperature | Sentinel-2 has no thermal band, and Landsat dates won't always match these dates |
| `population_density` | WorldPop or GHSL population | About 100 m resolution, coarser than this 10 m grid |
| `infrastructure_density` | GHSL built-up surface or OpenStreetMap | Needs to be converted to a density raster |
| NDBI, MNDWI | Sentinel-2 band B11 (SWIR) | Not in the current download |
| Cloud masking | Sentinel-2 L2A `SCL` band | Also gives atmospherically corrected values |

## 6. Data source and credit

| | |
|---|---|
| Source | Copernicus Data Space Ecosystem, openEO collection `SENTINEL2_L1C` |
| Study area | Longitude 77.2 to 77.5, latitude 16.1 to 16.4 |
| Grid | 10 m, 3247 × 3358 pixels |
| Scene filter | Maximum 20% cloud cover per scene |
| Product used for the scaling check | `S2C_MSIL1C_20261027T051911_N0513_R062_T43QGU_20261027T085813` |

When you publish results or share the data, credit it as: **"Contains modified Copernicus Sentinel data 2026."**
