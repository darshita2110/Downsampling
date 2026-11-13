# Kartavya: dimensionality reduction for Raichur Sentinel-2 imagery

Everything in this folder is new. Nothing outside `Kartavya/` was changed.

## 1. The repository you are in

```text
Downsampling/                         (repository root)
├── requirements.txt                  versions used by the original data pipeline
├── .gitignore                        ignores data/ (too large for GitHub)
├── docs/DATA_ACCESS_GUIDE.md         how to get the data, what it contains, caveats
├── scripts/
│   ├── download_s2_many_days.py      downloads Sentinel-2 L1C B02/B03/B04/B08 from Copernicus (openEO)
│   ├── main.py                       makes the RGB / False_Color / NDVI / NDWI PNGs + Parquet/CSV dataset
│   └── explore_s2.py                 optional viewer for a raw .SAFE product
├── reports/dataset_statistics/       per-date row counts and per-feature min/max/mean
├── darshita/                         a teammate's band-selection work (ranking methods, Indian Pines)
├── data/                             NOT on GitHub, you add it yourself
│   ├── processed/sentinel2/{RGB,False_Color,NDVI,NDWI}/   5 PNGs each (input of the PNG experiment)
│   └── raw/sentinel2_l1c/*.nc                             original bands (input of the raw experiment)
└── Kartavya/                         this project
    ├── algorithms/
    │   └── linear/pca/
    │       ├── config.py             every setting (paths, split, K values, normalisation)
    │       ├── pca.py                PCA implemented from scratch in NumPy
    │       ├── train.py              inspect data, fit PCA, save the model
    │       └── test.py               reduce + reconstruct for every K, metrics, graphs
    ├── common/                       shared by all future algorithms (ICA, LDA, GA, PSO ...)
    │   ├── data_loader.py            sorted loading, index mapping, figure-panel cropping,
    │   │                             colour-map inversion, alignment check, raw NetCDF loading
    │   ├── preprocessing.py          [0,1] normalisation, image <-> matrix, standardiser, splits
    │   ├── metrics.py                MSE, RMSE, MAE, PSNR, SSIM
    │   ├── evaluation.py             dimensionality reduction %, storage sizes
    │   ├── visualization.py          all graphs
    │   └── utils.py                  logging, timing + memory measurement, JSON
    ├── documentation/PCA.md          PCA from the basics, design decisions, results
    ├── outputs/                      everything the scripts produce (see section 4)
    └── requirements.txt
```

## 2. Set up from scratch

You need Python 3.10 or newer (tested on 3.13) and git.

```bash
# 1. get the code
git clone https://github.com/darshita2110/Downsampling.git
cd Downsampling

# 2. create an isolated Python environment (keeps these packages away from your system Python)
python -m venv .venv
#    activate it - Windows:      .venv\Scripts\activate
#                  macOS/Linux:  source .venv/bin/activate

# 3. install the packages Kartavya needs
pip install -r Kartavya/requirements.txt
```

**Add the data.** Put the processed images here, keeping the names `scripts/main.py` gave them:

```text
data/processed/sentinel2/RGB/raichur_rgb_<date>.png                 (5 files)
data/processed/sentinel2/False_Color/raichur_false_color_<date>.png (5 files)
data/processed/sentinel2/NDVI/raichur_ndvi_<date>.png               (5 files)
data/processed/sentinel2/NDWI/raichur_ndwi_<date>.png               (5 files)
```

Optional, for the experiment on the original bands: `data/raw/sentinel2_l1c/raichur_s2_<date>.nc`
(from `scripts/download_s2_many_days.py`, see `docs/DATA_ACCESS_GUIDE.md`).

## 3. Run

Always run from the repository root (the `Downsampling/` folder).

```bash
# PNG products, 8 channels, standardised (default)
python Kartavya/algorithms/linear/pca/train.py
python Kartavya/algorithms/linear/pca/test.py

# Same, without standardisation (PCA on the [0,1] values directly)
python Kartavya/algorithms/linear/pca/train.py --no-standardize
python Kartavya/algorithms/linear/pca/test.py  --no-standardize

# Original Sentinel-2 bands B02, B03, B04, B08 (only if data/raw/sentinel2_l1c/*.nc exist;
# otherwise train.py says so and stops - no bands are invented)
python Kartavya/algorithms/linear/pca/train.py --dataset raw
python Kartavya/algorithms/linear/pca/test.py  --dataset raw
```

Use the same options for `train.py` and `test.py`. A run of both takes about 25 seconds at the
default 1024-pixel width. Settings (split, K values, resolution) are in `config.py`.

## 4. Where results go

Each experiment (`png`, `png_nostd`, `raw`) writes to `Kartavya/outputs/<category>/pca/<experiment>/`:

| Folder | Contents |
|---|---|
| `models/` | `pca_model.npz` (mean, components, eigenvalues, standardiser), `split.json` |
| `reduced_data/<sample>/` | `reduced_K{k}.npy`: the K-channel component image (H x W x K, float32) |
| `reconstructed_data/<sample>/` | `reconstructed_K{k}.npy`, `original_normalized.npy`, `valid_mask.npy` |
| `metrics/` | `filename_index_mapping.json`, `dataset_inspection.json`, `explained_variance.csv`, `train_reconstruction.csv`, `test_metrics.csv` (one row per K), `test_metrics_per_product.csv`, `test_metrics_per_channel.csv`, `train_summary.json`, `test_summary.json` |
| `graphs/` | explained variance, channel correlation, loadings, quality vs K, per-product RMSE/SSIM, storage/time/memory vs K, PSNR vs storage saving, reconstruction gallery, component images |
| `logs/` | `train.log`, `test.log` |

The `.npy` arrays (~225 MB per experiment) are git-ignored; re-create them by running the scripts.
