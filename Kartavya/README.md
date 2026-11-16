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
    │   └── linear/
    │       ├── pca/                  config.py, pca.py (NumPy PCA), train.py, test.py
    │       ├── ica/                  config.py, ica.py (NumPy FastICA), train.py, test.py
    │       └── lda/                  config.py, lda.py (NumPy Fisher LDA), train.py, test.py (needs labels)
    │   └── combinatorial/
    │       ├── ga/                   config.py, ga.py (NumPy genetic algorithm), train.py, test.py
    │       ├── pso/                  config.py, pso.py (NumPy particle swarm), train.py, test.py
    │       ├── aco/                  config.py, aco.py (NumPy ant colony), train.py, test.py
    │       ├── sa/                   config.py, sa.py (NumPy simulated annealing), train.py, test.py
    │       └── ubscnn/               config.py, model.py (PyTorch BS-Net-Conv), train.py, test.py
    ├── common/                       shared by all future algorithms (ICA, LDA, GA, PSO ...)
    │   ├── data_loader.py            sorted loading, index mapping, figure-panel cropping,
    │   │                             colour-map inversion, alignment check, raw NetCDF loading
    │   ├── preprocessing.py          [0,1] normalisation, image <-> matrix, standardiser, splits
    │   ├── metrics.py                MSE, RMSE, MAE, PSNR, SSIM
    │   ├── evaluation.py             dimensionality reduction %, storage sizes
    │   ├── visualization.py          all graphs
    │   └── utils.py                  logging, timing + memory measurement, JSON
    ├── documentation/            PCA.md, ICA.md, LDA.md, GA.md, PSO.md, ACO.md, SA.md, UBSCNN.md: each method from the basics, decisions, results
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

ICA uses the same commands with `ica` in place of `pca`:

```bash
python Kartavya/algorithms/linear/ica/train.py      # fits one FastICA model per K on TRAINING pixels only
python Kartavya/algorithms/linear/ica/test.py       # loads the saved models; nothing is re-fitted
python Kartavya/algorithms/linear/ica/train.py --no-standardize
python Kartavya/algorithms/linear/ica/test.py  --no-standardize
```

ICA also writes `metrics/convergence.csv` (iterations, converged flag, final change, scikit-learn match),
`metrics/test_independence.csv`, and the graphs `convergence.png`, `independence_ica_vs_pca.png`
and `ica_mixing_K6.png`. Its models are `models/ica/<experiment>/ica_K{k}.npz`, one per K.

LDA is supervised and needs real per-pixel class labels in `data/labels/sentinel2/`
(see `documentation/LDA.md`). The dataset has none, so today `train.py` inspects the images, writes
`metrics/lda/png/label_report.json`, fits nothing and exits with code 3; `test.py` does the same.

```bash
python Kartavya/algorithms/linear/lda/train.py                # stops with the label report (exit code 3)
python Kartavya/algorithms/linear/lda/test.py
python Kartavya/algorithms/linear/lda/train.py --self-check   # verifies the code on Iris and Wine
```

The Genetic Algorithm selects K of the 8 channels to keep (no new values are created) and rebuilds
the dropped ones with a small linear model. See `documentation/GA.md`.

```bash
python Kartavya/algorithms/combinatorial/ga/train.py   # GA on TRAINING pixels, K = 1..7, 5 seeds each (~75 s)
python Kartavya/algorithms/combinatorial/ga/test.py    # applies the saved subsets to the test region (~20 s)
```

GA also writes `metrics/selected_features.csv`, `convergence.csv`, `ga_vs_exhaustive.csv`,
`ga_runs_K{k}.csv`, `ga_vs_pca.csv` and the graphs `ga_convergence.png`, `selected_channels.png`,
`fitness_terms_vs_k.png`, `ga_vs_pca.png`. Its models are `models/ga/png/ga_K{k}.npz`.

Particle Swarm Optimization selects channels with the same fitness function as the GA
(see `documentation/PSO.md`):

```bash
python Kartavya/algorithms/combinatorial/pso/train.py  # PSO on TRAINING pixels, K = 1..7, 5 swarms each (~75 s)
python Kartavya/algorithms/combinatorial/pso/test.py   # applies the saved subsets to the test region (~20 s)
```

Ant Colony Optimization uses the same fitness function as the GA and PSO (see `documentation/ACO.md`):

```bash
python Kartavya/algorithms/combinatorial/aco/train.py  # ACO on TRAINING pixels, K = 1..7, 5 colonies each (~60 s)
python Kartavya/algorithms/combinatorial/aco/test.py   # applies the saved subsets to the test region (~20 s)
```

Simulated Annealing uses the same objective as the GA, PSO and ACO (see `documentation/SA.md`):

```bash
python Kartavya/algorithms/combinatorial/sa/train.py   # SA on TRAINING pixels, K = 1..7, 5 starts each (~90 s)
python Kartavya/algorithms/combinatorial/sa/test.py    # evaluation only: applies the saved subsets (~15 s)
```

UBS-CNN is implemented as BS-Net-Conv (Cai et al., IEEE TGRS 2020), see `documentation/UBSCNN.md`.
It needs PyTorch (`pip install torch`, CPU is enough):

```bash
python Kartavya/algorithms/combinatorial/ubscnn/train.py   # trains 3 networks x 100 epochs (~20 min on 2 CPU cores)
python Kartavya/algorithms/combinatorial/ubscnn/test.py    # no training: applies the saved selection to the test region
```

Use the same options for `train.py` and `test.py`. A run of both takes about 25 seconds at the
default 1024-pixel width. Settings (split, K values, resolution) are in `config.py`.

## 4. Where results go

Each experiment (`png`, `png_nostd`, `raw`) writes to `Kartavya/outputs/<category>/<pca|ica|lda|ga|pso|aco|sa|ubscnn>/<experiment>/`:

| Folder | Contents |
|---|---|
| `models/` | `pca_model.npz` (mean, components, eigenvalues, standardiser), `split.json` |
| `reduced_data/<sample>/` | `reduced_K{k}.npy`: the K-channel component image (H x W x K, float32) |
| `reconstructed_data/<sample>/` | `reconstructed_K{k}.npy`, `original_normalized.npy`, `valid_mask.npy` |
| `metrics/` | `filename_index_mapping.json`, `dataset_inspection.json`, `explained_variance.csv`, `train_reconstruction.csv`, `test_metrics.csv` (one row per K), `test_metrics_per_product.csv`, `test_metrics_per_channel.csv`, `train_summary.json`, `test_summary.json` |
| `graphs/` | explained variance, channel correlation, loadings, quality vs K, per-product RMSE/SSIM, storage/time/memory vs K, PSNR vs storage saving, reconstruction gallery, component images |
| `logs/` | `train.log`, `test.log` |

The `.npy` arrays (~225 MB per experiment) are git-ignored; re-create them by running the scripts.
