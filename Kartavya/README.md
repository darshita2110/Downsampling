# Kartavya: dataset-independent channel / band reduction

Eight channel-reduction algorithms that run on **any** multi-channel image dataset: you only give the
dataset path. Each algorithm runs on its own, and `run_all_algorithms.py` runs all of them and compares them.

| Algorithm | Folder | Kind | Output |
|---|---|---|---|
| PCA | `algorithms/pca` | linear transform | K new components + reconstruction |
| ICA (FastICA) | `algorithms/ica` | linear transform | K independent sources + reconstruction |
| LDA | `algorithms/lda` | supervised projection (**needs labels**) | K discriminant components + classification scores |
| GA | `algorithms/ga` | band selection (genetic algorithm) | K kept original bands + reconstruction |
| PSO | `algorithms/pso` | band selection (particle swarm) | same |
| ACO | `algorithms/aco` | band selection (ant colony) | same |
| SA | `algorithms/sa` | band selection (simulated annealing) | same |
| UBS-CNN | `algorithms/ubscnn` | band selection (BS-Net-Conv attention CNN, **needs PyTorch**) | same |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # torch / rasterio / h5py ... are optional, see the file
```

## Run

```bash
# Option A - one algorithm
python algorithms/pca/main.py                         # asks for the dataset path
python algorithms/pca/main.py --dataset path/to/data  # or set DATASET_PATH at the top of main.py

# Option B - every algorithm + comparison report
python run_all_algorithms.py --dataset path/to/data
python run_all_algorithms.py --dataset path/to/data --only pca,ga,sa --epochs 20
```

Options: `--labels PATH` (LDA ground truth), `--no-standardize` (PCA/ICA/LDA), `--epochs N` (UBS-CNN),
`--output DIR`, `--self-check` (LDA on Iris/Wine), `--protocol spatial|full` (see **Protocol** below),
`--bands N` (true band count of a cube whose band axis is ambiguous). Exit codes: 0 ok, 2 invalid dataset,
3 not applicable (LDA without usable labels).

**Numbers for the paper (same protocol as the team tables)** - run every dataset with `--protocol full`:

```bash
python run_all_algorithms.py --dataset path/to/Indian_pines_corrected.mat --protocol full
python run_all_algorithms.py --dataset path/to/Salinas_corrected.mat     --protocol full
```

## What a dataset can be

| Input | Interpreted as |
|---|---|
| one file | one image (H × W × C) |
| folder of files | one sample per file (same channel count; sizes may differ) |
| folder of single-band files of equal size (`B02.tif`, `B03.tif` …) | the bands of one image |
| folder of sub-folders with the same number of images (`RGB/`, `NIR/` …) | i-th image of each sub-folder stacked into sample i (resampled to one grid) |

Formats: PNG, JPG/JPEG, BMP, GIF, WebP, TIFF/GeoTIFF, JPEG2000, VRT, ERDAS `.img`, ENVI (`.hdr`), NumPy
`.npy`/`.npz`, MATLAB `.mat` (incl. v7.3), NetCDF `.nc`, HDF5 `.h5`. Detected automatically: size, channel
count and order (H,W,C vs C,H,W), dtype, band names, no-data (NaN, nodata value, transparent alpha).
**Band axis:** the standard hyperspectral benchmarks are recognised by file name and their band count is
enforced (Indian Pines corrected 200, Salinas corrected 204, PaviaU 103, Pavia 102, KSC 176, Botswana 145).
For any other cube whose band count is close to the image size the run stops and asks for `--bands N`
instead of guessing (the old "smallest axis = bands" rule read Indian Pines as 145 bands).
Data stay in native precision until normalisation to [0, 1] (8-bit: 0–255; otherwise per-channel training
min/max). Files named like `*_gt*`, `*label*`, `*groundtruth*` are labels for LDA, never image data.

**Protocol** (`--protocol`, default `spatial`):

| Protocol | Fit on | Metrics on | Output folder | Use for |
|---|---|---|---|---|
| `spatial` | ≥ 2 samples: first 60 %; 1 sample: top half | the held-out samples / bottom half | `outputs/<id>/` | how well a reduction transfers to unseen pixels |
| `full` | every pixel | every pixel | `outputs/<id>__full/` | the paper tables (same as the team pipelines, e.g. PSNR on the whole cube) |

Every `metrics.csv` and `comparison.csv` has a `protocol` column. Do not put `spatial` and `full` PSNR values in
one table: `spatial` is measured on unseen pixels and is lower by design.

**LDA classification split** (independent of `--protocol`, see `algorithms/lda/config.py`): LDA needs every
class in both training and test, which the top/bottom split does not give. Its labelled pixels are split
like the team classification pipeline: `auto` = block split (40 % of blocks train) for images ≥ 256 px on
the short side, otherwise stratified random (10 % of each class trains); classes with < 10 labelled pixels
are excluded and listed in `label_report.json`. Macro precision/recall/F1 average only the classes that have
test pixels (`average_accuracy` = AA of the team tables).

## Outputs

```text
outputs/<dataset_name>_<hash>/            # hash of the files -> different data never mix ("__full" suffix: --protocol full)
├── dataset/        filename_index_mapping.json, dataset_inspection.json, split.json, normalization.json
├── <algorithm>/
│   ├── models/              saved models / selected bands (+ .pt networks for UBS-CNN)
│   ├── reduced_data/        reduced_K{k}.npy per test sample (+ reduced_native_dtype_K{k}.npy for band selection)
│   ├── reconstructed_data/  reconstructed_K{k}.npy, original_normalized.npy, valid_mask.npy
│   ├── metrics/             test_metrics.csv (+ per channel / per group), train/test summaries, method-specific
│   │                        tables (explained variance, convergence, exhaustive checks, independence, labels ...)
│   ├── graphs/              reconstruction gallery, reduced-band images, quality/storage/time vs K, method plots
│   ├── logs/                train.log, test.log
│   └── metrics.csv  metrics.json  metrics_report.txt     (per-K summary)
└── comparison/     comparison.csv, comparison.json, metric_comparison/<metric>_by_k.csv + _vs_k.png,
                    execution_time_comparison.(csv|png), total_time_comparison.png,
                    quality_vs_storage_tradeoff.png, final_report.txt, run_all.log
```

Each run of an algorithm replaces that algorithm's folder for that dataset.

## Metrics (unchanged from the original repository)

Reconstruction: MSE, RMSE, MAE, PSNR (MAX = 1), SSIM — overall, per channel, per channel group.
Size: dimensionality reduction %, theoretical storage incl. model, compression ratio, storage saving %, `.npy` sizes.
Cost: forward/inverse time (median of 5), peak memory, process RSS, training time.
Method-specific: explained variance (PCA), convergence + independence + PCA-equivalence check (ICA),
accuracy / precision / recall / F1 / confusion matrix / Fisher ratio (LDA), fitness terms, convergence,
exhaustive-optimum check (GA/PSO/ACO/SA), training curves, band weights, ranking (UBS-CNN).

The comparison knows each metric's direction (errors/time: lower is better; PSNR/SSIM/savings/accuracy:
higher), ranks over the K values shared by all algorithms, reports ties, and leaves non-applicable metrics
empty (e.g. no reconstruction for LDA).

## Repository layout

```text
algorithms/<name>/  main.py (entry point + config section), pipeline.py (train/test), implementation.py (the method), config.py (hyper-parameters)
algorithms/__init__.py   registry / discovery of algorithms
common/  config.py (shared protocol settings), data_loader.py, preprocessing.py, metrics.py, evaluation.py,
         visualization.py, utils.py, pipeline.py (common interface), reduction_eval.py, band_selection.py, comparison.py
documentation/   method write-ups (see documentation/README.md)
run_all_algorithms.py
```

Adding an algorithm: create `algorithms/<name>/pipeline.py` with `NAME, DISPLAY, FAMILY, REQUIRES,
output_name(), train(ctx), test(ctx) -> DataFrame`; it is picked up by `run_all_algorithms.py` automatically.
