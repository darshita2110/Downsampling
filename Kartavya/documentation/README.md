# Method documentation

`PCA.md`, `ICA.md`, `LDA.md`, `GA.md`, `PSO.md`, `ACO.md`, `SA.md` and `UBSCNN.md` explain each method from
the basics and the design decisions behind it. They were written for the earlier Sentinel-2 PNG experiment.

**The numerical results quoted in those files belong to that earlier dataset and are not valid results of the
current pipeline.** Run the algorithms on your own data to get current results
(`outputs/<dataset_id>/<algorithm>/metrics_report.txt`, `outputs/<dataset_id>/comparison/final_report.txt`).

Methodology and hyper-parameters described there are unchanged, with these generalisations:

| Before (dataset-specific) | Now (dataset-independent) |
|---|---|
| hard-coded folders RGB / False_Color / NDVI / NDWI, cropping of matplotlib figures, colour-map decoding | format detected from the file; any image / raster / array file, folder of files, folder of single-band files, or folder of product sub-folders |
| fixed value ranges per product (0-255, -1..1) | dtype range for 8-bit data (identical to before), otherwise per-channel min/max of the training pixels |
| fixed 8 channels, `TARGET_WIDTH = 1024` resampling | any number of channels ≥ 2, native resolution (optional `MAX_IMAGE_SIDE`) |
| K = 1 … C−1 | same for C ≤ 16; for hyperspectral data a shared grid of K values (`common/config.py`) |
| exhaustive subset check always | only when C-choose-K ≤ `EXHAUSTIVE_MAX_SUBSETS` (500); otherwise reported as "not run" |
| UBS-CNN stride 16 | 16, lowered automatically for small images until ≥ 512 training patches exist |
| LDA labels at `data/labels/sentinel2/raichur_labels_<date>` | label files detected by name (`*_gt*`, `*label*` …) or `--labels PATH` |
