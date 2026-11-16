"""
Fit PCA on the training samples and save the model.

Run from the repository root:
    python Kartavya/algorithms/linear/pca/train.py                  # PNG products (8 channels)
    python Kartavya/algorithms/linear/pca/train.py --dataset raw    # original B02/B03/B04/B08, if present
    python Kartavya/algorithms/linear/pca/train.py --no-standardize # PCA on the [0,1] values directly

What it does
  1. lists the images in sorted-filename order and saves the filename -> index mapping
  2. inspects every image (size, mode, dtype, value range, map panel, alignment)
  3. builds the training feature matrix (valid pixels x channels), normalised to [0, 1]
  4. standardises with training statistics and fits PCA (our NumPy implementation)
  5. cross-checks the result against scikit-learn's PCA
  6. saves the model, explained-variance table, graphs and a log
"""
import argparse
import sys
import time
from pathlib import Path

# Make "import Kartavya..." work when this file is run directly from any folder
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd

from Kartavya.algorithms.linear.pca import config as cfg
from Kartavya.algorithms.linear.pca.pca import PCA
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.metrics import mae, mse, psnr, rmse
from Kartavya.common.preprocessing import Standardizer, image_to_matrix, sample_rows
from Kartavya.common.utils import ensure_dirs, get_logger, measure, rss_mb, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train PCA for channel reduction")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--no-standardize", action="store_true", help="skip z-scoring before PCA")
    p.add_argument("--target-width", type=int, default=None, help="override config.TARGET_WIDTH")
    p.add_argument("--tag", default=None, help="experiment name suffix (default: derived from options)")
    return p.parse_args()


def experiment_name(args):
    name = args.dataset + ("_nostd" if args.no_standardize else "")
    return f"{name}_{args.tag}" if args.tag else name


def main():
    args = parse_args()
    if args.target_width:
        cfg.TARGET_WIDTH = args.target_width
    standardize = cfg.STANDARDIZE and not args.no_standardize
    exp = experiment_name(args)
    dirs = cfg.output_dirs(exp)
    ensure_dirs(*dirs.values())
    log = get_logger(f"pca.train.{exp}", dirs["logs"] / "train.log")
    rng = set_seed(cfg.RANDOM_SEED)
    t_start = time.perf_counter()
    log.info(f"Experiment '{exp}' | dataset={args.dataset} | standardize={standardize} | seed={cfg.RANDOM_SEED}")

    # ---------------------------------------------------------- 1. ordering + mapping
    mapping = dl.discover_samples(args.dataset, cfg, log)
    if not mapping["samples"]:
        where = cfg.RAW_DATASET_DIR if args.dataset == "raw" else cfg.PNG_DATASET_DIR
        log.error(f"No usable samples found in {where}. "
                  + ("Original Sentinel-2 bands are not available, so the raw experiment is skipped "
                     "(no bands are fabricated)." if args.dataset == "raw" else ""))
        sys.exit(2)
    save_json(mapping, dirs["metrics"] / "filename_index_mapping.json")
    for s in mapping["samples"]:
        log.info(f"  index {s['index']}: date {s['date']} -> {s['files']}")

    split = dl.make_split(len(mapping["samples"]), cfg)
    save_json(split, dirs["models"] / "split.json")
    log.info(f"Split mode: {split['mode']} | train {split['train']} | test {split['test']}")
    if split["mode"] == "spatial":
        log.warning("Only ONE date is available: fitting on the top half of the image and testing on the "
                    "bottom half. With all five dates, earlier dates train and later dates test.")

    # ---------------------------------------------------------- 2. load + inspect
    stats = {}
    inspection, train_blocks, names = {}, [], None
    with measure(stats, "load_and_preprocess"):
        for entry in split["train"]:
            s = mapping["samples"][entry["index"]]
            data = dl.load_normalized_sample(args.dataset, s, cfg)
            names = data["names"]
            align = dl.alignment_check(data) if args.dataset == "png" else {}
            inspection[s["date"]] = {"files": data["inspection"], "common_grid_hw": data["grid_hw"],
                                     "valid_pct_on_grid": float(100 * data["valid"].mean()),
                                     "channel_natural_range": dict(zip(names, data["channels_natural_range"])),
                                     "normalization": data["normalization"], "alignment": align}
            rs = dl.rows_slice(data["valid"].shape[0], entry["rows"], cfg.SPATIAL_TRAIN_FRACTION)
            valid = data["valid"].copy()
            valid[:rs.start] = False
            valid[rs.stop:] = False
            train_blocks.append(image_to_matrix(data["channels"], valid))
            log.info(f"  loaded {s['date']}: grid {data['grid_hw']}, valid {100 * data['valid'].mean():.3f}%, "
                     f"training pixels {train_blocks[-1].shape[0]:,} (rows {rs.start}-{rs.stop})")
            for f, info in data["inspection"].items():
                extra = ""
                if "colormap_decode" in info:
                    d = info["colormap_decode"]
                    extra = (f", decoded range {info['value_range_decoded']}, exact colour match "
                             f"{d['exact_match_pct']:.1f}%, no-data {d['white_nodata_pct']:.3f}%")
                log.info(f"    {f}: {info.get('mode', '')} {info.get('shape')} {info.get('dtype')} "
                         f"min {info.get('min')} max {info.get('max')}"
                         + (f", map panel {info['map_panel_hw']}" if "map_panel_hw" in info else "") + extra)
            for pair, r in align.items():
                log.info(f"    alignment {pair}: best shift {r['best_shift_dy_dx']}, r={r['corr_at_best']:.4f}")
    save_json(inspection, dirs["metrics"] / "dataset_inspection.json")

    x_all = np.concatenate(train_blocks)
    x_fit = sample_rows(x_all, cfg.MAX_TRAIN_PIXELS, rng)
    log.info(f"Feature matrix: {x_all.shape[0]:,} pixels x {x_all.shape[1]} channels; "
             f"fitting on {x_fit.shape[0]:,} sampled pixels")
    log.info(f"Channels: {names}")

    # ---------------------------------------------------------- 3. standardise + fit
    scaler = Standardizer(standardize).fit(x_fit)
    z_fit = scaler.transform(x_fit)
    with measure(stats, "pca_fit"):
        model = PCA().fit(z_fit)
    log.info(f"PCA fit in {stats['pca_fit']['seconds']:.3f} s, peak {stats['pca_fit']['peak_python_mb']:.1f} MB")

    evr = model.all_explained_variance_ratio_
    cum = np.cumsum(evr)
    ev_table = pd.DataFrame({"component": np.arange(1, len(evr) + 1), "eigenvalue": model.explained_variance_,
                             "explained_variance_ratio": evr, "cumulative_explained_variance": cum})
    ev_table.to_csv(dirs["metrics"] / "explained_variance.csv", index=False)
    for _, r in ev_table.iterrows():
        log.info(f"  PC{int(r.component)}: {100 * r.explained_variance_ratio:6.2f}%  cumulative "
                 f"{100 * r.cumulative_explained_variance:6.2f}%")
    k_for = {f"{int(t * 100)}%": model.components_for_variance(t) for t in cfg.VARIANCE_THRESHOLDS}
    log.info(f"Components needed: {k_for}")

    # ---------------------------------------------------------- 4. cross-check with scikit-learn
    check = {}
    try:
        from sklearn.decomposition import PCA as SkPCA
        sk = SkPCA(svd_solver="full").fit(z_fit)
        check["max_abs_diff_explained_variance_ratio"] = float(np.abs(sk.explained_variance_ratio_ - evr).max())
        # Same axes up to sign: |cos| between matching components should be 1. Components with
        # (near-)zero variance are skipped: when channels are exact duplicates their axes are not unique,
        # so any orthogonal basis of that zero-variance space is equally correct.
        cos = np.abs(np.sum(sk.components_ * model.components_, axis=1))
        nondegenerate = evr > 1e-9
        check["n_zero_variance_components"] = int((~nondegenerate).sum())
        check["min_abs_cosine_nonzero_components"] = float(cos[nondegenerate].min())
        k2 = 2
        rec_ours = model.inverse_transform(model.transform(z_fit[:20000], k2))
        sk2 = SkPCA(n_components=k2, svd_solver="full").fit(z_fit)
        rec_sk = sk2.inverse_transform(sk2.transform(z_fit[:20000]))
        check["max_abs_diff_reconstruction_k2"] = float(np.abs(rec_ours - rec_sk).max())
        log.info(f"scikit-learn cross-check: {check}")
    except ImportError:
        log.warning("scikit-learn not installed; skipping the cross-check")

    # ---------------------------------------------------------- 5. training reconstruction error
    rows = []
    for k in range(1, model.n_features_ + 1):
        rec = scaler.inverse_transform(model.inverse_transform(model.transform(z_fit, k)))
        rows.append({"k": k, "MSE": mse(x_fit, rec), "RMSE": rmse(x_fit, rec), "MAE": mae(x_fit, rec),
                     "PSNR_dB": psnr(x_fit, rec), "cumulative_explained_variance": cum[k - 1]})
    pd.DataFrame(rows).to_csv(dirs["metrics"] / "train_reconstruction.csv", index=False)

    # ---------------------------------------------------------- 6. save model + graphs
    model_path = dirs["models"] / "pca_model.npz"
    st = model.state()
    np.savez(model_path, **{f"pca_{k}": np.asarray(v) for k, v in st.items()},
             scaler_enabled=standardize, scaler_mean=scaler.mean_, scaler_std=scaler.std_,
             channel_names=np.array(names), experiment=exp, dataset=args.dataset,
             target_width=cfg.TARGET_WIDTH)
    log.info(f"Saved model -> {model_path}")

    corr = np.corrcoef(x_fit.T)
    viz.plot_explained_variance(evr, dirs["graphs"] / "explained_variance.png",
                                title=f"Explained variance per component ({exp})",
                                thresholds=cfg.VARIANCE_THRESHOLDS)
    viz.plot_correlation(corr, names, dirs["graphs"] / "channel_correlation.png")
    viz.plot_loadings(model.components_, names, dirs["graphs"] / "pca_loadings.png")

    summary = {"experiment": exp, "dataset": args.dataset, "standardize": standardize,
               "channels": names, "n_channels": len(names), "split": split,
               "n_training_pixels_total": int(x_all.shape[0]), "n_pixels_used_for_fit": int(x_fit.shape[0]),
               "explained_variance_ratio": evr, "cumulative_explained_variance": cum,
               "components_for_variance": k_for, "sklearn_check": check, "timing_memory": stats,
               "process_rss_mb_end": rss_mb(), "total_seconds": time.perf_counter() - t_start,
               "model_file": model_path}
    save_json(summary, dirs["metrics"] / "train_summary.json")
    log.info(f"Training finished in {summary['total_seconds']:.1f} s. Outputs in Kartavya/outputs/*/pca/{exp}/")


if __name__ == "__main__":
    main()
