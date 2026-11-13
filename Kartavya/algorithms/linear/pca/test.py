"""
Apply the trained PCA to the test samples for every K and measure what is lost and what is saved.

Run from the repository root (after train.py, with the same options):
    python Kartavya/algorithms/linear/pca/test.py
    python Kartavya/algorithms/linear/pca/test.py --dataset raw
    python Kartavya/algorithms/linear/pca/test.py --no-standardize

For every test sample and every K = 1 .. C it
  1. projects pixels to K components (forward transform) and reshapes them into a K-channel image
  2. reconstructs C channels with the inverse transform and reshapes into an image
  3. computes MSE, RMSE, MAE, PSNR, SSIM (overall, per channel, per product)
  4. computes dimensionality reduction %, storage sizes, time and memory
and saves reduced/reconstructed arrays, CSV/JSON metrics, graphs and a log.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd

from Kartavya.algorithms.linear.pca import config as cfg
from Kartavya.algorithms.linear.pca.pca import PCA
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.evaluation import dimensionality_reduction_pct, storage_report
from Kartavya.common.metrics import all_metrics
from Kartavya.common.preprocessing import Standardizer, image_to_matrix, matrix_to_image
from Kartavya.common.utils import (ensure_dirs, file_size_bytes, get_logger, human_bytes, load_json,
                                   measure, rss_mb, save_json)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PCA channel reduction")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def load_model(path):
    d = np.load(path, allow_pickle=False)
    model = PCA.from_state({k[4:]: d[k] for k in d.files if k.startswith("pca_")})
    scaler = Standardizer.from_state({"enabled": bool(d["scaler_enabled"]), "mean": d["scaler_mean"],
                                      "std": d["scaler_std"]})
    return model, scaler, [str(n) for n in d["channel_names"]], int(d["target_width"])


def main():
    args = parse_args()
    exp = args.dataset + ("_nostd" if args.no_standardize else "") + (f"_{args.tag}" if args.tag else "")
    dirs = cfg.output_dirs(exp)
    ensure_dirs(*dirs.values())
    log = get_logger(f"pca.test.{exp}", dirs["logs"] / "test.log")
    t_start = time.perf_counter()

    model_path = dirs["models"] / "pca_model.npz"
    if not model_path.exists():
        log.error(f"No model at {model_path}. Run train.py with the same options first.")
        sys.exit(2)
    model, scaler, names, target_width = load_model(model_path)
    cfg.TARGET_WIDTH = target_width                     # same grid as training
    split = load_json(dirs["models"] / "split.json")
    mapping = load_json(dirs["metrics"] / "filename_index_mapping.json")
    current = dl.discover_samples(args.dataset, cfg)
    if [s["files"] for s in current["samples"]] != [s["files"] for s in mapping["samples"]]:
        log.error("The files on disk differ from those used in training (filename -> index mapping "
                  "changed). Re-run train.py so that both steps use the same ordering.")
        sys.exit(2)

    C = model.n_features_
    ks = cfg.COMPONENT_COUNTS or list(range(1, C + 1))
    save_k = set(cfg.SAVE_RECONSTRUCTED_K or ks)
    gallery_k = [k for k in cfg.GALLERY_K if k in ks]
    groups = dl.channel_groups(names)
    n_model_extra = 2 * C if scaler.enabled else 0       # standardiser mean + std also travel with the model
    log.info(f"Experiment '{exp}': model with {C} channels {names}; testing K = {ks}")

    records, group_records, per_channel_records = [], [], []
    for entry in split["test"]:
        sample = mapping["samples"][entry["index"]]
        date = sample["date"]
        tag = f"{entry['index']:02d}_{date}" + ("" if entry["rows"] == "all" else f"_{entry['rows']}")
        load_stats = {}
        with measure(load_stats, "load"):
            data = dl.load_normalized_sample(args.dataset, sample, cfg)
        rs = dl.rows_slice(data["valid"].shape[0], entry["rows"], split.get("spatial_train_fraction", 0.5))
        img = data["channels"][rs]                      # (H, W, C) in [0, 1]
        valid = data["valid"][rs]
        x = image_to_matrix(img, valid)                 # (N, C)
        n_pix = x.shape[0]
        h, w = valid.shape
        if n_pix == 0:
            log.warning(f"{tag}: no valid pixels, skipped")
            continue
        log.info(f"{tag}: test image {h} x {w}, {n_pix:,} valid pixels ({100 * valid.mean():.2f}%), "
                 f"loaded in {load_stats['load']['seconds']:.1f} s")

        sample_dirs = {k: dirs[k] / tag for k in ["reduced_data", "reconstructed_data"]}
        ensure_dirs(*sample_dirs.values())
        orig_path = sample_dirs["reconstructed_data"] / "original_normalized.npy"
        np.save(orig_path, img.astype(np.float32))
        np.save(sample_dirs["reconstructed_data"] / "valid_mask.npy", valid)
        original_disk = file_size_bytes(orig_path)

        z_all = scaler.transform(x)
        gallery, reduced_full = {}, None
        for k in ks:
            # Millisecond timings are noisy, so each step is repeated and the median kept.
            runs = []
            for _ in range(cfg.TIMING_REPEATS):
                st = {}
                with measure(st, "transform"):
                    z = model.transform(z_all, k)                       # forward: C -> K
                with measure(st, "inverse"):
                    rec = scaler.inverse_transform(model.inverse_transform(z))   # inverse: K -> C
                runs.append(st)
            st = {step: {key: float(np.median([r[step][key] for r in runs])) for key in runs[0][step]}
                  for step in ("transform", "inverse")}
            reduced_img = matrix_to_image(z.astype(np.float32), valid)        # (H, W, K)
            rec_img = matrix_to_image(rec.astype(np.float32), valid)          # (H, W, C)
            rec_img_clip = np.clip(rec_img, 0, 1)        # values must stay in the valid [0, 1] range

            red_path = sample_dirs["reduced_data"] / f"reduced_K{k}.npy"
            np.save(red_path, reduced_img)
            if k in save_k:
                np.save(sample_dirs["reconstructed_data"] / f"reconstructed_K{k}.npy", rec_img_clip)
            if k == C:
                reduced_full = reduced_img
            if k in gallery_k:
                gallery[k] = rec_img_clip

            m = all_metrics(img, rec_img_clip, valid, names, groups)
            sto = storage_report(n_pix, C, k, model.n_model_values(k) + n_model_extra)
            rec_row = {"date": date, "sample": tag, "k": k, "n_channels": C, "n_pixels": n_pix,
                       "dimensionality_reduction_pct": dimensionality_reduction_pct(C, k),
                       "cumulative_explained_variance_train": float(np.cumsum(model.all_explained_variance_ratio_)[k - 1]),
                       **m["overall"], **sto,
                       "original_disk_bytes_npy": original_disk,
                       "reduced_disk_bytes_npy": file_size_bytes(red_path),
                       "transform_seconds": st["transform"]["seconds"],
                       "inverse_seconds": st["inverse"]["seconds"],
                       "transform_peak_mb": st["transform"]["peak_python_mb"],
                       "inverse_peak_mb": st["inverse"]["peak_python_mb"],
                       "process_rss_mb": rss_mb()}
            records.append(rec_row)
            for g, gm in m["per_group"].items():
                group_records.append({"date": date, "sample": tag, "k": k, "group": g, **gm})
            for cname, cm in m["per_channel"].items():
                per_channel_records.append({"date": date, "sample": tag, "k": k, "channel": cname, **cm})
            log.info(f"  K={k}: reduction {rec_row['dimensionality_reduction_pct']:5.1f}% | "
                     f"RMSE {m['overall']['RMSE']:.4f} MAE {m['overall']['MAE']:.4f} "
                     f"PSNR {m['overall']['PSNR_dB']:6.2f} dB SSIM {m['overall']['SSIM']:.4f} | "
                     f"{human_bytes(sto['original_bytes'])} -> {human_bytes(sto['reduced_plus_model_bytes'])} | "
                     f"fwd {1000 * st['transform']['seconds']:.1f} ms inv {1000 * st['inverse']['seconds']:.1f} ms")

        # per-sample pictures
        viz.plot_reconstruction_gallery(img, gallery, valid, names,
                                        dirs["graphs"] / f"reconstruction_{tag}.png",
                                        f"PCA reconstruction, {date} ({exp}, test region)")
        viz.plot_component_images(reduced_full, valid, dirs["graphs"] / f"components_{tag}.png",
                                  f"Principal-component images, {date} ({exp})")

    if not records:
        log.error("No test sample produced results.")
        sys.exit(2)

    # ------------------------------------------------------------- save tables
    df = pd.DataFrame(records)
    gdf = pd.DataFrame(group_records)
    cdf = pd.DataFrame(per_channel_records)
    df["lossless_to_float_precision"] = df["MSE"] < 1e-20
    df.to_csv(dirs["metrics"] / "test_metrics.csv", index=False)
    gdf.to_csv(dirs["metrics"] / "test_metrics_per_product.csv", index=False)
    cdf.to_csv(dirs["metrics"] / "test_metrics_per_channel.csv", index=False)

    # ------------------------------------------------------------- graphs comparing K
    # A perfect reconstruction has MSE ~1e-30 (float round-off) and PSNR > 300 dB, which would flatten
    # the plot. The CSV keeps the true value; graphs cap PSNR at PSNR_PLOT_CAP dB.
    PSNR_PLOT_CAP = 100.0
    df["PSNR_plot"] = df["PSNR_dB"].clip(upper=PSNR_PLOT_CAP)
    viz.plot_metric_panels(df, [("PSNR_plot", f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f}) - higher is better"),
                                ("SSIM", "SSIM - higher is better"),
                                ("RMSE", "RMSE (0-1 scale) - lower is better"),
                                ("MAE", "MAE (0-1 scale) - lower is better"),
                                ("MSE", "MSE (0-1 scale) - lower is better"),
                                ("cumulative_explained_variance_train", "Cumulative explained variance")],
                           dirs["graphs"] / "quality_vs_components.png",
                           f"Reconstruction quality vs retained components ({exp})")
    groups_order = list(groups)
    first = gdf[gdf["sample"] == gdf["sample"].iloc[0]]
    viz.plot_group_lines(first, "RMSE", groups_order, dirs["graphs"] / "rmse_per_product.png",
                         f"RMSE per product vs K ({first['date'].iloc[0]})", "RMSE (0-1 scale)")
    viz.plot_group_lines(first, "SSIM", groups_order, dirs["graphs"] / "ssim_per_product.png",
                         f"SSIM per product vs K ({first['date'].iloc[0]})", "SSIM")
    df_mb = df.assign(original_MB=df.original_bytes / 1e6, reduced_MB=df.reduced_plus_model_bytes / 1e6,
                      transform_ms=1000 * df.transform_seconds, inverse_ms=1000 * df.inverse_seconds)
    viz.plot_metric_panels(df_mb, [("reduced_MB", "Reduced data + model (MB, float32)"),
                                   ("compression_ratio", "Compression ratio (original / reduced)"),
                                   ("dimensionality_reduction_pct", "Dimensionality reduction (%)"),
                                   ("transform_ms", "Forward transform time (ms)"),
                                   ("inverse_ms", "Inverse transform time (ms)"),
                                   ("transform_peak_mb", "Forward transform peak memory (MB)")],
                           dirs["graphs"] / "storage_time_memory_vs_components.png",
                           f"Storage, time and memory vs retained components ({exp})")
    viz.plot_tradeoff(df, "storage_saving_pct", "PSNR_plot", dirs["graphs"] / "psnr_vs_storage_saving.png",
                      f"Quality vs storage saving ({exp})", "Storage saving incl. model (%)", f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f})")

    summary = {"experiment": exp, "components_tested": ks,
               "per_k_mean_over_test_samples": df.groupby("k")[["RMSE", "MAE", "PSNR_dB", "SSIM",
                                                               "dimensionality_reduction_pct",
                                                               "storage_saving_pct"]].mean().reset_index()
                                                   .to_dict(orient="records"),
               "process_rss_mb_end": rss_mb(), "total_seconds": time.perf_counter() - t_start}
    save_json(summary, dirs["metrics"] / "test_summary.json")
    log.info(f"Testing finished in {summary['total_seconds']:.1f} s. Metrics: {dirs['metrics']}")


if __name__ == "__main__":
    main()
