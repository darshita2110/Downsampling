"""
Apply the SAVED ICA models (from train.py) to the test samples and evaluate every K.
Nothing is re-fitted here: the standardiser and the unmixing/mixing matrices come from the .npz files.

Run from the repository root (same options as train.py):
    python Kartavya/algorithms/linear/ica/test.py
    python Kartavya/algorithms/linear/ica/test.py --no-standardize
    python Kartavya/algorithms/linear/ica/test.py --dataset raw

For every test sample and every trained K it
  1. unmixes the pixels into K independent sources and reshapes them into a K-channel image
  2. reconstructs the C channels with the mixing matrix and reshapes into an image
  3. computes MSE, RMSE, MAE, PSNR, SSIM (overall, per channel, per product), reduction %, storage,
     time and memory
  4. measures how independent the components are (ICA sources vs the same-K PCA/whitened axes)
  5. checks that the reconstruction equals the K-dimensional PCA projection (theory says it must)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd

from Kartavya.algorithms.linear.ica import config as cfg
from Kartavya.algorithms.linear.ica.ica import FastICA
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.evaluation import dimensionality_reduction_pct, storage_report
from Kartavya.common.metrics import all_metrics, component_statistics
from Kartavya.common.preprocessing import Standardizer, image_to_matrix, matrix_to_image
from Kartavya.common.utils import (ensure_dirs, file_size_bytes, get_logger, human_bytes, load_json,
                                   measure, rss_mb, save_json)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ICA channel reduction with saved models")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def load_model(path):
    """Read one saved model: (FastICA, Standardizer, channel names, grid width)."""
    d = np.load(path, allow_pickle=False)
    model = FastICA.from_state({k[4:]: d[k] for k in d.files if k.startswith("ica_")})
    scaler = Standardizer.from_state({"enabled": bool(d["scaler_enabled"]), "mean": d["scaler_mean"],
                                      "std": d["scaler_std"]})
    return model, scaler, [str(n) for n in d["channel_names"]], int(d["target_width"])


def timed(fn, repeats):
    """Run fn `repeats` times under measure(); return the last result and the median statistics."""
    runs, out = [], None
    for _ in range(repeats):
        st = {}
        with measure(st, "x"):
            out = fn()
        runs.append(st["x"])
    return out, {key: float(np.median([r[key] for r in runs])) for key in runs[0]}


def main():
    args = parse_args()
    exp = args.dataset + ("_nostd" if args.no_standardize else "") + (f"_{args.tag}" if args.tag else "")
    dirs = cfg.output_dirs(exp)
    ensure_dirs(*dirs.values())
    log = get_logger(f"ica.test.{exp}", dirs["logs"] / "test.log")
    t_start = time.perf_counter()

    model_files = sorted(dirs["models"].glob("ica_K*.npz"), key=lambda p: int(p.stem.split("K")[1]))
    if not model_files:
        log.error(f"No models in {dirs['models']}. Run train.py with the same options first.")
        sys.exit(2)
    models = {}
    for p in model_files:
        m, scaler, names, width = load_model(p)
        models[m.n_components] = m
    cfg.TARGET_WIDTH = width                                   # same grid as training
    ks = sorted(models)
    C = len(names)
    split = load_json(dirs["models"] / "split.json")
    mapping = load_json(dirs["metrics"] / "filename_index_mapping.json")
    current = dl.discover_samples(args.dataset, cfg)
    if [s["files"] for s in current["samples"]] != [s["files"] for s in mapping["samples"]]:
        log.error("Files on disk differ from those used in training (filename -> index mapping changed). "
                  "Re-run train.py first.")
        sys.exit(2)
    groups = dl.channel_groups(names)
    save_k = set(cfg.SAVE_RECONSTRUCTED_K or ks)
    gallery_k = [k for k in cfg.GALLERY_K if k in ks]
    n_scaler_values = 2 * C if scaler.enabled else 0
    log.info(f"Experiment '{exp}': loaded saved models for K = {ks}; standardiser enabled={scaler.enabled}")
    for k in ks:
        m = models[k]
        log.info(f"  K={k}: trained with {m.n_iter_} iterations, converged={m.converged_}, tol={m.tol:g}, "
                 f"alpha={m.fun_alpha}, seed={m.random_state}")

    records, group_records, channel_records, indep_records = [], [], [], []
    for entry in split["test"]:
        sample = mapping["samples"][entry["index"]]
        date = sample["date"]
        tag = f"{entry['index']:02d}_{date}" + ("" if entry["rows"] == "all" else f"_{entry['rows']}")
        load_stats = {}
        with measure(load_stats, "load"):
            data = dl.load_normalized_sample(args.dataset, sample, cfg)
        rs = dl.rows_slice(data["valid"].shape[0], entry["rows"], split.get("spatial_train_fraction", 0.5))
        img, valid = data["channels"][rs], data["valid"][rs]
        x = image_to_matrix(img, valid)
        n_pix = x.shape[0]
        if n_pix == 0:
            log.warning(f"{tag}: no valid pixels, skipped")
            continue
        log.info(f"{tag}: test image {valid.shape[0]} x {valid.shape[1]}, {n_pix:,} valid pixels, "
                 f"loaded in {load_stats['load']['seconds']:.1f} s")
        sdirs = {c: dirs[c] / tag for c in ["reduced_data", "reconstructed_data"]}
        ensure_dirs(*sdirs.values())
        orig_path = sdirs["reconstructed_data"] / "original_normalized.npy"
        np.save(orig_path, img.astype(np.float32))
        np.save(sdirs["reconstructed_data"] / "valid_mask.npy", valid)

        z = scaler.transform(x)                                    # TRAINING mean/std
        gallery, sources_max = {}, None
        for k in ks:
            m = models[k]
            s, t_fwd = timed(lambda: m.transform(z), cfg.TIMING_REPEATS)                  # C -> K
            rec, t_inv = timed(lambda: scaler.inverse_transform(m.inverse_transform(s)),
                               cfg.TIMING_REPEATS)                                         # K -> C
            reduced_img = matrix_to_image(s.astype(np.float32), valid)
            rec_img = np.clip(matrix_to_image(rec.astype(np.float32), valid), 0, 1)
            red_path = sdirs["reduced_data"] / f"reduced_K{k}.npy"
            np.save(red_path, reduced_img)
            if k in save_k:
                np.save(sdirs["reconstructed_data"] / f"reconstructed_K{k}.npy", rec_img)
            if k in gallery_k:
                gallery[k] = rec_img
            if k == ks[-1]:
                sources_max = reduced_img

            # ICA's rotation must not change the reconstruction: compare with the whitened (PCA) projection
            pca_rec = scaler.inverse_transform(m.project_onto_whitened_subspace(z))
            same_as_pca = float(np.abs(pca_rec - rec).max())

            # independence: ICA sources vs the PCA axes of the same subspace (whitened)
            pcs = (z - m.mean_) @ m.whitening_.T
            ica_stats, pca_stats = component_statistics(s), component_statistics(pcs)
            indep_records.append({"date": date, "sample": tag, "k": k,
                                  "ica_mean_abs_kurtosis": ica_stats["mean_abs_excess_kurtosis"],
                                  "pca_mean_abs_kurtosis": pca_stats["mean_abs_excess_kurtosis"],
                                  "ica_linear_corr": ica_stats["mean_abs_linear_corr"],
                                  "pca_linear_corr": pca_stats["mean_abs_linear_corr"],
                                  "ica_square_corr": ica_stats["mean_abs_square_corr"],
                                  "pca_square_corr": pca_stats["mean_abs_square_corr"],
                                  "ica_kurtosis_per_component": str(np.round(ica_stats["excess_kurtosis"], 3).tolist())})

            met = all_metrics(img, rec_img, valid, names, groups)
            sto = storage_report(n_pix, C, k, m.n_model_values() + n_scaler_values)
            row = {"date": date, "sample": tag, "k": k, "n_channels": C, "n_pixels": n_pix,
                   "dimensionality_reduction_pct": dimensionality_reduction_pct(C, k), **met["overall"], **sto,
                   "original_disk_bytes_npy": file_size_bytes(orig_path),
                   "reduced_disk_bytes_npy": file_size_bytes(red_path),
                   "transform_seconds": t_fwd["seconds"], "inverse_seconds": t_inv["seconds"],
                   "transform_peak_mb": t_fwd["peak_python_mb"], "inverse_peak_mb": t_inv["peak_python_mb"],
                   "process_rss_mb": rss_mb(), "max_abs_diff_vs_pca_projection": same_as_pca,
                   "train_n_iter": m.n_iter_, "train_converged": m.converged_}
            records.append(row)
            for g, gm in met["per_group"].items():
                group_records.append({"date": date, "sample": tag, "k": k, "group": g, **gm})
            for cname, cm in met["per_channel"].items():
                channel_records.append({"date": date, "sample": tag, "k": k, "channel": cname, **cm})
            log.info(f"  K={k}: reduction {row['dimensionality_reduction_pct']:5.1f}% | RMSE {met['overall']['RMSE']:.4f} "
                     f"MAE {met['overall']['MAE']:.4f} PSNR {met['overall']['PSNR_dB']:6.2f} dB "
                     f"SSIM {met['overall']['SSIM']:.4f} | {human_bytes(sto['original_bytes'])} -> "
                     f"{human_bytes(sto['reduced_plus_model_bytes'])} | fwd {1000 * t_fwd['seconds']:.1f} ms "
                     f"inv {1000 * t_inv['seconds']:.1f} ms | |kurt| ICA {ica_stats['mean_abs_excess_kurtosis']:.2f} "
                     f"vs PCA {pca_stats['mean_abs_excess_kurtosis']:.2f} | sq-corr ICA "
                     f"{ica_stats['mean_abs_square_corr']:.3f} vs PCA {pca_stats['mean_abs_square_corr']:.3f} | "
                     f"vs PCA recon {same_as_pca:.1e}")

        viz.plot_reconstruction_gallery(img, gallery, valid, names, dirs["graphs"] / f"reconstruction_{tag}.png",
                                        f"ICA reconstruction, {date} ({exp}, test region)")
        viz.plot_component_images(sources_max, valid, dirs["graphs"] / f"components_{tag}.png",
                                  f"Independent-component images, K={ks[-1]}, {date} ({exp})",
                                  n_show=min(4, ks[-1]), prefix="IC")

    if not records:
        log.error("No test sample produced results.")
        sys.exit(2)

    # ------------------------------------------------------------- tables
    df = pd.DataFrame(records)
    df["lossless_to_float_precision"] = df["MSE"] < 1e-20
    # A perfect reconstruction has PSNR > 300 dB (float round-off), which would flatten the plots.
    # The CSV keeps the true value; graphs cap it.
    PSNR_PLOT_CAP = 100.0
    gdf, cdf, idf = pd.DataFrame(group_records), pd.DataFrame(channel_records), pd.DataFrame(indep_records)
    df.to_csv(dirs["metrics"] / "test_metrics.csv", index=False)
    gdf.to_csv(dirs["metrics"] / "test_metrics_per_product.csv", index=False)
    cdf.to_csv(dirs["metrics"] / "test_metrics_per_channel.csv", index=False)
    idf.to_csv(dirs["metrics"] / "test_independence.csv", index=False)

    # ------------------------------------------------------------- graphs comparing K
    df_plot = df.assign(PSNR_plot=df["PSNR_dB"].clip(upper=PSNR_PLOT_CAP))
    viz.plot_metric_panels(df_plot, [("PSNR_plot", f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f}) - higher is better"), ("SSIM", "SSIM - higher is better"),
                                ("RMSE", "RMSE (0-1 scale) - lower is better"),
                                ("MAE", "MAE (0-1 scale) - lower is better"),
                                ("MSE", "MSE (0-1 scale) - lower is better"),
                                ("train_n_iter", "FastICA iterations to converge (training)")],
                           dirs["graphs"] / "quality_vs_components.png",
                           f"ICA reconstruction quality vs retained components ({exp})")
    first = gdf[gdf["sample"] == gdf["sample"].iloc[0]]
    viz.plot_group_lines(first, "RMSE", list(groups), dirs["graphs"] / "rmse_per_product.png",
                         f"RMSE per product vs K ({first['date'].iloc[0]})", "RMSE (0-1 scale)")
    viz.plot_group_lines(first, "SSIM", list(groups), dirs["graphs"] / "ssim_per_product.png",
                         f"SSIM per product vs K ({first['date'].iloc[0]})", "SSIM")
    df_mb = df.assign(reduced_MB=df.reduced_plus_model_bytes / 1e6, transform_ms=1000 * df.transform_seconds,
                      inverse_ms=1000 * df.inverse_seconds)
    viz.plot_metric_panels(df_mb, [("reduced_MB", "Reduced data + model (MB, float32)"),
                                   ("compression_ratio", "Compression ratio (original / reduced)"),
                                   ("dimensionality_reduction_pct", "Dimensionality reduction (%)"),
                                   ("transform_ms", "Unmixing (forward) time (ms)"),
                                   ("inverse_ms", "Mixing (inverse) time (ms)"),
                                   ("transform_peak_mb", "Forward peak memory (MB)")],
                           dirs["graphs"] / "storage_time_memory_vs_components.png",
                           f"Storage, time and memory vs retained components ({exp})")
    viz.plot_tradeoff(df_plot, "storage_saving_pct", "PSNR_plot", dirs["graphs"] / "psnr_vs_storage_saving.png",
                      f"Quality vs storage saving ({exp})", "Storage saving incl. model (%)", f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f})")
    ifirst = idf[idf["sample"] == idf["sample"].iloc[0]]
    viz.plot_paired_bars(ifirst[ifirst.k > 1], "k", ["ICA sources", "PCA axes (same subspace)"],
                         [("Non-Gaussianity: mean |excess kurtosis|", ["ica_mean_abs_kurtosis", "pca_mean_abs_kurtosis"]),
                          ("Non-linear dependence: mean |corr(s_i^2, s_j^2)|", ["ica_square_corr", "pca_square_corr"])],
                         dirs["graphs"] / "independence_ica_vs_pca.png",
                         f"How independent are the components? ({exp}, {ifirst['date'].iloc[0]} test region)")

    summary = {"experiment": exp, "components_tested": ks,
               "per_k_mean_over_test_samples": df.groupby("k")[["RMSE", "MAE", "PSNR_dB", "SSIM",
                                                               "dimensionality_reduction_pct", "storage_saving_pct",
                                                               "max_abs_diff_vs_pca_projection"]].mean()
                                                   .reset_index().to_dict(orient="records"),
               "independence": idf.groupby("k")[["ica_mean_abs_kurtosis", "pca_mean_abs_kurtosis",
                                                 "ica_square_corr", "pca_square_corr"]].mean().reset_index()
                                                 .to_dict(orient="records"),
               "process_rss_mb_end": rss_mb(), "total_seconds": time.perf_counter() - t_start}
    save_json(summary, dirs["metrics"] / "test_summary.json")
    log.info(f"Testing finished in {summary['total_seconds']:.1f} s. Metrics: {dirs['metrics']}")


if __name__ == "__main__":
    main()
