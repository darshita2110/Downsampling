"""
Apply the TRAINED UBS-CNN (BS-Net-Conv) selection to the test samples and evaluate every K.
Nothing is trained, re-selected or re-fitted here: the kept channels and reconstruction weights come from the
.npz files, and the trained networks (.pt) are only run forward on test patches to report their attention
weights and own reconstruction error.

Run from the repository root (same options as train.py):
    python Kartavya/algorithms/combinatorial/ubscnn/test.py
    python Kartavya/algorithms/combinatorial/ubscnn/test.py --dataset raw

For every test sample and every K it
  1. keeps only the selected channels (the reduced data) and saves them as an image array
  2. rebuilds all C channels with the saved linear model
  3. computes MSE, RMSE, MAE, PSNR, SSIM (overall, per product, per channel), reduction %, storage
     (theoretical and actual .npy size), time and memory
  4. compares with PCA at the same K if PCA test results exist (Kartavya/outputs/metrics/pca/<exp>/)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd
import torch

from Kartavya.algorithms.combinatorial.ubscnn import config as cfg
from Kartavya.algorithms.combinatorial.ga.ga import LinearReconstructor
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.evaluation import dimensionality_reduction_pct, storage_report
from Kartavya.common.metrics import all_metrics
from Kartavya.algorithms.combinatorial.ubscnn.model import BSNetConv, band_weights, reconstruction_mse
from Kartavya.common.preprocessing import extract_patches, image_to_matrix, matrix_to_image
from Kartavya.common.utils import (ensure_dirs, file_size_bytes, get_logger, human_bytes, load_json,
                                   measure, rss_mb, save_json)

PSNR_PLOT_CAP = 100.0   # a perfect reconstruction has PSNR > 300 dB; graphs cap it, CSVs keep the true value


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UBS-CNN channel selection (no training)")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def load_model(path):
    d = np.load(path, allow_pickle=False)
    model = LinearReconstructor.from_state({"selected": d["selected"], "coef": d["coef"]})
    return model, [str(n) for n in d["channel_names"]], int(d["target_width"]), float(d["fitness"])


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
    exp = args.dataset + (f"_{args.tag}" if args.tag else "")
    dirs = cfg.output_dirs(exp)
    ensure_dirs(*dirs.values())
    log = get_logger(f"ubscnn.test.{exp}", dirs["logs"] / "test.log")
    t_start = time.perf_counter()

    files = sorted(dirs["models"].glob("ubscnn_K*.npz"), key=lambda p: int(p.stem.split("K")[1]))
    if not files:
        log.error(f"No models in {dirs['models']}. Run train.py with the same options first.")
        sys.exit(2)
    models, fitness = {}, {}
    for p in files:
        m, names, width, fit = load_model(p)
        models[len(m.selected)], fitness[len(m.selected)] = m, fit
    cfg.TARGET_WIDTH = width
    ks, C = sorted(models), len(names)
    split = load_json(dirs["models"] / "split.json")
    mapping = load_json(dirs["metrics"] / "filename_index_mapping.json")
    current = dl.discover_samples(args.dataset, cfg)
    if [s["files"] for s in current["samples"]] != [s["files"] for s in mapping["samples"]]:
        log.error("Files on disk differ from those used in training (filename -> index mapping changed). "
                  "Re-run train.py first.")
        sys.exit(2)
    groups = dl.channel_groups(names)
    gallery_k = [k for k in cfg.GALLERY_K if k in ks]
    log.info(f"Experiment '{exp}': loaded saved subsets for K = {ks}")
    for k in ks:
        log.info(f"  K={k}: keep {[names[i] for i in models[k].selected]} (training fitness {fitness[k]:.5f})")

    # trained networks: forward pass only (eval mode, no gradients, no optimiser)
    nets = {}
    for p in sorted(dirs["models"].glob("bsnet_conv_seed*.pt")):
        ck = torch.load(p, map_location="cpu")
        net = BSNetConv(ck["n_bands"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        nets[int(p.stem.split("seed")[1])] = net
    log.info(f"Loaded {len(nets)} trained BS-Net-Conv networks (seeds {sorted(nets)}); running them forward only")
    train_rank = pd.read_csv(dirs["metrics"] / "channel_ranking.csv")
    cnn_records = []

    records, group_records, channel_records = [], [], []
    for entry in split["test"]:
        sample = mapping["samples"][entry["index"]]
        date = sample["date"]
        tag = f"{entry['index']:02d}_{date}" + ("" if entry["rows"] == "all" else f"_{entry['rows']}")
        data = dl.load_normalized_sample(args.dataset, sample, cfg)
        rs = dl.rows_slice(data["valid"].shape[0], entry["rows"], split.get("spatial_train_fraction", 0.5))
        img, valid = data["channels"][rs], data["valid"][rs]
        x = image_to_matrix(img, valid).astype(np.float32)
        n_pix = x.shape[0]
        if n_pix == 0:
            log.warning(f"{tag}: no valid pixels, skipped")
            continue
        log.info(f"{tag}: test image {valid.shape[0]} x {valid.shape[1]}, {n_pix:,} valid pixels")
        sdirs = {c: dirs[c] / tag for c in ["reduced_data", "reconstructed_data"]}
        ensure_dirs(*sdirs.values())
        orig_path = sdirs["reconstructed_data"] / "original_normalized.npy"
        np.save(orig_path, img.astype(np.float32))
        np.save(sdirs["reconstructed_data"] / "valid_mask.npy", valid)

        # ---- the CNN on TEST patches: attention weights and its own reconstruction error
        tp, _ = extract_patches(img, valid, cfg.PATCH_SIZE, cfg.PATCH_STRIDE)
        tp_t = torch.from_numpy(tp)
        test_w = {}
        for seed, net in nets.items():
            t_st = {}
            with measure(t_st, "fwd"):
                w = band_weights(net, tp_t)
            test_w[seed] = w.mean(0)
            cnn_records.append({"date": date, "sample": tag, "seed": seed, "n_test_patches": len(tp),
                                "network_reconstruction_mse": reconstruction_mse(net, tp_t),
                                "forward_seconds": t_st["fwd"]["seconds"],
                                **{f"w_{n}": float(v) for n, v in zip(names, test_w[seed])}})
        if test_w:
            mw = np.mean(list(test_w.values()), axis=0)
            test_rank = [names[i] for i in np.argsort(-mw)]
            log.info(f"  CNN on {len(tp):,} test patches: network reconstruction MSE "
                     f"{np.mean([r['network_reconstruction_mse'] for r in cnn_records if r['sample'] == tag]):.5f}; "
                     f"test-region ranking {test_rank} (training ranking {train_rank.channel.tolist()})")
            cnn_records[-1]["test_ranking_mean_of_seeds"] = " > ".join(test_rank)

        gallery = {}
        for k in ks:
            m = models[k]
            r, t_red = timed(lambda: m.reduce(x), cfg.TIMING_REPEATS)                      # C -> K
            rec, t_rec = timed(lambda: m.reconstruct_from_reduced(r), cfg.TIMING_REPEATS)  # K -> C
            reduced_img = matrix_to_image(r.astype(cfg.REDUCED_DTYPE), valid)
            rec_img = np.clip(matrix_to_image(rec, valid), 0, 1)
            red_path = sdirs["reduced_data"] / f"reduced_K{k}.npy"
            np.save(red_path, reduced_img)
            np.save(sdirs["reconstructed_data"] / f"reconstructed_K{k}.npy", rec_img)
            if k in gallery_k:
                gallery[k] = rec_img
            # kept channels must come back exactly (sanity check of the reconstruction model)
            kept_err = float(np.abs(rec[:, m.selected] - x[:, m.selected]).max())

            met = all_metrics(img, rec_img, valid, names, groups)
            sto = storage_report(n_pix, C, k, m.n_model_values(),
                                 bytes_per_value=np.dtype(cfg.REDUCED_DTYPE).itemsize)
            row = {"date": date, "sample": tag, "k": k, "n_channels": C, "n_pixels": n_pix,
                   "selected": "+".join(names[i] for i in m.selected), "train_fitness": fitness[k],
                   "dimensionality_reduction_pct": dimensionality_reduction_pct(C, k), **met["overall"], **sto,
                   "original_disk_bytes_npy": file_size_bytes(orig_path),
                   "reduced_disk_bytes_npy": file_size_bytes(red_path),
                   "reduce_seconds": t_red["seconds"], "reconstruct_seconds": t_rec["seconds"],
                   "reduce_peak_mb": t_red["peak_python_mb"], "reconstruct_peak_mb": t_rec["peak_python_mb"],
                   "process_rss_mb": rss_mb(), "max_abs_error_on_kept_channels": kept_err}
            records.append(row)
            for g, gm in met["per_group"].items():
                group_records.append({"date": date, "sample": tag, "k": k, "group": g, **gm})
            for cname, cm in met["per_channel"].items():
                channel_records.append({"date": date, "sample": tag, "k": k, "channel": cname,
                                        "kept": cname in row["selected"].split("+"), **cm})
            log.info(f"  K={k}: reduction {row['dimensionality_reduction_pct']:5.1f}% | RMSE {met['overall']['RMSE']:.4f} "
                     f"MAE {met['overall']['MAE']:.4f} PSNR {met['overall']['PSNR_dB']:6.2f} dB "
                     f"SSIM {met['overall']['SSIM']:.4f} | {human_bytes(sto['original_bytes'])} -> "
                     f"{human_bytes(sto['reduced_plus_model_bytes'])} (file {human_bytes(row['reduced_disk_bytes_npy'])}) | "
                     f"reduce {1000 * t_red['seconds']:.1f} ms, rebuild {1000 * t_rec['seconds']:.1f} ms | "
                     f"kept-channel error {kept_err:.1e}")

        viz.plot_reconstruction_gallery(img, gallery, valid, names, dirs["graphs"] / f"reconstruction_{tag}.png",
                                        f"UBS-CNN (BS-Net-Conv) selection: reconstruction, {date} ({exp}, test region)")

    if not records:
        log.error("No test sample produced results.")
        sys.exit(2)

    # ------------------------------------------------------------- tables
    df = pd.DataFrame(records)
    df["lossless_to_float_precision"] = df["MSE"] < 1e-12
    gdf, cdf = pd.DataFrame(group_records), pd.DataFrame(channel_records)
    df.to_csv(dirs["metrics"] / "test_metrics.csv", index=False)
    pd.DataFrame(cnn_records).to_csv(dirs["metrics"] / "test_network_outputs.csv", index=False)
    gdf.to_csv(dirs["metrics"] / "test_metrics_per_product.csv", index=False)
    cdf.to_csv(dirs["metrics"] / "test_metrics_per_channel.csv", index=False)

    # ------------------------------------------------------------- graphs comparing K
    df_plot = df.assign(PSNR_plot=df["PSNR_dB"].clip(upper=PSNR_PLOT_CAP))
    viz.plot_metric_panels(df_plot, [("PSNR_plot", f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f})"),
                                     ("SSIM", "SSIM - higher is better"),
                                     ("RMSE", "RMSE (0-1 scale) - lower is better"),
                                     ("MAE", "MAE (0-1 scale) - lower is better"),
                                     ("MSE", "MSE (0-1 scale) - lower is better"),
                                     ("train_fitness", "Objective (training, evaluation only)")],
                           dirs["graphs"] / "quality_vs_features.png",
                           f"UBS-CNN selection: quality vs selected channels ({exp})",
                           x_label="Selected channels (K)")
    first = gdf[gdf["sample"] == gdf["sample"].iloc[0]]
    viz.plot_group_lines(first, "RMSE", list(groups), dirs["graphs"] / "rmse_per_product.png",
                         f"RMSE per product vs K ({first['date'].iloc[0]})", "RMSE (0-1 scale)",
                         x_label="Selected channels (K)")
    viz.plot_group_lines(first, "SSIM", list(groups), dirs["graphs"] / "ssim_per_product.png",
                         f"SSIM per product vs K ({first['date'].iloc[0]})", "SSIM",
                         x_label="Selected channels (K)")
    df_mb = df.assign(reduced_MB=df.reduced_disk_bytes_npy / 1e6, reduce_ms=1000 * df.reduce_seconds,
                      reconstruct_ms=1000 * df.reconstruct_seconds)
    viz.plot_metric_panels(df_mb, [("reduced_MB", "Reduced data on disk (MB, .npy)"),
                                   ("compression_ratio", "Compression ratio (incl. model)"),
                                   ("dimensionality_reduction_pct", "Dimensionality reduction (%)"),
                                   ("reduce_ms", "Reduce time (ms)"),
                                   ("reconstruct_ms", "Reconstruct time (ms)"),
                                   ("reconstruct_peak_mb", "Reconstruct peak memory (MB)")],
                           dirs["graphs"] / "storage_time_memory_vs_features.png",
                           f"Storage, time and memory vs selected channels ({exp})",
                           x_label="Selected channels (K)")
    viz.plot_tradeoff(df_plot, "storage_saving_pct", "PSNR_plot", dirs["graphs"] / "psnr_vs_storage_saving.png",
                      f"Quality vs storage saving ({exp})", "Storage saving incl. model (%)",
                      f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f})")

    # ------------------------------------------------------------- SA vs PCA at the same K (if available)
    comparison = []
    pca_csv = cfg.OUTPUT_ROOT / "metrics" / "pca" / exp / "test_metrics.csv"
    if pca_csv.exists():
        pca = pd.read_csv(pca_csv)
        cmp = df.merge(pca[["sample", "k", "PSNR_dB", "SSIM", "RMSE"]], on=["sample", "k"],
                       suffixes=("_ubscnn", "_pca"))
        if len(cmp):
            cmp.to_csv(dirs["metrics"] / "ubscnn_vs_pca.csv", index=False)
            c1 = cmp[cmp["sample"] == cmp["sample"].iloc[0]].copy()
            for s in ["ubscnn", "pca"]:
                c1[f"PSNR_{s}_plot"] = c1[f"PSNR_dB_{s}"].clip(upper=PSNR_PLOT_CAP)
            viz.plot_paired_bars(c1, "k", ["UBS-CNN (keep K original channels)", "PCA (K new components)"],
                                 [(f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f})", ["PSNR_ubscnn_plot", "PSNR_pca_plot"]),
                                  ("SSIM", ["SSIM_ubscnn", "SSIM_pca"]), ("RMSE (0-1 scale)", ["RMSE_ubscnn", "RMSE_pca"])],
                                 dirs["graphs"] / "ubscnn_vs_pca.png",
                                 f"UBS-CNN selection vs PCA at the same K ({c1['date'].iloc[0]} test region)",
                                 x_label="Channels / components kept (K)")
            comparison = cmp[["sample", "k", "PSNR_dB_ubscnn", "PSNR_dB_pca", "SSIM_ubscnn", "SSIM_pca",
                              "RMSE_ubscnn", "RMSE_pca"]].to_dict(orient="records")
            log.info("UBS-CNN vs PCA at the same K (PSNR dB / SSIM):")
            for r in comparison:
                log.info(f"  K={r['k']}: UBS-CNN {r['PSNR_dB_ubscnn']:.2f} / {r['SSIM_ubscnn']:.4f}  vs  "
                         f"PCA {r['PSNR_dB_pca']:.2f} / {r['SSIM_pca']:.4f}")
    else:
        log.info(f"No PCA results at {pca_csv}; run the PCA scripts to get the UBS-CNN-vs-PCA comparison.")

    # ------------------------------------------------------------- UBS-CNN vs the search methods (GA, PSO, ACO, SA)
    others = {}
    for other in ["ga", "pso", "aco", "sa"]:
        csv = cfg.OUTPUT_ROOT / "metrics" / other / exp / "test_metrics.csv"
        if csv.exists():
            o = pd.read_csv(csv)[["sample", "k", "selected", "train_fitness", "PSNR_dB", "SSIM"]]
            others[other] = o.rename(columns={c: f"{c}_{other}" for c in ["selected", "train_fitness", "PSNR_dB", "SSIM"]})
    if others:
        cg = df[["sample", "k", "selected", "train_fitness", "PSNR_dB", "SSIM"]].rename(
            columns={c: f"{c}_ubscnn" for c in ["selected", "train_fitness", "PSNR_dB", "SSIM"]})
        for other, o in others.items():
            cg = cg.merge(o, on=["sample", "k"])
            cg[f"same_fitness_as_{other}"] = (cg.train_fitness_ubscnn - cg[f"train_fitness_{other}"]).abs() < 1e-9
            log.info(f"UBS-CNN vs {other.upper()}: same training objective at "
                     f"{int(cg[f'same_fitness_as_{other}'].sum())}/{len(cg)} K values")
        cg.to_csv(dirs["metrics"] / "ubscnn_vs_search_methods.csv", index=False)

    summary = {"experiment": exp, "features_tested": ks,
               "selected": {int(k): [names[i] for i in models[k].selected] for k in ks},
               "per_k_mean_over_test_samples": df.groupby("k")[["RMSE", "MAE", "PSNR_dB", "SSIM",
                                                               "dimensionality_reduction_pct", "storage_saving_pct",
                                                               "reduced_disk_bytes_npy"]].mean()
                                                   .reset_index().to_dict(orient="records"),
               "ubscnn_vs_pca": comparison, "process_rss_mb_end": rss_mb(),
               "total_seconds": time.perf_counter() - t_start}
    save_json(summary, dirs["metrics"] / "test_summary.json")
    log.info(f"Testing finished in {summary['total_seconds']:.1f} s. Metrics: {dirs['metrics']}")


if __name__ == "__main__":
    main()
