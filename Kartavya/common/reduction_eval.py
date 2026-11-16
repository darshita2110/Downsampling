"""
Test-time evaluation shared by every reconstruction-based algorithm (PCA, ICA, GA, PSO, ACO, SA, UBS-CNN).

This is the loop that used to be copied into every test.py. For every test sample and every K it
  1. reduces the pixels (C -> K) and saves the reduced K-channel image
  2. rebuilds all C channels (K -> C) and saves the reconstruction
  3. computes MSE, RMSE, MAE, PSNR, SSIM (overall, per channel group, per channel), dimensionality reduction %,
     storage (theoretical and actual .npy size), time and memory (median of TIMING_REPEATS runs)
and then draws the standard graphs of the original repository (quality vs K, RMSE/SSIM per group,
storage/time/memory vs K, PSNR vs storage saving, reconstruction gallery, reduced-channel images).
Algorithm-specific numbers are added through `row_extra` / `side` callbacks, so nothing of the original
per-algorithm output is lost.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from common import visualization as viz
from common.data_loader import channel_groups, rows_slice, sample_tag
from common.evaluation import dimensionality_reduction_pct, storage_report
from common.metrics import all_metrics
from common.preprocessing import image_to_matrix, matrix_to_image
from common.utils import ensure_dirs, file_size_bytes, human_bytes, rss_mb, save_json, timed

PSNR_PLOT_CAP = 100.0   # a perfect reconstruction has PSNR > 300 dB; graphs cap it, CSVs keep the true value


@dataclass
class EvalSpec:
    ks: list
    forward: Callable                      # (k, x) -> z          x: (N, C) normalised pixels
    inverse: Callable                      # (k, z) -> x_hat      (N, C)
    n_model_values: Callable               # k -> floats the receiver needs besides the reduced data
    kind: str = "components"               # "components" (new axes) or "features" (kept channels)
    method_label: str = "PCA"
    reduced_dtype: str = "float32"
    gallery_k: list = field(default_factory=lambda: [1, 2, 3, 4, 6])
    timing_repeats: int = 5
    save_reconstructed: object = None      # None = all K
    row_extra: Callable = None             # (k, x, z, rec) -> dict of extra columns
    side: Callable = None                  # (k, sample_info, x, z) -> dict appended to `side_records`
    per_sample: Callable = None            # (tag, sample_name, img, valid) -> None (e.g. CNN forward pass)
    reduced_image_labels: Callable = None  # k -> labels of the reduced channels (kept band names)
    native_selection: Callable = None      # k -> channel indices; also save the kept bands in native dtype
    quality_extra: tuple = None            # (column, panel title) added to the quality graph
    lossless_tol: float = 1e-20
    side_records: list = field(default_factory=list)


def evaluate(ctx, spec):
    """Run the evaluation; returns (per-sample/K table, per-group table, per-channel table)."""
    log, dirs, ds = ctx.log, ctx.dirs, ctx.dataset
    names, C = ctx.names, ctx.C
    groups = channel_groups(names)
    ks = list(spec.ks)
    save_k = set(spec.save_reconstructed or ks)
    gallery_k = [k for k in spec.gallery_k if k in ks] or ks[:min(4, len(ks))]
    bpv = np.dtype(spec.reduced_dtype).itemsize
    records, group_records, channel_records = [], [], []
    for entry in ctx.split["test"]:
        sample = ds.samples[entry["index"]]
        tag = sample_tag(ds, entry)
        img_full, valid_full = ds.load_normalized(entry["index"])
        rs = rows_slice(valid_full.shape[0], entry["rows"], ctx.fraction)
        img, valid = img_full[rs], valid_full[rs]
        x = image_to_matrix(img, valid)
        n_pix = x.shape[0]
        if n_pix == 0:
            log.warning(f"{tag}: no valid pixels, skipped")
            continue
        log.info(f"{tag}: test image {valid.shape[0]} x {valid.shape[1]}, {n_pix:,} valid pixels "
                 f"({100 * valid.mean():.2f}%)")
        sdirs = {c: dirs[c] / tag for c in ["reduced_data", "reconstructed_data"]}
        ensure_dirs(*sdirs.values())
        orig_path = sdirs["reconstructed_data"] / "original_normalized.npy"
        np.save(orig_path, img.astype(np.float32))
        np.save(sdirs["reconstructed_data"] / "valid_mask.npy", valid)
        if spec.per_sample:
            spec.per_sample(tag, sample["name"], img, valid)
        native = None
        if spec.native_selection:
            native = ds.load_native(entry["index"])[0][rs]

        gallery, reduced_last = {}, None
        for k in ks:
            z, t_fwd = timed(lambda: spec.forward(k, x), spec.timing_repeats)          # C -> K
            rec, t_inv = timed(lambda: spec.inverse(k, z), spec.timing_repeats)        # K -> C
            reduced_img = matrix_to_image(np.asarray(z, dtype=spec.reduced_dtype), valid, dtype=spec.reduced_dtype)
            rec_img = np.clip(matrix_to_image(np.asarray(rec, dtype=np.float32), valid), 0, 1)
            red_path = sdirs["reduced_data"] / f"reduced_K{k}.npy"
            np.save(red_path, reduced_img)
            if native is not None:
                np.save(sdirs["reduced_data"] / f"reduced_native_dtype_K{k}.npy", native[..., spec.native_selection(k)])
            if k in save_k:
                np.save(sdirs["reconstructed_data"] / f"reconstructed_K{k}.npy", rec_img)
            if k in gallery_k:
                gallery[k] = rec_img
            reduced_last = (k, reduced_img)

            met = all_metrics(img, rec_img, valid, names, groups)
            sto = storage_report(n_pix, C, k, spec.n_model_values(k), bytes_per_value=bpv)
            row = {"sample_name": sample["name"], "sample": tag, "k": k, "n_channels": C, "n_pixels": n_pix,
                   "dimensionality_reduction_pct": dimensionality_reduction_pct(C, k), **met["overall"], **sto,
                   "original_native_bytes": int(n_pix * C * np.dtype(ds.dtype).itemsize),
                   "original_disk_bytes_npy": file_size_bytes(orig_path),
                   "reduced_disk_bytes_npy": file_size_bytes(red_path),
                   "forward_seconds": t_fwd["seconds"], "inverse_seconds": t_inv["seconds"],
                   "forward_peak_mb": t_fwd["peak_python_mb"], "inverse_peak_mb": t_inv["peak_python_mb"],
                   "process_rss_mb": rss_mb()}
            if spec.kind == "components":   # original column names of PCA/ICA
                row.update({"transform_seconds": t_fwd["seconds"], "inverse_transform_seconds": t_inv["seconds"],
                            "transform_peak_mb": t_fwd["peak_python_mb"]})
            else:                           # original column names of GA/PSO/ACO/SA/UBS-CNN
                row.update({"reduce_seconds": t_fwd["seconds"], "reconstruct_seconds": t_inv["seconds"],
                            "reduce_peak_mb": t_fwd["peak_python_mb"], "reconstruct_peak_mb": t_inv["peak_python_mb"]})
            if spec.row_extra:
                row.update(spec.row_extra(k, x, z, rec))
            if spec.side:
                spec.side_records.append({"sample_name": sample["name"], "sample": tag, "k": k, **spec.side(k, x, z)})
            records.append(row)
            for g, gm in met["per_group"].items():
                group_records.append({"sample_name": sample["name"], "sample": tag, "k": k, "group": g, **gm})
            kept = set(spec.native_selection(k)) if spec.native_selection else set()
            for ci, (cname, cm) in enumerate(met["per_channel"].items()):
                extra = {"kept": ci in kept} if spec.native_selection else {}
                channel_records.append({"sample_name": sample["name"], "sample": tag, "k": k, "channel": cname,
                                        **extra, **cm})
            log.info(f"  K={k}: reduction {row['dimensionality_reduction_pct']:5.1f}% | RMSE {met['overall']['RMSE']:.4f} "
                     f"MAE {met['overall']['MAE']:.4f} PSNR {met['overall']['PSNR_dB']:6.2f} dB "
                     f"SSIM {met['overall']['SSIM']:.4f} | {human_bytes(sto['original_bytes'])} -> "
                     f"{human_bytes(sto['reduced_plus_model_bytes'])} | fwd {1000 * t_fwd['seconds']:.1f} ms "
                     f"inv {1000 * t_inv['seconds']:.1f} ms")

        unit = "" if "selection" in spec.method_label else ("selection" if spec.kind == "features" else "reconstruction")
        viz.plot_reconstruction_gallery(img, gallery, valid, names, dirs["graphs"] / f"reconstruction_{tag}.png",
                                        f"{spec.method_label}{' ' + unit if unit else ''}, {sample['name']} (test region)")
        k_last, red = reduced_last
        labels = spec.reduced_image_labels(k_last) if spec.reduced_image_labels else None
        prefix = "IC" if spec.method_label.startswith("ICA") else ("PC" if spec.kind == "components" else "band ")
        viz.plot_component_images(red, valid, dirs["graphs"] / f"components_{tag}.png",
                                  f"{spec.method_label}: reduced data, K={k_last}, {sample['name']}",
                                  n_show=min(4, red.shape[-1]), prefix=prefix, labels=labels)

    if not records:
        raise RuntimeError("No test sample produced results (no valid pixels in the test region).")
    df = pd.DataFrame(records)
    df["lossless_to_float_precision"] = df["MSE"] < spec.lossless_tol
    gdf, cdf = pd.DataFrame(group_records), pd.DataFrame(channel_records)
    df.to_csv(dirs["metrics"] / "test_metrics.csv", index=False)
    gdf.to_csv(dirs["metrics"] / "test_metrics_per_product.csv", index=False)
    cdf.to_csv(dirs["metrics"] / "test_metrics_per_channel.csv", index=False)
    standard_graphs(ctx, df, gdf, spec, list(groups))
    return df, gdf, cdf


def standard_graphs(ctx, df, gdf, spec, groups):
    dirs = ctx.dirs
    suffix = "components" if spec.kind == "components" else "features"
    xl = "Retained components (K)" if spec.kind == "components" else "Selected channels (K)"
    dp = df.assign(PSNR_plot=df["PSNR_dB"].clip(upper=PSNR_PLOT_CAP))
    panels = [("PSNR_plot", f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f}) - higher is better"),
              ("SSIM", "SSIM - higher is better"), ("RMSE", "RMSE (0-1 scale) - lower is better"),
              ("MAE", "MAE (0-1 scale) - lower is better"), ("MSE", "MSE (0-1 scale) - lower is better")]
    if spec.quality_extra:
        panels.append(spec.quality_extra)
    viz.plot_metric_panels(dp, panels, dirs["graphs"] / f"quality_vs_{suffix}.png",
                           f"{spec.method_label}: reconstruction quality vs K", series_col="sample_name", x_label=xl)
    first = gdf[gdf["sample"] == gdf["sample"].iloc[0]]
    viz.plot_group_lines(first, "RMSE", groups, dirs["graphs"] / "rmse_per_product.png",
                         f"RMSE per channel group vs K ({first['sample_name'].iloc[0]})", "RMSE (0-1 scale)", x_label=xl)
    viz.plot_group_lines(first, "SSIM", groups, dirs["graphs"] / "ssim_per_product.png",
                         f"SSIM per channel group vs K ({first['sample_name'].iloc[0]})", "SSIM", x_label=xl)
    dm = df.assign(reduced_MB=df.reduced_plus_model_bytes / 1e6, forward_ms=1000 * df.forward_seconds,
                   inverse_ms=1000 * df.inverse_seconds)
    viz.plot_metric_panels(dm, [("reduced_MB", "Reduced data + model (MB)"),
                                ("compression_ratio", "Compression ratio (incl. model)"),
                                ("dimensionality_reduction_pct", "Dimensionality reduction (%)"),
                                ("forward_ms", "Forward (reduce) time (ms)"),
                                ("inverse_ms", "Inverse (reconstruct) time (ms)"),
                                ("forward_peak_mb", "Forward peak memory (MB)")],
                           dirs["graphs"] / f"storage_time_memory_vs_{suffix}.png",
                           f"{spec.method_label}: storage, time and memory vs K", series_col="sample_name", x_label=xl)
    viz.plot_tradeoff(dp, "storage_saving_pct", "PSNR_plot", dirs["graphs"] / "psnr_vs_storage_saving.png",
                      f"{spec.method_label}: quality vs storage saving", "Storage saving incl. model (%)",
                      f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f})", series_col="sample_name")


def write_test_summary(ctx, df, extra=None, t_start=None):
    import time
    cols = [c for c in ["RMSE", "MAE", "PSNR_dB", "SSIM", "dimensionality_reduction_pct", "storage_saving_pct",
                        "reduced_disk_bytes_npy"] if c in df]
    summary = {"algorithm": ctx.name, "k_tested": sorted(df["k"].unique().tolist()),
               "per_k_mean_over_test_samples": df.groupby("k")[cols].mean().reset_index().to_dict(orient="records"),
               "process_rss_mb_end": rss_mb(), **(extra or {})}
    if t_start is not None:
        summary["total_seconds"] = time.perf_counter() - t_start
    save_json(summary, ctx.dirs["metrics"] / "test_summary.json")
    return summary


def compare_with_pca(ctx, df, label):
    """The original '<method>_vs_pca' table + graph, if PCA results for the SAME dataset exist."""
    pca_csv = None
    for name in ("pca", "pca_nostd"):
        p = ctx.sibling(name) / "metrics" / "test_metrics.csv"
        if p.exists():
            pca_csv = p
            break
    if pca_csv is None:
        ctx.log.info(f"No PCA results for this dataset yet; run PCA to get the {label.upper()}-vs-PCA comparison.")
        return []
    pca = pd.read_csv(pca_csv)
    cmp = df.merge(pca[["sample", "k", "PSNR_dB", "SSIM", "RMSE"]], on=["sample", "k"], suffixes=(f"_{label}", "_pca"))
    if not len(cmp):
        return []
    cmp.to_csv(ctx.dirs["metrics"] / f"{label}_vs_pca.csv", index=False)
    c1 = cmp[cmp["sample"] == cmp["sample"].iloc[0]].copy()
    for s in [label, "pca"]:
        c1[f"PSNR_{s}_plot"] = c1[f"PSNR_dB_{s}"].clip(upper=PSNR_PLOT_CAP)
    viz.plot_paired_bars(c1, "k", [f"{label.upper()} (keep K original channels)", "PCA (K new components)"],
                         [(f"PSNR (dB, capped at {PSNR_PLOT_CAP:.0f})", [f"PSNR_{label}_plot", "PSNR_pca_plot"]),
                          ("SSIM", [f"SSIM_{label}", "SSIM_pca"]), ("RMSE (0-1 scale)", [f"RMSE_{label}", "RMSE_pca"])],
                         ctx.dirs["graphs"] / f"{label}_vs_pca.png",
                         f"{label.upper()} vs PCA at the same K ({c1['sample_name'].iloc[0]} test region)",
                         x_label="Channels / components kept (K)")
    return cmp[["sample", "k", f"PSNR_dB_{label}", "PSNR_dB_pca", f"SSIM_{label}", "SSIM_pca"]].to_dict(orient="records")


def compare_with_searches(ctx, df, label, others):
    """The original '<method>_vs_ga_pso_aco.csv' style table: same objective, different search."""
    found = {}
    for other in others:
        csv = ctx.sibling(other) / "metrics" / "test_metrics.csv"
        if csv.exists():
            o = pd.read_csv(csv)
            cols = [c for c in ["sample", "k", "selected", "train_fitness", "PSNR_dB", "SSIM"] if c in o]
            found[other] = o[cols].rename(columns={c: f"{c}_{other}" for c in cols if c not in ("sample", "k")})
    if not found:
        return None
    cols = [c for c in ["sample", "k", "selected", "train_fitness", "PSNR_dB", "SSIM"] if c in df]
    cg = df[cols].rename(columns={c: f"{c}_{label}" for c in cols if c not in ("sample", "k")})
    for other, o in found.items():
        cg = cg.merge(o, on=["sample", "k"])
        if f"train_fitness_{other}" in cg and f"train_fitness_{label}" in cg:
            cg[f"same_fitness_as_{other}"] = (cg[f"train_fitness_{label}"] - cg[f"train_fitness_{other}"]).abs() < 1e-9
            ctx.log.info(f"{label.upper()} vs {other.upper()}: same training objective at "
                         f"{int(cg[f'same_fitness_as_{other}'].sum())}/{len(cg)} K values")
    name = f"{label}_vs_{'_'.join(found)}.csv"
    cg.to_csv(ctx.dirs["metrics"] / name, index=False)
    return name
