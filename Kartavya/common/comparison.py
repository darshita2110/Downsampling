"""
Consolidated comparison of every algorithm run on one dataset (used by run_all_algorithms.py).

Fairness rules applied here (and guaranteed upstream by run_all_algorithms.py):
  * every algorithm used the same dataset, the same train/test split, the same [0, 1] normalisation
    (fitted once on training pixels) and the same K values
  * metric directions come from common/evaluation.METRICS - lower is better for errors and costs,
    higher for PSNR, SSIM, savings and classification scores
  * a metric that does not apply to an algorithm (e.g. reconstruction error for LDA, accuracy for PCA) is
    left empty (NaN / null) and never replaced by an artificial value
  * "overall best" averages a metric over the K values that EVERY algorithm with that metric evaluated, so an
    algorithm is never rewarded for being tested at an easier K (e.g. PCA's lossless K = C)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import visualization as viz
from common.evaluation import METRICS, RUN_METRICS, best_index
from common.utils import ensure_dirs, save_json


def _long_table(results):
    rows = []
    for r in results:
        pk = r.get("per_k")
        if pk is None or not len(pk):
            continue
        for _, row in pk.iterrows():
            rec = {"algorithm": r["display"], "algorithm_id": r["algorithm"], "family": r.get("family", ""),
                   "k": int(row["k"])}
            for m in METRICS:
                rec[m] = float(row[m]) if m in row and pd.notna(row[m]) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float) and np.isinf(v):
        return "inf"
    return f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v)


def build(results, dataset, split, ks, out_dir, logger=None):
    out_dir = Path(out_dir)
    mdir = out_dir / "metric_comparison"
    ensure_dirs(out_dir, mdir)
    long = _long_table(results)
    run_rows = []
    for r in results:
        run_rows.append({"algorithm": r["display"], "algorithm_id": r["algorithm"], "status": r["status"],
                         "reason_or_error": r.get("reason") or r.get("error_short") or "",
                         **{m: r.get(m, np.nan) for m in RUN_METRICS}, "output_dir": r.get("output_dir", "")})
    runs = pd.DataFrame(run_rows)
    runs.to_csv(out_dir / "execution_time_comparison.csv", index=False)

    # ---------------------------------------------------------------- consolidated CSV (one row per algorithm x K)
    if len(long):
        table = long.merge(runs[["algorithm", "status"] + list(RUN_METRICS)], on="algorithm", how="left")
    else:
        table = pd.DataFrame(columns=["algorithm", "k", *METRICS])
    no_rows = runs[~runs["algorithm"].isin(long["algorithm"] if len(long) else [])]
    if len(no_rows):                          # failed / not applicable algorithms still appear, with empty metrics
        table = pd.concat([table, no_rows[["algorithm", "algorithm_id", "status"] + list(RUN_METRICS)]],
                          ignore_index=True)
    table.to_csv(out_dir / "comparison.csv", index=False)

    # ---------------------------------------------------------------- per-metric tables, best per K and overall
    best_per_metric, per_k_winners, metric_files = {}, {}, {}
    present = [m for m in METRICS if len(long) and long[m].notna().any()]
    for m in present:
        info = METRICS[m]
        piv = long.pivot_table(index="k", columns="algorithm", values=m, aggfunc="mean", dropna=False)
        piv = piv[[a for a in dict.fromkeys(long["algorithm"]) if a in piv.columns]]
        winners = []
        for k, row in piv.iterrows():
            i = best_index(row.values, info["direction"])
            winners.append(piv.columns[i] if i is not None else None)
        piv["best"] = winners
        piv.to_csv(mdir / f"{m}_by_k.csv")
        per_k_winners[m] = {int(k): w for k, w in zip(piv.index, winners)}
        applicable = [a for a in piv.columns if a != "best" and piv[a].notna().any()]
        # Rank over the shared K list. An algorithm that cannot reach every shared K (e.g. LDA: at most
        # classes - 1 components, ICA: at most the data rank) is reported separately instead of shrinking
        # the comparison of all the others to its few K values.
        has = {a: {int(k) for k in piv.index if pd.notna(piv.loc[k, a])} for a in applicable}
        full = [a for a in applicable if set(ks) <= has[a]]
        if full:
            ranked, common_k = full, [k for k in piv.index if int(k) in set(ks)]
        else:
            ranked = applicable
            common_k = [k for k in piv.index if all(pd.notna(piv.loc[k, a]) for a in applicable)]
        partial = {a: sorted(has[a]) for a in applicable if a not in ranked}
        means = {a: float(piv.loc[common_k, a].mean()) if common_k else np.nan for a in ranked}
        partial_means = {a: float(piv[a].dropna().mean()) for a in partial}
        names = list(means)
        bi = best_index([means[a] for a in names], info["direction"]) if names else None
        ties = []
        if bi is not None:
            bv = means[names[bi]]
            ties = [a for a in names if np.isclose(means[a], bv, rtol=1e-9, atol=1e-12) or means[a] == bv]
        wins = pd.Series([w for w in winners if w]).value_counts().to_dict()
        note = []
        if len(applicable) == 1:
            note.append("only one algorithm provides this metric")
        if len(ties) > 1:
            note.append(f"tie between {', '.join(ties)}")
        if partial:
            note.append("ranked separately (cannot reach every shared K): " +
                        ", ".join(f"{a} K={v}" for a, v in partial.items()))
        best_per_metric[m] = {"label": info["label"], "direction": info["direction"], "family": info["family"],
                              "applicable_algorithms": applicable,
                              "not_applicable": [r["display"] for r in results if r["display"] not in applicable],
                              "common_k": [int(k) for k in common_k], "mean_over_common_k": means,
                              "partial_k_coverage": partial, "mean_over_own_k_partial": partial_means,
                              "best_algorithm": (" = ".join(ties) if len(ties) > 1 else names[bi]) if bi is not None else None,
                              "best_value": means[names[bi]] if bi is not None else None,
                              "wins_per_k": {str(a): int(n) for a, n in wins.items()}, "note": "; ".join(note)}
        sub = long[long[m].notna()]
        plot_vals = sub.copy()
        if m == "PSNR_dB":
            plot_vals[m] = plot_vals[m].clip(upper=100.0)
        metric_files[m] = str(viz.plot_algorithms_vs_k(plot_vals, m, info["label"], info["direction"],
                                                       mdir / f"{m}_vs_k.png", info["label"] + " vs K"))

    # ---------------------------------------------------------------- run-level comparisons
    ok = runs[runs["status"] == "ok"]
    best_run = {}
    for m, info in RUN_METRICS.items():
        vals = ok[m].astype(float).tolist()
        i = best_index(vals, info["direction"])
        best_run[m] = {"label": info["label"], "direction": info["direction"],
                       "best_algorithm": ok["algorithm"].iloc[i] if i is not None else None,
                       "best_value": vals[i] if i is not None else None,
                       "values": dict(zip(ok["algorithm"], vals))}
    if len(ok):
        _time_plot(ok, out_dir / "execution_time_comparison.png")
        viz.plot_horizontal_bars(ok["algorithm"].tolist(), ok["total_seconds"].astype(float).tolist(),
                                 out_dir / "total_time_comparison.png", "Total run time per algorithm (lower is better)",
                                 "seconds", highlight=best_run["total_seconds"]["best_algorithm"])
    if len(long) and long["PSNR_dB"].notna().any():
        _tradeoff_plot(long, out_dir / "quality_vs_storage_tradeoff.png")

    # ---------------------------------------------------------------- JSON + report
    payload = {"dataset": dataset.summary(), "split": split, "k_values": ks,
               "fairness": {"same_dataset": True, "same_split": True, "same_normalisation": dataset.normalizer.state(),
                            "same_k_values": ks, "evaluation": "test region only; models fitted on training data"},
               "metric_directions": {m: METRICS[m]["direction"] for m in METRICS},
               "algorithms": [{k: v for k, v in r.items() if k != "per_k"} |
                              {"per_k": r["per_k"].to_dict(orient="records") if r.get("per_k") is not None and len(r["per_k"]) else []}
                              for r in results],
               "best_per_metric": best_per_metric, "best_per_metric_per_k": per_k_winners,
               "run_level": best_run, "metric_graphs": metric_files}
    save_json(payload, out_dir / "comparison.json")
    _report(payload, runs, long, out_dir / "final_report.txt", dataset)
    if logger:
        logger.info(f"Comparison written to {out_dir}")
    return payload


def _time_plot(ok, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(ok) + 2))
    labels = ok["algorithm"].tolist()
    y = np.arange(len(labels))
    for ax, col, title in [(axes[0], "train_seconds", "Training time (s)"), (axes[1], "test_seconds", "Evaluation time (s)")]:
        ax.barh(y, ok[col].astype(float), color=viz.SERIES[0] if col == "train_seconds" else viz.SERIES[1], zorder=2)
        ax.set_yticks(y, labels)
        ax.set_title(title + " - lower is better", loc="left")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    fig.suptitle("Execution time per algorithm (same dataset, same split)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    viz._save(fig, path)


def _tradeoff_plot(long, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (alg, t) in enumerate(long[long["PSNR_dB"].notna()].groupby("algorithm", sort=False)):
        t = t.sort_values("storage_saving_pct")
        ax.plot(t["storage_saving_pct"], t["PSNR_dB"].clip(upper=100), color=viz.color(i), marker="o", label=alg)
    ax.set_xlabel("Storage saving incl. model (%) - higher is better")
    ax.set_ylabel("PSNR (dB, capped at 100) - higher is better")
    ax.set_title("Quality vs storage trade-off, all algorithms", loc="left")
    ax.legend(ncol=2)
    viz._save(fig, path)


def _report(p, runs, long, path, dataset):
    s = p["dataset"]
    lines = ["CHANNEL / BAND REDUCTION - ALGORITHM COMPARISON", "=" * 72,
             f"Dataset      : {s['path']}", f"Dataset id   : {s['dataset_id']}",
             f"Layout       : {s['layout']}, {s['n_samples']} sample(s), {s['n_channels']} channels, dtype {s['dtype']}",
             f"Split        : {p['split']['mode']}  train {p['split']['train']}  test {p['split']['test']}",
             f"Normalisation: {dataset.normalizer.method_used} (fitted once on training pixels, shared by all)",
             f"K values     : {p['k_values']} (identical for every algorithm)", "",
             "ALGORITHM STATUS", "-" * 72]
    for _, r in runs.iterrows():
        extra = f"  [{r['reason_or_error']}]" if r["reason_or_error"] else ""
        t = f"{r['total_seconds']:.1f} s" if pd.notna(r["total_seconds"]) else "-"
        lines.append(f"  {r['algorithm']:<10} {r['status']:<15} total {t:>10}{extra}")
    lines += ["", "BEST ALGORITHM PER METRIC (mean over the shared K values; direction-aware; ties reported)",
              "-" * 72]
    for m, b in p["best_per_metric"].items():
        arrow = "higher is better" if b["direction"] == "max" else "lower is better"
        lines.append(f"  {b['label']:<32} ({arrow:<16}): {b['best_algorithm'] or 'n/a'} = {_fmt(b['best_value'])}")
        if b["not_applicable"]:
            lines.append(f"  {'':<52}not applicable: {', '.join(b['not_applicable'])}")
        if b["note"]:
            lines.append(f"  {'':<52}note: {b['note']}")
    lines += ["", "RUN-LEVEL COST", "-" * 72]
    for m, b in p["run_level"].items():
        lines.append(f"  {b['label']:<40}: {b['best_algorithm'] or 'n/a'} = {_fmt(b['best_value'])}")
    if len(long):
        lines += ["", "MEAN OF KEY METRICS OVER THE SHARED K VALUES (per algorithm; per-K tables in metric_comparison/)",
                  "-" * 72]
        cols = [c for c in ["PSNR_dB", "SSIM", "RMSE", "MAE", "storage_saving_pct", "accuracy", "f1_macro",
                            "forward_seconds"] if long[c].notna().any()]
        shared = long[long["k"].isin(p["k_values"])]
        agg = shared.groupby("algorithm", sort=False)[cols].mean()
        agg.insert(0, "K evaluated", shared.groupby("algorithm", sort=False)["k"].apply(lambda s: len(set(s))))
        lines.append(agg.to_string(float_format=lambda v: f"{v:.5g}", na_rep="n/a"))
        lines += ["", "WINNER PER K", "-" * 72]
        for m in [c for c in ["PSNR_dB", "SSIM", "RMSE", "accuracy"] if c in p["best_per_metric_per_k"]]:
            w = {k: a for k, a in p["best_per_metric_per_k"][m].items() if a}
            lines.append(f"  {METRICS[m]['label']:<12}: " + ", ".join(f"K={k}: {a}" for k, a in w.items()))
    lines += ["", "Notes", "-" * 72,
              "  * n/a = the metric does not apply to that algorithm (e.g. LDA has no reconstruction; only LDA has",
              "    classification scores) or the algorithm did not run. No artificial values are inserted.",
              "  * PSNR is infinite for a lossless reconstruction (e.g. PCA with K = C); graphs cap it at 100 dB.",
              "  * Per-algorithm details (models, reduced/reconstructed data, all original metrics and graphs) are in",
              "    the algorithm folders listed in comparison.json."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
