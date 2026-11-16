"""
Summarise results/benchmarks/<dataset>_all.json into one cross-dataset table
plus mean ranks. Reads saved JSON only - no refitting.

Output: results/benchmarks_summary.md and results/benchmarks_summary.csv
Ranks are computed per dataset among the methods that ran on ALL 8,000-pixel
scenes (PCA, Isomap, Kernel PCA, LLE); SMACOF (2,000 pixels) is excluded from
ranking because its sample size differs. Only 3 labelled scenes + 1 unlabelled
scene: mean ranks are descriptive, not a statistical test.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "benchmarks"
ORDER = ["indian_pines", "salinas", "ksc", "botswana"]
METHODS = ["PCA baseline", "Isomap", "Kernel PCA", "LLE", "Isomap with SMACOF"]
RANKED = ["PCA baseline", "Isomap", "Kernel PCA", "LLE"]


def main():
    data = {n: json.loads((BENCH / f"{n}_all.json").read_text()) for n in ORDER if (BENCH / f"{n}_all.json").exists()}
    header = ["Dataset", "Method", "Pixels", "Trust.", "Contin.", "Nbr. pres.", "Time (s)", "Mem (MB)",
              "OA random", "OA spatial", "AA spatial", "Kappa spatial"]
    rows = []
    for ds, res in data.items():
        raw = res.get("Raw bands (reference)")
        for m in METHODS:
            r = res.get(m)
            if not r or "metrics" not in r:
                continue
            mt, c = r["metrics"], r.get("classification_5nn")
            t = lambda k, f: f"{mt[k]:{f}}" if mt.get(k) is not None else "n/a"
            rows.append([ds, m, r["n_pixels_used"], t("trustworthiness", ".4f"), t("continuity", ".4f"),
                         t("neighborhood_preservation", ".3f"), t("fit_time_seconds", ".2f"),
                         t("peak_memory_mb", ".1f")] +
                        ([f"{c['random_5fold']['overall_accuracy']:.3f}", f"{c['spatial_block_5fold']['overall_accuracy']:.3f}",
                          f"{c['spatial_block_5fold']['average_class_accuracy']:.3f}",
                          f"{c['spatial_block_5fold']['kappa']:.3f}"] if c else ["no labels"] * 4))
        if raw:
            c = raw["classification"]
            rows.append([ds, "All bands (reference)", raw["n_pixels_used"], "-", "-", "-", "-", "-",
                         f"{c['random_5fold']['overall_accuracy']:.3f}", f"{c['spatial_block_5fold']['overall_accuracy']:.3f}",
                         f"{c['spatial_block_5fold']['average_class_accuracy']:.3f}", f"{c['spatial_block_5fold']['kappa']:.3f}"])

    def ranks(metric_getter, higher_better=True):
        out = {m: [] for m in RANKED}
        for ds, res in data.items():
            vals = {m: metric_getter(res.get(m)) for m in RANKED}
            if any(v is None for v in vals.values()):
                continue
            order = sorted(RANKED, key=lambda m: vals[m], reverse=higher_better)
            for i, m in enumerate(order, 1):
                out[m].append(i)
        return out

    np_r = ranks(lambda r: r["metrics"]["neighborhood_preservation"] if r and "metrics" in r else None)
    oa_r = ranks(lambda r: r["classification_5nn"]["spatial_block_5fold"]["overall_accuracy"]
                 if r and r.get("classification_5nn") else None)
    mem_r = ranks(lambda r: r["metrics"]["peak_memory_mb"] if r and "metrics" in r else None, higher_better=False)

    def mean(v): return f"{sum(v)/len(v):.2f} (n={len(v)})" if v else "n/a"
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    md += ["", "Mean rank (1 = best) across scenes; lower is better:", "",
           "| Method | Neighbourhood preservation (4 scenes) | Spatial-block accuracy (3 labelled scenes) | Peak memory (4 scenes) |",
           "|---|---|---|---|"]
    md += [f"| {m} | {mean(np_r[m])} | {mean(oa_r[m])} | {mean(mem_r[m])} |" for m in RANKED]
    md += ["", "Notes: SMACOF excluded from ranks (2,000-pixel sample). Timings: single machine, one run.",
           "Botswana has no ground truth supplied, so only neighbourhood metrics exist for it.",
           "Pavia University could not be run: only its ground-truth file was supplied, not the image cube."]
    (ROOT / "results" / "benchmarks_summary.md").write_text("\n".join(md))
    with open(ROOT / "results" / "benchmarks_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
