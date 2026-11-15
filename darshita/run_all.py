"""
RUN ALL - every method on every dataset, then one summary across datasets.

For each dataset it runs step 6 (rebuild quality: PSNR, SAM) and, if the dataset has labels,
step 8 (classification: OA, AA, Kappa). Then it ranks the methods on every dataset and
averages the ranks: a method that is near the top on many different datasets is a
general winner, not a lucky one.

Run (from the repo folder):
    python darshita/run_all.py --list                         # which datasets are ready
    python darshita/run_all.py                                # all available datasets
    python darshita/run_all.py indian_pines pavia_u salinas   # only these
    python darshita/run_all.py --runs 10 --classifiers svm knn rf   # full paper protocol (slow)
    python darshita/run_all.py --summary-only                 # just rebuild the summary

Outputs (darshita/outputs/):
    all_datasets_summary.md / .csv     PSNR and OA of every method on every dataset + average rank
    all_datasets_rank.png              average rank of every method (1 = best)
"""
import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common import OUT_DIR, available_datasets, dataset_name, default_k, load_labels

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY_COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a", "Baseline": "#555555"}

p = argparse.ArgumentParser()
p.add_argument("datasets", nargs="*", help="dataset names or files (default: all available)")
p.add_argument("--list", action="store_true", help="only show which datasets are ready")
p.add_argument("--runs", type=int, default=5)
p.add_argument("--classifiers", nargs="+", default=["svm"])
p.add_argument("--summary-only", action="store_true", help="skip running, only rebuild the summary")
p.add_argument("--redo", action="store_true", help="run step 6 again even if its results exist")
args = p.parse_args()

datasets = args.datasets or available_datasets()
if args.list or not datasets:
    print("Ready:", ", ".join(available_datasets()) or "nothing yet")
    print("Get more with:  python darshita/data_3_download_benchmarks.py")
    sys.exit()


def run(script, *extra):
    cmd = [sys.executable, os.path.join(HERE, script), *extra]
    print("\n>>>", " ".join(cmd[1:]))
    if subprocess.run(cmd).returncode != 0:
        print(f"!!! {script} failed on {extra[0]}; continuing with the next one")


if not args.summary_only:
    for ds in datasets:
        if args.redo or not os.path.exists(f"{OUT_DIR}/final_results_{dataset_name(ds)}.csv"):
            run("step6_compare_all.py", ds)
        if load_labels(ds) is not None:
            run("step8_classification.py", ds, "--runs", str(args.runs), "--classifiers", *args.classifiers)

# ---------------------------------------------------------------- summary across datasets
psnr_rows, oa_rows = [], []
for ds in datasets:
    tag = dataset_name(ds)
    f6 = f"{OUT_DIR}/final_results_{tag}.csv"
    if os.path.exists(f6):
        d = pd.read_csv(f6)
        d = d[~d.Family.isin(["Best possible"])]
        n_bands = round(d.K.iloc[0] * 100 / d["Data sent %"].iloc[0])   # total bands of this dataset
        k_main = default_k(n_bands)
        for _, r in d[d.K == k_main].iterrows():
            psnr_rows.append({"Dataset": tag, "Method": r.Method, "Family": r.Family, "K": k_main, "PSNR": r.PSNR})
    f8 = f"{OUT_DIR}/step8_classification_{tag}.csv"
    if os.path.exists(f8):
        d = pd.read_csv(f8)
        d = d[(d.Classifier == d.Classifier.iloc[0]) & (d.Family != "Reference")]
        aoa = d.groupby(["Method", "Family"]).OA.mean().reset_index()      # average over all K
        for _, r in aoa.iterrows():
            oa_rows.append({"Dataset": tag, "Method": r.Method, "Family": r.Family, "AOA": r.OA})

lines = ["# All datasets: summary", ""]
tables = {}
for title, rows, col in [("Rebuild quality: PSNR (dB) at the main K", psnr_rows, "PSNR"),
                         ("Classification: AOA (%), OA averaged over K = 5..30", oa_rows, "AOA")]:
    if not rows:
        continue
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["Method", "Family"], columns="Dataset", values=col)
    ranks = wide.rank(ascending=False)                       # 1 = best on that dataset
    wide["Average rank"] = ranks.mean(axis=1)
    wide = wide.sort_values("Average rank")
    tables[col] = wide
    lines += [f"## {title}", "", "| Method | Family | " + " | ".join(wide.columns) + " |",
              "|---|---|" + "---|" * len(wide.columns)]
    for (m, fam), r in wide.iterrows():
        lines.append(f"| {m} | {fam} | " + " | ".join(f"{v:.2f}" if pd.notna(v) else "–" for v in r.values) + " |")
    fam_rank = wide.groupby(level="Family")["Average rank"].mean().sort_values()
    lines += ["", "Family average rank (1 = best): " +
              ", ".join(f"{f} {v:.2f}" for f, v in fam_rank.items()), ""]
    wide.to_csv(f"{OUT_DIR}/all_datasets_{col.lower()}.csv", float_format="%.4f")

text = "\n".join(lines)
with open(f"{OUT_DIR}/all_datasets_summary.md", "w", encoding="utf-8") as f:
    f.write(text)
print("\n" + text)

if tables:
    fig, axes = plt.subplots(1, len(tables), figsize=(6.5 * len(tables), 5), squeeze=False)
    for ax, (col, wide) in zip(axes[0], tables.items()):
        w = wide.reset_index()
        ax.barh(w.Method, w["Average rank"], color=[FAMILY_COLOUR.get(f, "#888") for f in w.Family])
        ax.invert_yaxis()
        for i, v in enumerate(w["Average rank"]):
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
        n = len([c for c in wide.columns if c != "Average rank"])
        ax.set_xlabel(f"average rank over {n} dataset(s), 1 = best")
        ax.set_title("Rebuild (PSNR)" if col == "PSNR" else "Classification (AOA)")
        ax.set_xlim(0, w["Average rank"].max() + 1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOUR.values()]
    fig.legend(handles, list(FAMILY_COLOUR), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT_DIR}/all_datasets_rank.png", dpi=120)
    print(f"Saved all_datasets_summary.md and all_datasets_rank.png in {OUT_DIR}/")
