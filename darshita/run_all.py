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
    all_datasets_summary.md / .csv     PSNR, SSIM and AOA of every method on every dataset + average rank
    all_datasets_rank.png              average rank of every method (1 = best)
    all_results.xlsx                   every value of every table above, one sheet each
"""
import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common import OUT_DIR, available_datasets, dataset_name, default_k, load_labels

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY_COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a", "Hybrid": "#a5359b",
                 "Baseline": "#555555"}

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
        f6 = f"{OUT_DIR}/final_results_{dataset_name(ds)}.csv"
        if args.redo or not os.path.exists(f6) or "SSIM" not in pd.read_csv(f6, nrows=1).columns:
            run("step6_compare_all.py", ds)
        if load_labels(ds) is not None:
            run("step8_classification.py", ds, "--runs", str(args.runs), "--classifiers", *args.classifiers)

# ---------------------------------------------------------------- summary across datasets
psnr_rows, ssim_rows, oa_rows = [], [], []
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
            if "SSIM" in d.columns:                     # results made before SSIM was added have none
                ssim_rows.append({"Dataset": tag, "Method": r.Method, "Family": r.Family, "K": k_main,
                                  "SSIM": r.SSIM})
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
                         ("Rebuild quality: SSIM (0-1) at the main K", ssim_rows, "SSIM"),
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
        fmt = "{:.4f}" if col == "SSIM" else "{:.2f}"
        lines.append(f"| {m} | {fam} | " + " | ".join(fmt.format(v) if pd.notna(v) else "–" for v in r.values[:-1])
                     + f" | {r.values[-1]:.2f} |")
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
        ax.set_title({"PSNR": "Rebuild (PSNR)", "SSIM": "Rebuild (SSIM)", "AOA": "Classification (AOA)"}[col])
        ax.set_xlim(0, w["Average rank"].max() + 1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOUR.values()]
    fig.legend(handles, list(FAMILY_COLOUR), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT_DIR}/all_datasets_rank.png", dpi=200)
    print(f"Saved all_datasets_summary.md and all_datasets_rank.png in {OUT_DIR}/")

# ---------------------------------------------------------------- one Excel file with every value
sheets = []
for ds in datasets:
    tag = dataset_name(ds)
    for prefix, label in [("final_results", "Rebuild"), ("step8_classification", "Classif")]:
        path = f"{OUT_DIR}/{prefix}_{tag}.csv"
        if os.path.exists(path):
            sheets.append((f"{label} {tag}"[:31], pd.read_csv(path)))
for col, wide in tables.items():
    sheets.append((f"All datasets {col}", wide.reset_index()))
if sheets:
    from openpyxl.styles import Font, PatternFill
    xlsx = f"{OUT_DIR}/all_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        pd.DataFrame({"Sheet": [n for n, _ in sheets], "Rows": [len(d) for _, d in sheets]}).to_excel(
            xl, sheet_name="Contents", index=False, startrow=6)
        ws = xl.sheets["Contents"]
        for i, line in enumerate(["Band selection results - Darshita Bansal",
                                  "Rebuild: PSNR (dB) and SSIM (0-1) higher = better; SAM (degrees), "
                                  "Redundancy (0-1) and Time (s) lower = better",
                                  "Classif: OA, AA (%) and Kappa from an SVM trained on 10% of the labelled pixels",
                                  "All datasets: value at the main K (10 bands, 4 for Raichur) + average rank, "
                                  "1 = best"], start=1):
            ws.cell(row=i, column=1, value=line)
        ws.cell(row=1, column=1).font = Font(bold=True, size=13)
        for name, d in sheets:
            d.to_excel(xl, sheet_name=name, index=False)
        for name, ws in xl.sheets.items():
            head = 7 if name == "Contents" else 1
            for c in ws[head]:
                c.font, c.fill = Font(bold=True), PatternFill("solid", fgColor="DDDDDD")
            ws.freeze_panes = ws.cell(row=head + 1, column=1)
            for column in ws.columns:
                width = max(len(str(c.value)) for c in column[head - 1:] if c.value is not None) if any(
                    c.value is not None for c in column[head - 1:]) else 8
                ws.column_dimensions[column[0].column_letter].width = min(max(9, width + 2), 40)
            for row in ws.iter_rows(min_row=head + 1):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.0000"
    print(f"Saved {xlsx} (every value, one sheet per table)")
