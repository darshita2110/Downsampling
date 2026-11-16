"""
STEP 7 - Do the findings hold on every date? Combine step 6 results of all Raichur dates.

First run step 6 once per date, e.g.
    python darshita/step6_compare_all.py raichur_2026-09-27
then:
    python darshita/step7_all_dates.py

Reads darshita/outputs/final_results_raichur_<date>.csv for every date found. Writes:
    final_all_dates_raichur.md    PSNR of every method on every date, wins per family
    final_all_dates_raichur.csv   the same table as numbers
    final_all_dates_raichur.png   best of each family vs K, one panel per date
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from common import OUT_DIR, default_k

FAMILIES = ["Search", "Clustering", "Ranking"]
COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a", "Baseline": "#555555"}
TIE_DB = 0.005                                   # PSNR printed with 2 decimals: closer = a tie

# ---------------------------------------------- read every date
frames = []
for path in sorted(glob.glob(f"{OUT_DIR}/final_results_raichur_*.csv")):
    date = re.search(r"raichur_(\d{4}-\d{2}-\d{2})\.csv$", path)
    if date:
        frames.append(pd.read_csv(path).assign(Date=date.group(1)))
if not frames:
    raise SystemExit("No per-date results. Run: python darshita/step6_compare_all.py raichur_<date>")
df = pd.concat(frames, ignore_index=True)
dates = sorted(df.Date.unique())
n_channels = round(df.K.iloc[0] * 100 / df["Data sent %"].iloc[0])
K = default_k(n_channels)
print(f"Dates: {', '.join(dates)}   ({n_channels} bands, main K = {K})\n")

# ---------------------------------------------- table 1: every method on every date at the main K
at_k = df[df.K == K].pivot(index="Method", columns="Date", values="PSNR")[dates]
at_k["Mean"] = at_k[dates].mean(axis=1)
at_k["Std"] = at_k[dates].std(axis=1)
family = df.drop_duplicates("Method").set_index("Method").Family
at_k.insert(0, "Family", family.reindex(at_k.index))
at_k = at_k.sort_values("Mean", ascending=False)
at_k.to_csv(f"{OUT_DIR}/final_all_dates_raichur.csv", float_format="%.4f")

# ---------------------------------------------- table 2: best of each family, per date and K
best = (df[df.Family.isin(FAMILIES)].groupby(["Date", "K", "Family"]).PSNR.max().unstack("Family"))
top = best.max(axis=1)
wins = {f: int((best[f] >= top - TIE_DB).sum()) for f in FAMILIES}   # ties count for each family
n_cases = len(best)
gap_k = best.xs(K, level="K")
gap_k = pd.DataFrame({f"Search minus {f}": gap_k["Search"] - gap_k[f] for f in ["Clustering", "Ranking"]})
uniform = df[(df.Method == "Uniform") & (df.K == K)].set_index("Date").PSNR
below_uniform = [d for d in dates if best.loc[(d, K), "Ranking"] < uniform[d]]
optimum = df[(df.Method == "Exhaustive") & (df.K == K)].set_index("Date").PSNR
reach = {m: [d for d in dates if abs(at_k.loc[m, d] - optimum[d]) < TIE_DB] for m in ["GA", "PSO", "SFS"]}

# ---------------------------------------------- print and save the summary
fmt = lambda v: f"{v:.2f}"
lines = [f"## Raichur, all dates: {K} of {n_channels} bands", "",
         f"PSNR (dB) of every method on every date at K = {K}, sorted by the mean.", "",
         "| Method | Family | " + " | ".join(dates) + " | Mean | Std |",
         "|---|---|" + "---|" * (len(dates) + 2)]
for m, r in at_k.iterrows():
    lines.append(f"| {m} | {r.Family} | " + " | ".join(fmt(r[d]) for d in dates)
                 + f" | {fmt(r.Mean)} | {fmt(r.Std)} |")
lines += ["", f"Best family over {n_cases} cases ({len(dates)} dates x {best.index.get_level_values('K').nunique()} values of K), "
          "ties counted for each:", "",
          "| Family | Cases won or tied |", "|---|---|"]
lines += [f"| {f} | {wins[f]} of {n_cases} |" for f in FAMILIES]
lines += ["", f"How far the best clustering and best ranking method are behind search at K = {K} (dB):", "",
          "| Date | Search minus Clustering | Search minus Ranking |", "|---|---|---|"]
lines += [f"| {d} | {fmt(r.iloc[0])} | {fmt(r.iloc[1])} |" for d, r in gap_k.iterrows()]
lines += ["", f"Reached the true best (exhaustive) at K = {K}: "
          + ", ".join(f"{m} on {len(v)} of {len(dates)} dates" for m, v in reach.items()) + ".",
          f"Best ranking method below the uniform baseline at K = {K}: "
          + (", ".join(below_uniform) if below_uniform else "never") + "."]
text = "\n".join(lines) + "\n"
with open(f"{OUT_DIR}/final_all_dates_raichur.md", "w", encoding="utf-8") as f:
    f.write(text)
print(text)

# ---------------------------------------------- picture: best of each family vs K, one panel per date
fig, axes = plt.subplots(1, len(dates), figsize=(4.2 * len(dates), 4.6), sharey=True, squeeze=False)
for ax, d in zip(axes[0], dates):
    part = best.xs(d, level="Date")
    for f in FAMILIES:
        ax.plot(part.index, part[f], "-o", color=COLOUR[f], lw=2, ms=5, label=f"best {f.lower()}")
    u = df[(df.Method == "Uniform") & (df.Date == d)]
    ax.plot(u.K, u.PSNR, "-.", color=COLOUR["Baseline"], lw=1.5, label="uniform (baseline)")
    e = df[(df.Method == "Exhaustive") & (df.Date == d)]
    if len(e):
        ax.plot(e.K, e.PSNR, ":", color="black", lw=1.5, label="true best (exhaustive)")
    ax.set_title(d, fontsize=11, fontweight="bold")
    ax.set_xticks(sorted(df.K.unique()))
    ax.set_xlabel("K = bands sent")
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel("PSNR (dB), higher = better")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10)
fig.suptitle("Raichur: best method of each family on every date", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f"{OUT_DIR}/final_all_dates_raichur.png", dpi=120)
print(f"Saved final_all_dates_raichur.md, .csv and .png in {OUT_DIR}/")
