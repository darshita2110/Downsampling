"""
STEP 6 - Final comparison: ranking vs search vs clustering, on one dataset.

For every method and every K it records PSNR, SAM, redundancy and time, then makes the
final table and graphs for the report.

Run:
    python darshita/step6_compare_all.py                  # practice image (Indian Pines)
    python darshita/step6_compare_all.py raichur          # our real Sentinel-2 data

Outputs (darshita/outputs/):
    final_results_<dataset>.csv      every method x every K
    final_psnr_vs_k_<dataset>.png    quality vs number of channels sent
    final_at_k_<dataset>.png         bar chart at the default K
    final_summary_<dataset>.md       the table, ready to paste into the report
"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering import ap_select, kl_distance, mi_distance, ward_select
from common import (OUT_DIR, default_k, evaluate, k_values, load_dataset, scale_01,
                    split_pixels, to_table, uniform)
from ranking import entropy_scores, mvpca_scores, top_k, variance_scores
from search import exhaustive, genetic_algorithm, make_cost, n_groups, pso, sfs

DATASET = sys.argv[1] if len(sys.argv) > 1 else "indian_pines"
cube, names = load_dataset(DATASET)
H, W, C = cube.shape
K_MAIN = default_k(C)
KS = k_values(C)
raw = to_table(cube)
pixels01 = scale_01(raw)
calib, test = split_pixels(len(pixels01))
cost = make_cost(pixels01, calib)
sample = np.random.default_rng(0).choice(len(pixels01), min(20000, len(pixels01)), replace=False)

FAMILY = {"Variance": "Ranking", "Entropy": "Ranking", "MVPCA": "Ranking",
          "SFS": "Search", "GA": "Search", "PSO": "Search",
          "WaLuDi": "Clustering", "WaLuMI": "Clustering", "AffinityProp": "Clustering",
          "Uniform": "Baseline", "Exhaustive": "Best possible"}
COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a",
          "Baseline": "#555555", "Best possible": "#000000"}
STYLE = {"Variance": "-o", "Entropy": "--s", "MVPCA": ":^", "SFS": "-o", "GA": "--s", "PSO": ":^",
         "WaLuDi": "-o", "WaLuMI": "--s", "AffinityProp": ":^", "Uniform": "-.", "Exhaustive": ":"}


# ---------------------------------------------- things that do not depend on K: compute once
def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


print("Preparing (scores and distances that do not depend on K) ...")
var_s, t_var = timed(lambda: variance_scores(raw))
ent_s, t_ent = timed(lambda: entropy_scores(raw))
D_kl, t_kl = timed(lambda: kl_distance(pixels01[sample]))
D_mi, t_mi = timed(lambda: mi_distance(pixels01[sample]))

methods = {
    "Variance": lambda k: (top_k(var_s, k), t_var),
    "Entropy": lambda k: (top_k(ent_s, k), t_ent),
    "MVPCA": lambda k: timed(lambda: top_k(mvpca_scores(raw, k), k)),
    "SFS": lambda k: timed(lambda: sfs(cost, C, k)[0]),
    "GA": lambda k: timed(lambda: genetic_algorithm(cost, C, k, seed=0)[0]),
    "PSO": lambda k: timed(lambda: pso(cost, C, k, seed=0)[0]),
    "WaLuDi": lambda k: (lambda r: (r[0][0], r[1] + t_kl))(timed(lambda: ward_select(D_kl, k))),
    "WaLuMI": lambda k: (lambda r: (r[0][0], r[1] + t_mi))(timed(lambda: ward_select(D_mi, k))),
    "AffinityProp": lambda k: timed(lambda: ap_select(pixels01[sample], k)[0]),
    "Uniform": lambda k: (uniform(C, k), 0.0),
}
if n_groups(C, max(KS)) <= 5000:                        # few channels: the true best is computable
    methods["Exhaustive"] = lambda k: timed(lambda: exhaustive(cost, C, k)[0])

# ---------------------------------------------- run everything
rows = []
for k in KS:
    print(f"K = {k} ...")
    for name, run in methods.items():
        group, seconds = run(k)
        s = evaluate(pixels01, raw, group, calib, test)
        rows.append({"Method": name, "Family": FAMILY[name], "K": k, "Data sent %": 100 * k / C,
                     "PSNR": s["PSNR"], "SAM": s["SAM"], "Redundancy": s["Redundancy"],
                     "Time s": seconds, "Channels": " ".join(names[c] for c in group)})
df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/final_results_{DATASET}.csv", index=False, float_format="%.4f")

# GA and PSO are random: repeat them with 5 seeds at the main K to see how stable they are
seed_rows = []
for name, fn in [("GA", genetic_algorithm), ("PSO", pso)]:
    vals = [evaluate(pixels01, raw, fn(cost, C, K_MAIN, seed=s)[0], calib, test)["PSNR"] for s in range(5)]
    seed_rows.append((name, np.mean(vals), np.std(vals)))

# ---------------------------------------------- table at the main K
main = df[df.K == K_MAIN].sort_values("PSNR", ascending=False)
print(f"\n=== {DATASET}: sending {K_MAIN} of {C} channels ({100 * K_MAIN / C:.0f}% of the data) ===")
print(f"  {'Method':13s} {'Family':13s} {'PSNR dB':>8s} {'SAM deg':>8s} {'Redund.':>8s} {'Time s':>7s}")
for _, r in main.iterrows():
    print(f"  {r.Method:13s} {r.Family:13s} {r.PSNR:8.2f} {r.SAM:8.2f} {r.Redundancy:8.3f} {r['Time s']:7.2f}")
print("\nGA / PSO over 5 random seeds: " +
      ", ".join(f"{n} {m:.2f} ± {s:.2f} dB" for n, m, s in seed_rows))

fam = (df[~df.Family.isin(["Baseline", "Best possible"])]
       .groupby(["Family", "K"]).PSNR.max().unstack())
print("\nBest PSNR (dB) of each family at every K:")
print(fam.round(2).to_string())

with open(f"{OUT_DIR}/final_summary_{DATASET}.md", "w", encoding="utf-8") as f:   # utf-8: keeps ° and ± on Windows
    f.write(f"## {DATASET}: {K_MAIN} of {C} channels ({100 * K_MAIN / C:.0f}% of the data)\n\n")
    f.write("| Method | Family | PSNR (dB) | SAM (°) | Redundancy | Time (s) | Channels sent |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for _, r in main.iterrows():
        f.write(f"| {r.Method} | {r.Family} | {r.PSNR:.2f} | {r.SAM:.2f} | {r.Redundancy:.3f} | "
                f"{r['Time s']:.2f} | {r.Channels} |\n")
    f.write("\nGA / PSO over 5 seeds: " +
            ", ".join(f"{n} {m:.2f} ± {s:.2f} dB" for n, m, s in seed_rows) + "\n\n")
    f.write("Best PSNR (dB) of each family at every K:\n\n| Family | " +
            " | ".join(f"K={k}" for k in fam.columns) + " |\n|---|" + "---|" * len(fam.columns) + "\n")
    for family, r in fam.iterrows():
        f.write(f"| {family} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |\n")

# ---------------------------------------------- graph 1: PSNR vs K
fig, ax = plt.subplots(figsize=(10, 5.5))
for name in methods:
    d = df[df.Method == name]
    ax.plot(d.K, d.PSNR, STYLE[name], color=COLOUR[FAMILY[name]], lw=2 if name != "Exhaustive" else 1.5,
            ms=6, label=f"{name} ({FAMILY[name]})")
ax.set_xlabel("K = number of channels sent")
ax.set_ylabel("PSNR of the rebuilt image (dB), higher = better")
ax.set_title(f"{DATASET}: quality vs channels sent. Blue = ranking, orange = search, green = clustering",
             fontsize=11)
ax.set_xticks(KS)
ax.grid(alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/final_psnr_vs_k_{DATASET}.png", dpi=120)

# ---------------------------------------------- graph 2: everything at the main K
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
order = main.Method.tolist()
colours = [COLOUR[FAMILY[m]] for m in order]
for ax, col, label in [(axes[0], "PSNR", "PSNR (dB), higher = better"),
                       (axes[1], "Redundancy", "redundancy of sent channels, lower = better"),
                       (axes[2], "Time s", "time to choose (s), log scale")]:
    vals = main.set_index("Method").loc[order, col].values
    ax.barh(order, np.maximum(vals, 1e-4) if col == "Time s" else vals, color=colours)
    for i, v in enumerate(vals):
        ax.text(max(v, 1e-4), i, f" {v:.2f}" if col != "Time s" else f" {v:.3g}", va="center", fontsize=9)
    ax.set_xlabel(label)
    if col == "Time s":
        ax.set_xscale("log")
    if col == "PSNR":
        ax.set_xlim(min(vals) - 2, max(vals) + 1.5)
    if col == "Redundancy":
        ax.set_xlim(0, 1.15)
axes[0].invert_yaxis()
fig.suptitle(f"{DATASET}: all methods at K = {K_MAIN} of {C} channels", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/final_at_k_{DATASET}.png", dpi=120)
print(f"\nSaved final_results_{DATASET}.csv, final_psnr_vs_k_{DATASET}.png, "
      f"final_at_k_{DATASET}.png, final_summary_{DATASET}.md in {OUT_DIR}/")
