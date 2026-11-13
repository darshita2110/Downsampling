"""
DEMO for the teacher - channel selection with all three families, told as a story in 6 screens.

Run from the repo folder:
    python darshita/demo.py raichur          # our real Sentinel-2 satellite data (12 bands)
    python darshita/demo.py raichur_2026-09-27   # one particular date
    python darshita/demo.py                  # practice image Indian Pines (200 channels)

Each screen opens in a window. Explain it, close the window, and the next one opens.
The terminal prints what to say. All screens are also saved as darshita/outputs/demo_*.png.
Screen 6 needs step 6 to have been run once on the same dataset.
"""
import os
import sys
import time

import matplotlib
if "--save-only" in sys.argv:                 # no windows, only save the pictures
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering import ap_select, waludi, walumi
from common import (OUT_DIR, default_k, evaluate, load_dataset, rebuild, scale_01, split_pixels,
                    to_table, uniform)
from ranking import entropy_scores, mvpca_scores, top_k, variance_scores
from search import genetic_algorithm, make_cost, pso, sfs

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DATASET = args[0] if args else "indian_pines"
SHOW = "--save-only" not in sys.argv
FAMILY = {"Variance": "Ranking", "Entropy": "Ranking", "MVPCA": "Ranking",
          "SFS": "Search", "GA": "Search", "PSO": "Search",
          "WaLuDi": "Clustering", "WaLuMI": "Clustering", "AffinityProp": "Clustering",
          "Uniform": "Baseline"}
COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a", "Baseline": "#555555"}


def screen(fig, name, text):
    path = f"{OUT_DIR}/demo_{DATASET}_{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(text)
    print(f"   (picture saved: {path})")
    if SHOW:
        print("   -> close the window to go to the next screen\n")
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ data
cube, names = load_dataset(DATASET)
H, W, C = cube.shape
K = default_k(C)
raw = to_table(cube)
p01 = scale_01(raw)
calib, test = split_pixels(len(p01))
size_mb = cube.astype(np.int16).nbytes / 1e6           # satellites store 16-bit numbers
if DATASET.startswith("raichur"):                     # "raichur" or one date, "raichur_<date>"
    RGB = [names.index("B04"), names.index("B03"), names.index("B02")]
    place = "Raichur, Karnataka (Sentinel-2 satellite, 20 m pixels)"
else:
    RGB = [29, 19, 9]
    place = "a farm area in Indiana, USA (AVIRIS, practice image)"


def colour_picture(table01_or_cube):
    c = table01_or_cube.reshape(H, W, C) if table01_or_cube.ndim == 2 else table01_or_cube
    out = []
    for b in RGB:
        ch = c[:, :, b].astype(np.float32)
        lo, hi = np.percentile(ch, [1, 99])
        out.append(np.clip((ch - lo) / (hi - lo + 1e-12), 0, 1))
    return np.dstack(out)


print("=" * 72)
print(f" CHANNEL SELECTION FOR SATELLITE DOWNLINKING  -  {DATASET}: {C} channels, send {K}")
print("=" * 72 + "\n")

# ================================================================ SCREEN 1
fig = plt.figure(figsize=(16, 6.4))
fig.suptitle(f"SCREEN 1 - What the satellite captured: one place, {C} channels", fontsize=15,
             fontweight="bold", y=1.0)
ax = fig.add_axes([0.01, 0.03, 0.30, 0.82])
ax.imshow(colour_picture(cube))
ax.set_title(f"{place}\n{H} x {W} pixels (colour picture made from 3 channels)", fontsize=10)
ax.axis("off")
shown = list(range(C)) if C <= 12 else list(range(0, C, 9))[:22]
per_row = 6 if C <= 12 else 11
for i, c in enumerate(shown):
    a = fig.add_axes([0.34 + (i % per_row) * (0.65 / per_row), 0.45 - (i // per_row) * 0.42,
                      0.62 / per_row, 0.34])
    lo, hi = np.percentile(cube[:, :, c], [1, 99])
    a.imshow(cube[:, :, c], cmap="gray", vmin=lo, vmax=hi)
    a.set_title(names[c], fontsize=10)
    a.axis("off")
fig.text(0.66, 0.90, f"Each grey picture is ONE channel (band): the same place in one colour of light."
         + ("" if C <= 12 else f"  22 of the {C} are shown."), ha="center", fontsize=11)
screen(fig, "1_image", f"""SCREEN 1 - THE IMAGE
   This is {place}.
   Our eyes see 3 channels (red, green, blue). This sensor records {C} channels:
   the same place photographed in {C} colours of light, visible to infrared.
   Each grey picture is one channel. Total data: {size_mb:.1f} MB for this image.
""")

# ================================================================ SCREEN 2
corr = np.abs(np.corrcoef(raw, rowvar=False))
np.fill_diagonal(corr, 0)
a_, b_ = np.unravel_index(corr.argmax(), corr.shape)             # the most similar pair
c_, d_ = np.unravel_index(np.where(corr > 0, corr, 2).argmin(), corr.shape)   # least similar
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
fig.suptitle("SCREEN 2 - Why channels can be thrown away: some channels are near copies",
             fontsize=15, fontweight="bold")
for ax, c in zip(axes, [a_, b_, c_, d_]):
    lo, hi = np.percentile(cube[:, :, c], [1, 99])
    ax.imshow(cube[:, :, c], cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(names[c], fontsize=12)
    ax.axis("off")
fig.text(0.30, 0.02, f"{names[a_]} vs {names[b_]}: {100 * corr[a_, b_]:.1f}% similar -> sending both is a waste",
         ha="center", fontsize=12, color="#b00020")
fig.text(0.72, 0.02, f"{names[c_]} vs {names[d_]}: {100 * corr[c_, d_]:.1f}% similar -> different information",
         ha="center", fontsize=12, color="#1a7f37")
screen(fig, "2_problem", f"""SCREEN 2 - THE PROBLEM AND THE IDEA
   The downlink (radio link from satellite to ground) is slow, so we cannot send all {C} channels.
   But {names[a_]} and {names[b_]} are {100 * corr[a_, b_]:.1f}% the same. So we send only K = {K} channels
   ({100 * K / C:.0f}% of the data) and rebuild the other {C - K} on the ground.
   QUESTION: WHICH {K} channels? Three families of methods answer it differently.
""")

# ================================================================ run all methods at K
print("   (running all 9 methods, a few seconds ...)\n")
sample = np.random.default_rng(0).choice(len(p01), min(20000, len(p01)), replace=False)
cost = make_cost(p01, calib)
runs = {
    "Variance": lambda: top_k(variance_scores(raw), K),
    "Entropy": lambda: top_k(entropy_scores(raw), K),
    "MVPCA": lambda: top_k(mvpca_scores(raw, K), K),
    "SFS": lambda: sfs(cost, C, K)[0],
    "GA": lambda: genetic_algorithm(cost, C, K, seed=0)[0],
    "PSO": lambda: pso(cost, C, K, seed=0)[0],
    "WaLuDi": lambda: waludi(p01[sample], K)[0],
    "WaLuMI": lambda: walumi(p01[sample], K)[0],
    "AffinityProp": lambda: ap_select(p01[sample], K)[0],
    "Uniform": lambda: uniform(C, K),
}
picked, scores, seconds = {}, {}, {}
for m, run in runs.items():
    t = time.perf_counter()
    picked[m] = run()
    seconds[m] = time.perf_counter() - t
    scores[m] = evaluate(p01, raw, picked[m], calib, test)

# ================================================================ SCREEN 3
fig, ax = plt.subplots(figsize=(15, 6.2))
fig.suptitle(f"SCREEN 3 - Which {K} channels each method sends", fontsize=15, fontweight="bold")
for row, m in enumerate(picked):
    ax.barh(row, C, left=-0.5, height=0.62, color="#f0f0f0")
    ax.vlines(picked[m], row - 0.31, row + 0.31, color=COLOUR[FAMILY[m]], lw=7 if C <= 12 else 4)
ax.set_yticks(range(len(picked)))
ax.set_yticklabels([f"{m}  ({FAMILY[m]})" for m in picked], fontsize=11)
ax.invert_yaxis()
ax.set_xlim(-0.5, C - 0.5)
if C <= 12:
    ax.set_xticks(range(C))
    ax.set_xticklabels(names, fontsize=11)
ax.set_xlabel("channel   -   each coloured mark = one channel that is sent")
for y in (2.5, 5.5, 8.5):
    ax.axhline(y, color="white", lw=6)
screen(fig, "3_selection", f"""SCREEN 3 - THREE WAYS TO CHOOSE
   RANKING (blue): score every channel ALONE, keep the top {K}.
       Variance -> {[names[c] for c in picked['Variance']]}
   SEARCH (orange): try many GROUPS of {K}, keep the group that rebuilds the image best.
       GA       -> {[names[c] for c in picked['GA']]}
   CLUSTERING (green): put similar channels into {K} groups, send one from each group.
       WaLuDi   -> {[names[c] for c in picked['WaLuDi']]}
   UNIFORM (grey): no thinking, evenly spaced channels - the minimum to beat.
""")

# ================================================================ SCREEN 4
best_of = {fam: max((m for m in picked if FAMILY[m] == fam), key=lambda m: scores[m]["PSNR"])
           for fam in ["Ranking", "Search", "Clustering", "Baseline"]}
fig, axes = plt.subplots(2, 5, figsize=(17, 7.6))
fig.suptitle(f"SCREEN 4 - BEFORE vs AFTER: send {K} channels, rebuild all {C} on the ground",
             fontsize=15, fontweight="bold")
axes[0, 0].imshow(colour_picture(p01))
axes[0, 0].set_title(f"BEFORE: original\n{C} channels, {size_mb:.1f} MB", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")
axes[1, 0].text(0.5, 0.5, "Bottom row = error\n(bright = wrong,\ndark = correct)", ha="center",
                va="center", fontsize=12, transform=axes[1, 0].transAxes)
for i, (fam, m) in enumerate(best_of.items(), start=1):
    r = rebuild(p01, picked[m], calib)
    axes[0, i].imshow(colour_picture(r))
    axes[0, i].set_title(f"AFTER - best {fam.lower()}: {m}\n{K} channels ({size_mb * K / C:.2f} MB)  "
                         f"{scores[m]['PSNR']:.1f} dB", fontsize=10, color=COLOUR[fam], fontweight="bold")
    axes[1, i].imshow(np.abs(p01 - r).mean(1).reshape(H, W), cmap="magma", vmin=0, vmax=0.1)
    axes[1, i].set_xlabel(f"error of {m}", fontsize=10)
for a in axes.ravel():
    a.set_xticks([]), a.set_yticks([])
fig.tight_layout()
screen(fig, "4_before_after", f"""SCREEN 4 - BEFORE AND AFTER
   Only {K} channels are sent ({size_mb:.1f} MB -> {size_mb * K / C:.2f} MB). On the ground every channel
   is rebuilt from them with linear regression. One picture per family (its best method).
   Bottom row = where the rebuild is wrong. The darker the error map, the better the choice.
""")

# ================================================================ SCREEN 5
order = sorted(picked, key=lambda m: scores[m]["PSNR"], reverse=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.4, 1]})
fig.suptitle(f"SCREEN 5 - Scoreboard: sending {K} of {C} channels", fontsize=15, fontweight="bold")
vals = [scores[m]["PSNR"] for m in order]
a1.barh(order, vals, color=[COLOUR[FAMILY[m]] for m in order])
a1.invert_yaxis()
a1.set_xlim(min(vals) - 2, max(vals) + 1.5)
for i, v in enumerate(vals):
    a1.text(v + 0.05, i, f"{v:.2f} dB", va="center", fontsize=11)
a1.set_xlabel("PSNR (dB): how close the rebuilt image is to the original, higher = better")
red = [100 * scores[m]["Redundancy"] for m in order]
a2.barh(order, red, color=[COLOUR[FAMILY[m]] for m in order])
a2.invert_yaxis()
for i, v in enumerate(red):
    a2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=11)
a2.set_xlim(0, 115)
a2.set_xlabel("how similar the sent channels are to each other, lower = better")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOUR.values()]
a2.legend(handles, list(COLOUR), loc="lower right")
fig.tight_layout()
screen(fig, "5_scoreboard", f"""SCREEN 5 - RESULT
   Winner: {order[0]} ({FAMILY[order[0]]}) with {scores[order[0]]['PSNR']:.2f} dB.
   Best of each family: """ + ", ".join(f"{fam} {m} {scores[m]['PSNR']:.2f} dB" for fam, m in best_of.items()) + f"""
   Ranking judges channels alone, so it can send near-copies. Search judges the whole group and
   clustering sends one channel per group of look-alikes, so both avoid wasting the downlink.
""")

# ================================================================ SCREEN 6
csv = f"{OUT_DIR}/final_results_{DATASET}.csv"
if os.path.exists(csv):
    df = pd.read_csv(csv)
    fam_best = (df[df.Family.isin(["Ranking", "Search", "Clustering"])]
                .groupby(["Family", "K"]).PSNR.max().unstack(0))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("SCREEN 6 - Sending more channels: which family wins every time?", fontsize=15,
                 fontweight="bold")
    for fam in ["Search", "Clustering", "Ranking"]:
        ax.plot(fam_best.index, fam_best[fam], "-o", color=COLOUR[fam], lw=2.5, ms=7, label=f"best {fam.lower()}")
    u = df[df.Method == "Uniform"]
    ax.plot(u.K, u.PSNR, "-.", color=COLOUR["Baseline"], lw=2, label="uniform (baseline)")
    if "Exhaustive" in set(df.Method):
        e = df[df.Method == "Exhaustive"]
        ax.plot(e.K, e.PSNR, ":", color="black", lw=1.5, label="true best (tries every group)")
    ax.set_xticks(sorted(df.K.unique()))
    ax.set_xlabel("K = number of channels sent")
    ax.set_ylabel("PSNR (dB), higher = better")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    last = fam_best.index.max()
    screen(fig, "6_vs_k", f"""SCREEN 6 - MORE CHANNELS
   Quality goes up as more channels are sent. The order of the families stays mostly the same:
   at K = {last}: """ + ", ".join(f"{fam} {fam_best.loc[last, fam]:.2f} dB" for fam in fam_best.columns) + """
""")
else:
    print(f"SCREEN 6 skipped: run  python darshita/step6_compare_all.py {DATASET}  once first.\n")

print("=" * 72 + "\n DEMO FINISHED\n" + "=" * 72)
