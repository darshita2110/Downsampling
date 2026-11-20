"""
SHOW REDUCTION - the picture before and after sending only K channels, family by family.

It picks K channels with EVERY method, "sends" only those, rebuilds all channels on the
ground, and shows what was lost. One screen (and one saved file) per family, then a final one:

  Screen 1  RANKING     Variance, Entropy, MVPCA
  Screen 2  SEARCH      SFS, GA, PSO (+ Exhaustive = true best, when there are few channels)
  Screen 3  CLUSTERING  WaLuDi, WaLuMI, Affinity Propagation
  Screen 4  FINAL       best method of each family + Uniform, side by side
  Screen 5  SCOREBOARD  PSNR, redundancy and time of every method in one chart

Every method row shows 4 things:
  original picture | picture rebuilt from only K channels | difference map | one pixel's spectrum
Close a window and the next one opens. The terminal prints a table for every family.

Run (from the repo folder):
    python darshita/show_reduction.py raichur                  # Sentinel-2, K = 4 of 12
    python darshita/show_reduction.py indian_pines             # K = 10 of 200
    python darshita/show_reduction.py pavia_u --k 5            # choose K yourself
    python darshita/show_reduction.py raichur --only search    # one screen: ranking, search, clustering, final
    python darshita/show_reduction.py raichur --no-show        # only save the pictures
    python darshita/show_reduction.py indian_pines --k 25 --search SFS   # K = 25, only SFS from search
Saved as darshita/outputs/show_<dataset>_K<k>_1_ranking.png ... _5_scoreboard.png
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from clustering import ap_select, kl_distance, mi_distance, segward_select, ward_select
from common import (OUT_DIR, dataset_name, default_k, load_dataset, psnr, rebuild, redundancy,
                    sam_degrees, scale_01, split_pixels, ssim, to_table, uniform)
from ranking import entropy_scores, mvpca_scores, top_k, variance_scores
from search import exhaustive, genetic_algorithm, make_cost, n_groups, pso, sfs

p = argparse.ArgumentParser()
p.add_argument("dataset", nargs="?", default="raichur")
p.add_argument("--k", type=int, help="channels to send (default: 4 for 12 bands, 10 for hyperspectral)")
p.add_argument("--only", choices=["ranking", "search", "clustering", "final"], help="show only this screen")
p.add_argument("--no-show", action="store_true", help="only save the pictures, open no window")
p.add_argument("--search", nargs="+", default=["SFS", "GA", "PSO"], type=str.upper, choices=["SFS", "GA", "PSO"],
               help="search methods to run, e.g. --search SFS (default: SFS GA PSO)")
args = p.parse_args()

COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a", "Hybrid": "#a5359b",
          "Baseline": "#555555", "Best possible": "#000000"}

# ------------------------------------------------------------ 1. data (same set-up as step 6)
cube, names = load_dataset(args.dataset)
DATASET = dataset_name(args.dataset)
H, W, C = cube.shape
K = args.k or default_k(C)
raw = to_table(cube)
p01 = scale_01(raw)
calib, test = split_pixels(len(p01))
sample = np.random.default_rng(0).choice(len(p01), min(20000, len(p01)), replace=False)
size_all, size_k = H * W * C * 2 / 1e6, H * W * K * 2 / 1e6          # 16-bit numbers

if DATASET.startswith("raichur"):
    RGB, rgb_note = [names.index("B04"), names.index("B03"), names.index("B02")], "true colour B04 B03 B02"
elif DATASET == "indian_pines":
    RGB, rgb_note = [29, 19, 9], "bands 29, 19, 9"
else:
    RGB = [int(C * 0.145), int(C * 0.095), int(C * 0.045)]
    rgb_note = f"bands {RGB[0]}, {RGB[1]}, {RGB[2]}"
orig = p01.reshape(H, W, C)
LO = [np.percentile(orig[:, :, b], 1) for b in RGB]          # one colour stretch for ALL pictures,
HI = [np.percentile(orig[:, :, b], 99) for b in RGB]         # taken from the original: comparable


def colour(table01):
    c = table01.reshape(H, W, C)
    return np.dstack([np.clip((c[:, :, b] - lo) / (hi - lo + 1e-12), 0, 1) for b, lo, hi in zip(RGB, LO, HI)])


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


# ------------------------------------------------------------ 2. every method chooses K channels
print(f"\n{DATASET}: {H} x {W} pixels, {C} channels. Sending only K = {K} "
      f"({100 * K / C:.0f}% of the data, {size_all:.1f} MB -> {size_k:.1f} MB)\nChoosing channels ...")
cost = make_cost(p01, calib)
FAMILIES = {
    "Ranking": {"Variance": lambda: top_k(variance_scores(raw), K),
                "Entropy": lambda: top_k(entropy_scores(raw), K),
                "MVPCA": lambda: top_k(mvpca_scores(raw, K), K)},
    "Search": {"SFS": lambda: sfs(cost, C, K)[0],
               "GA": lambda: genetic_algorithm(cost, C, K, seed=0)[0],
               "PSO": lambda: pso(cost, C, K, seed=0)[0]},
    "Clustering": {"WaLuDi": lambda: ward_select(kl_distance(p01[sample]), K)[0],
                   "WaLuMI": lambda: ward_select(mi_distance(p01[sample]), K)[0],
                   "AffinityProp": lambda: ap_select(p01[sample], K)[0]},
    "Hybrid": {"SegWard": lambda: segward_select(p01[sample], K)[0]},
    "Baseline": {"Uniform": lambda: uniform(C, K)},
}
FAMILIES["Search"] = {m: f for m, f in FAMILIES["Search"].items() if m in args.search}
if n_groups(C, K) <= 5000:                          # few channels: the true best can be found
    FAMILIES["Search"]["Exhaustive"] = lambda: exhaustive(cost, C, K)[0]

R = {}                                              # method -> everything about it
for family, methods in FAMILIES.items():
    for name, fn in methods.items():
        group, secs = timed(fn)
        rebuilt = rebuild(p01, group, calib)
        R[name] = {"family": family, "group": group, "rebuilt": rebuilt, "time": secs,
                   "psnr": psnr(p01[test], rebuilt[test]), "sam": sam_degrees(p01[test], rebuilt[test]),
                   "ssim": ssim(p01, rebuilt, (H, W), test),
                   "red": redundancy(raw, group),
                   "error": np.abs(p01 - rebuilt).mean(axis=1).reshape(H, W)}

import pandas as pd                                             # every score of this run, also as a table
pd.DataFrame([{"Method": m, "Family": r["family"], "K": K, "Data sent %": 100 * K / C, "PSNR": r["psnr"],
               "SSIM": r["ssim"], "SAM": r["sam"], "Redundancy": r["red"], "Time s": r["time"],
               "Channels": " ".join(names[c] for c in r["group"])} for m, r in R.items()]
             ).sort_values("PSNR", ascending=False).to_csv(f"{OUT_DIR}/show_{DATASET}_K{K}_scores.csv",
                                                          index=False, float_format="%.4f")

ERR_MAX = np.percentile(R["Uniform"]["error"], 99.5) * 1.5      # same colour scale on every error map
PIXEL = test[np.argmax(R["Uniform"]["error"].ravel()[test])]     # one hard pixel, same in every plot


def sent(group):
    return " ".join(names[c] for c in group) if len(group) <= 8 else f"{len(group)} channels"


def print_table(title, rows):
    print(f"\n{title}\n  {'Method':13s} {'PSNR dB':>8s} {'SSIM':>7s} {'SAM °':>7s} {'Redund.':>8s} {'Time s':>7s}  Channels sent")
    for m in rows:
        r = R[m]
        print(f"  {m:13s} {r['psnr']:8.2f} {r['ssim']:7.4f} {r['sam']:7.2f} {r['red']:8.3f} {r['time']:7.2f}  "
              + " ".join(names[c] for c in r["group"]))


# ------------------------------------------------------------ 3. one screen: rows of methods
def screen(rows, title, filename, note=""):
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.6 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1.1, 1.6]})
    for (a0, a1, a2, a3), m in zip(axes, rows):
        r = R[m]
        col = COLOUR[r["family"]]
        a0.imshow(colour(p01))
        a0.set_title(f"ORIGINAL, all {C} channels ({size_all:.1f} MB)", fontsize=9.5)
        a1.imshow(colour(r["rebuilt"]))
        a1.set_title(f"{m}: REBUILT from {K} channels ({size_k:.1f} MB)", fontsize=9.5,
                     color=col, fontweight="bold")
        im = a2.imshow(r["error"], cmap="magma", vmin=0, vmax=ERR_MAX)
        a2.set_title(f"DIFFERENCE: PSNR {r['psnr']:.2f} dB, SSIM {r['ssim']:.3f}\nSAM {r['sam']:.2f}° (brighter = more lost)",
                     fontsize=9.5)
        fig.colorbar(im, ax=a2, fraction=0.046, pad=0.02)
        for ax in (a0, a1, a2):
            ax.set_xticks([]), ax.set_yticks([])
        x = np.arange(C)
        a3.plot(x, p01[PIXEL], color="black", lw=1.8, label="original")
        a3.plot(x, r["rebuilt"][PIXEL], color=col, lw=1.8, ls="--", label="rebuilt")
        a3.scatter(r["group"], p01[PIXEL, r["group"]], color=col, edgecolor="black", s=45, zorder=3,
                   label="sent channels")
        a3.set_title(f"One pixel's spectrum. Sent: {sent(r['group'])}\n"
                     f"redundancy {r['red']:.2f}, chosen in {r['time']:.2f} s", fontsize=9.5)
        if C <= 20:
            a3.set_xticks(x, names, rotation=90, fontsize=8)
        a3.set_ylim(-0.05, 1.05)
        a3.grid(alpha=0.3)
        a3.legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"{title}   |   {DATASET}: {K} of {C} channels sent = {100 * K / C:.0f}% of the data\n{note}",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, f"Pictures: {rgb_note}, same colour stretch in all. Error maps: same scale in all.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.95))
    fig.savefig(f"{OUT_DIR}/show_{DATASET}_K{K}_{filename}.png", dpi=200)
    print(f"  saved {OUT_DIR}/show_{DATASET}_K{K}_{filename}.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)


def by_psnr(names_):
    return sorted(names_, key=lambda m: -R[m]["psnr"])


fam = {f: by_psnr(list(ms)) for f, ms in FAMILIES.items()}
NOTES = {
    "Ranking": "Each channel gets a score ALONE, top K are sent. Look at the spectrum: the sent dots bunch "
               "together -> near-copies, high redundancy.",
    "Search": "Whole groups are tried and the group that rebuilds best is kept. Dots spread out, "
              "difference map is darkest." + (" Exhaustive = every group tried = the true best."
                                               if "Exhaustive" in R else ""),
    "Clustering": "Similar channels are grouped into K groups and one channel is sent from each group. "
                  "Dots cover the whole spectrum, chosen in a fraction of a second.",
}
best = {f: fam[f][0] for f in ["Ranking", "Search", "Clustering"]}
if "Exhaustive" in R and best["Search"] == "Exhaustive":       # show a real search method as the winner
    best["Search"] = next(m for m in fam["Search"] if m != "Exhaustive")

screens = [("ranking", "1_ranking", "RANKING family", fam["Ranking"]),
           ("search", "2_search", "SEARCH family", fam["Search"]),
           ("clustering", "3_clustering", "CLUSTERING family", fam["Clustering"])]
for key, file, title, rows in screens:
    print_table(f"{title} (best first)", rows)
    if args.only in (None, key):
        screen(rows, title, file, NOTES[title.split()[0].capitalize()])

final_rows = by_psnr(list(best.values()) + ["SegWard", "Uniform"])
print_table("FINAL: best of each family + Uniform", final_rows)
if args.only in (None, "final"):
    screen(final_rows, "FINAL: best of each family + SegWard (hybrid) vs Uniform", "4_final",
           "Uniform = K evenly spaced channels, no thinking. A method is only useful if it beats Uniform.")

    # ---------------------------------------------------- 4. scoreboard of every method
    order = by_psnr(list(R))
    fig, axes = plt.subplots(1, 4, figsize=(20, 0.45 * len(order) + 2), sharey=True)
    cols = [COLOUR[R[m]["family"]] for m in order]
    for ax, key, label in zip(axes, ["psnr", "ssim", "red", "time"],
                              ["PSNR (dB), higher = better", "SSIM (0-1), higher = better",
                               "Redundancy, lower = better",
                               "Time to choose (s), lower = better"]):
        vals = [R[m][key] for m in order]
        ax.barh(order, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}" if key == "ssim" else f" {v:.2f}", va="center", fontsize=9)
        ax.set_title(label, fontsize=10)
        if key == "ssim":
            ax.set_xlim(max(0, min(vals) - 0.1), min(1.1, max(vals) + 0.08))
        else:
            ax.set_xlim(min(vals) * 0.9 if key == "psnr" else 0, max(vals) * 1.15 + 1e-9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in COLOUR.items() if any(R[m]["family"] == f for m in R)]
    fig.legend(handles, [f for f in COLOUR if any(R[m]["family"] == f for m in R)], loc="lower center",
               ncol=5, frameon=False)
    fig.suptitle(f"SCOREBOARD   |   {DATASET}: every method, {K} of {C} channels", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(f"{OUT_DIR}/show_{DATASET}_K{K}_5_scoreboard.png", dpi=200)
    print(f"  saved {OUT_DIR}/show_{DATASET}_K{K}_5_scoreboard.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)