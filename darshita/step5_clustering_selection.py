"""
STEP 5 - Graph / clustering-based channel selection (WaLuDi, WaLuMI, Affinity Propagation).

Group channels that look alike into K groups, then send one channel per group.
Code of the methods: clustering.py

Run:
    python darshita/step5_clustering_selection.py                  # practice image (Indian Pines)
    python darshita/step5_clustering_selection.py raichur          # our real Sentinel-2 data
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from clustering import ap_select, waludi, walumi
from common import (OUT_DIR, default_k, evaluate, load_dataset, scale_01, split_pixels,
                    to_table, uniform)

DATASET = sys.argv[1] if len(sys.argv) > 1 else "indian_pines"
cube, names = load_dataset(DATASET)
H, W, C = cube.shape
K = default_k(C)
raw = to_table(cube)
pixels01 = scale_01(raw)
calib, test = split_pixels(len(pixels01))
# distances between channels are measured on 20,000 random pixels (enough, and fast)
sample = np.random.default_rng(0).choice(len(pixels01), min(20000, len(pixels01)), replace=False)
P = pixels01[sample]

print(f"Grouping {C} channels into {K} groups, then sending one channel per group\n")
results, rows = {}, []
for name, run in [("WaLuDi", lambda: waludi(P, K)), ("WaLuMI", lambda: walumi(P, K)),
                  ("AffinityProp", lambda: ap_select(P, K))]:
    t = time.perf_counter()
    out = run()
    seconds = time.perf_counter() - t
    results[name] = out
    group = out[0]
    rows.append((name, evaluate(pixels01, raw, group, calib, test), seconds))
    print(f"  {name:12s} sends {[names[c] for c in group]}")
chosen = {name: out[0] for name, out in results.items()}
chosen["Uniform"] = uniform(C, K)
rows.append(("Uniform", evaluate(pixels01, raw, chosen["Uniform"], calib, test), 0.0))

print(f"\n  {'Method':12s} {'PSNR dB':>8s} {'SAM deg':>8s} {'Redund.':>8s} {'Time s':>7s}")
for name, s, sec in rows:
    print(f"  {name:12s} {s['PSNR']:8.2f} {s['SAM']:8.2f} {s['Redundancy']:8.3f} {sec:7.2f}")

with open(f"{OUT_DIR}/step5_selected_{DATASET}.json", "w") as f:
    json.dump({"K": K, "selected": chosen}, f, indent=2)

# ---------------------------------------------------------- picture 1: the WaLuDi tree
group_of, tree, D = results["WaLuDi"][1], results["WaLuDi"][2], results["WaLuDi"][3]
fig, ax = plt.subplots(figsize=(13, 4.8))
cut = tree[-(K - 1), 2] if K > 1 else tree[-1, 2]          # height where the tree has K branches
dendrogram(tree, ax=ax, color_threshold=cut, no_labels=C > 40,
           labels=names if C <= 40 else None, above_threshold_color="#bbbbbb")
ax.axhline(cut, color="red", ls="--", lw=1)
ax.set_title(f"WaLuDi tree: channels that join low down are alike. Cutting at the red line gives "
             f"{K} groups (colours); one channel is sent from each", fontsize=10)
ax.set_xlabel("channels")
ax.set_ylabel("distance when merged")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/step5_tree_{DATASET}.png", dpi=120)
print(f"\nSaved {OUT_DIR}/step5_tree_{DATASET}.png")

# ---------------------------------------------------------- picture 2: similarity map + picks
corr = np.abs(np.corrcoef(P, rowvar=False))
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.1]})
im = axes[0].imshow(corr, cmap="viridis", vmin=0, vmax=1)
axes[0].set_title("How alike every pair of channels is (|correlation|)\nbright blocks = groups of near-copies")
axes[0].set_xlabel("channel")
axes[0].set_ylabel("channel")
fig.colorbar(im, ax=axes[0], fraction=0.046)
for row, (name, group) in enumerate(chosen.items()):
    axes[1].barh(row, C, color="#eeeeee", height=0.6)
    axes[1].vlines(group, row - 0.3, row + 0.3, color="#1baf7a" if name != "Uniform" else "#555555", lw=4)
axes[1].set_yticks(range(len(chosen)))
axes[1].set_yticklabels(list(chosen))
axes[1].invert_yaxis()
axes[1].set_xlim(-0.5, C - 0.5)
if C <= 40:
    axes[1].set_xticks(range(C))
    axes[1].set_xticklabels(names)
axes[1].set_title("Channels sent by each clustering method: spread across the groups")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/step5_groups_{DATASET}.png", dpi=120)
print(f"Saved {OUT_DIR}/step5_groups_{DATASET}.png")
