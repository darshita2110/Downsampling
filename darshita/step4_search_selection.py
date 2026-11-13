"""
STEP 4 - Search-based channel selection (SFS, Genetic Algorithm, PSO).

Ranking judged every channel alone. Search judges a whole GROUP of K channels at once:
"if only these K are sent, how badly is the image rebuilt?" and looks for the group with
the smallest error. Code of the methods: search.py

Run:
    python darshita/step4_search_selection.py                  # practice image (Indian Pines)
    python darshita/step4_search_selection.py raichur          # our real Sentinel-2 data
"""
import json
import sys
import time

import matplotlib.pyplot as plt

from common import (OUT_DIR, default_k, evaluate, load_dataset, scale_01, split_pixels,
                    to_table, uniform)
from search import exhaustive, genetic_algorithm, make_cost, n_groups, pso, sfs

DATASET = sys.argv[1] if len(sys.argv) > 1 else "indian_pines"
cube, names = load_dataset(DATASET)
H, W, C = cube.shape
K = default_k(C)
raw = to_table(cube)
pixels01 = scale_01(raw)
calib, test = split_pixels(len(pixels01))
cost = make_cost(pixels01, calib)            # the "how bad is this group" function

print(f"Choosing {K} of {C} channels. Number of possible groups: {n_groups(C, K):,}")
print("(far too many to try them all when C is large - that is why we need clever search)\n")

methods = {
    "SFS": lambda: sfs(cost, C, K),
    "GA": lambda: genetic_algorithm(cost, C, K, seed=0),
    "PSO": lambda: pso(cost, C, K, seed=0),
    "Exhaustive": lambda: exhaustive(cost, C, K),
}
chosen, histories, rows = {}, {}, []
for name, run in methods.items():
    t = time.perf_counter()
    group, history = run()
    seconds = time.perf_counter() - t
    if group is None:
        print(f"  {name:10s} skipped: {n_groups(C, K):,} groups is too many to try")
        continue
    chosen[name], histories[name] = group, history
    scores = evaluate(pixels01, raw, group, calib, test)
    rows.append((name, scores, seconds))
    print(f"  {name:10s} sends {[names[c] for c in group]}")
chosen["Uniform"] = uniform(C, K)
rows.append(("Uniform", evaluate(pixels01, raw, chosen["Uniform"], calib, test), 0.0))

print(f"\n  {'Method':10s} {'PSNR dB':>8s} {'SAM deg':>8s} {'Redund.':>8s} {'Time s':>7s}")
for name, s, sec in rows:
    print(f"  {name:10s} {s['PSNR']:8.2f} {s['SAM']:8.2f} {s['Redundancy']:8.3f} {sec:7.2f}")
if "Exhaustive" in chosen:
    print("\n  Exhaustive tried every group, so it is the true best. "
          "Same answer from GA / PSO = they found the optimum.")

with open(f"{OUT_DIR}/step4_selected_{DATASET}.json", "w") as f:
    json.dump({"K": K, "selected": chosen}, f, indent=2)

# ---------------------------------------------------------- picture: how the search improves
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name, colour in [("GA", "#eb6834"), ("PSO", "#2a78d6")]:
    axes[0].plot(histories[name], color=colour, lw=2, label=name)
axes[0].set_xlabel("generation (GA) / iteration (PSO)")
axes[0].set_ylabel("best rebuild error so far (lower = better)")
axes[0].set_title("GA and PSO: the best group keeps getting better")
if "Exhaustive" in histories:
    axes[0].axhline(histories["Exhaustive"][0], color="black", ls=":", label="true best (exhaustive)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(range(1, K + 1), histories["SFS"], marker="o", color="#1baf7a", lw=2)
axes[1].set_xlabel("number of channels chosen so far")
axes[1].set_ylabel("rebuild error")
axes[1].set_title("SFS: adding one best channel at a time")
axes[1].grid(alpha=0.3)
fig.suptitle(f"Search-based selection ({DATASET}, K = {K})", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/step4_convergence_{DATASET}.png", dpi=120)
print(f"\nSaved {OUT_DIR}/step4_convergence_{DATASET}.png")
