"""
STEP 8 - Classification test (Indian Pines only).

Steps 2-7 asked: "can the ground REBUILD all channels from the K that were sent?"
This step asks a second, practical question: "can a computer still tell the CROPS apart
using only the K channels that were sent?"

How:
  1. Load the labels map (which crop grows in each pixel, 16 crop classes).
  2. Keep only the labelled pixels. 10% of them train the classifier, 90% test it.
     The split keeps the share of every class the same ("stratified") and uses seed 0,
     so every method is tested on exactly the same pixels.
  3. For every method and every K, choose the channels exactly like step 6, then train an
     SVM classifier on ONLY those channels and score it on the test pixels:
       Overall Accuracy (OA) = % of test pixels given the right crop
       Cohen's kappa         = accuracy corrected for lucky guesses (1 = perfect, 0 = guessing)
  4. One reference row uses all 200 channels: the most the classifier can get here.

Needs: data/darshita/Indian_pines_gt.mat (variable indian_pines_gt) and scikit-learn.
Run:
    python darshita/step8_classification.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from clustering import ap_select, kl_distance, mi_distance, ward_select
from common import OUT_DIR, k_values, load_dataset, scale_01, split_pixels, to_table, uniform
from ranking import entropy_scores, mvpca_scores, top_k, variance_scores
from search import genetic_algorithm, make_cost, pso, sfs

LABELS_FILE = "data/darshita/Indian_pines_gt.mat"
K_MAIN = 10                                   # the K used in the summary table

# Same names, families, colours and line styles as step 6
FAMILY = {"Variance": "Ranking", "Entropy": "Ranking", "MVPCA": "Ranking",
          "SFS": "Search", "GA": "Search", "PSO": "Search",
          "WaLuDi": "Clustering", "WaLuMI": "Clustering", "AffinityProp": "Clustering",
          "Uniform": "Baseline"}
COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a", "Baseline": "#555555"}
STYLE = {"Variance": "-o", "Entropy": "--s", "MVPCA": ":^", "SFS": "-o", "GA": "--s", "PSO": ":^",
         "WaLuDi": "-o", "WaLuMI": "--s", "AffinityProp": ":^", "Uniform": "-."}

# ---------------------------------------------- 1. load and check the labels
if not os.path.exists(LABELS_FILE):
    raise SystemExit(f"Labels file not found: {LABELS_FILE}\n"
                     "Put Indian_pines_gt.mat (variable indian_pines_gt) there and run again.")
labels = loadmat(LABELS_FILE)["indian_pines_gt"]
cube, names = load_dataset("indian_pines")
H, W, C = cube.shape
if labels.shape != (H, W) or labels.min() < 0 or labels.max() > 16:
    raise SystemExit(f"Unexpected labels: shape {labels.shape}, values {labels.min()}..{labels.max()}. "
                     f"Expected shape {(H, W)} and values 0..16.")
print(f"Labels OK: shape {labels.shape}, {labels.max()} crop classes, "
      f"{(labels > 0).sum():,} labelled pixels (0 = no label, not used)")

raw = to_table(cube)                          # one row per pixel, one column per channel
crop = labels.ravel()                         # the crop class of every pixel, same order as `raw`

# ---------------------------------------------- 2. train / test split of the labelled pixels
labelled = np.where(crop > 0)[0]
train, test = train_test_split(labelled, train_size=0.10, stratify=crop[labelled], random_state=0)
print(f"Train pixels: {len(train):,} (10%)   Test pixels: {len(test):,} (90%)\n")


def classify(channels):
    """Train an SVM on the chosen channels only; return (OA in %, kappa) on the test pixels."""
    # StandardScaler: every channel gets mean 0 and std 1 (learned on the train pixels only).
    # SVC with RBF kernel; C=100 is fixed for every method (no tuning), so the comparison is fair.
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=100, gamma="scale"))
    model.fit(raw[np.ix_(train, channels)], crop[train])
    predicted = model.predict(raw[np.ix_(test, channels)])
    return 100 * accuracy_score(crop[test], predicted), cohen_kappa_score(crop[test], predicted)


# ---------------------------------------------- 3. choose channels exactly like step 6
# Channel selection never looks at the crop labels: it only uses the image itself.
pixels01 = scale_01(raw)
calib, _ = split_pixels(len(pixels01))       # the calibration pixels the search methods use
cost = make_cost(pixels01, calib)
sample = np.random.default_rng(0).choice(len(pixels01), min(20000, len(pixels01)), replace=False)

print("Preparing scores and distances that do not depend on K ...")
var_s = variance_scores(raw)
ent_s = entropy_scores(raw)
D_kl = kl_distance(pixels01[sample])
D_mi = mi_distance(pixels01[sample])

methods = {                                   # each one: K -> list of chosen channels
    "Variance": lambda k: top_k(var_s, k),
    "Entropy": lambda k: top_k(ent_s, k),
    "MVPCA": lambda k: top_k(mvpca_scores(raw, k), k),
    "SFS": lambda k: sfs(cost, C, k)[0],
    "GA": lambda k: genetic_algorithm(cost, C, k, seed=0)[0],
    "PSO": lambda k: pso(cost, C, k, seed=0)[0],
    "WaLuDi": lambda k: ward_select(D_kl, k)[0],
    "WaLuMI": lambda k: ward_select(D_mi, k)[0],
    "AffinityProp": lambda k: ap_select(pixels01[sample], k)[0],
    "Uniform": lambda k: uniform(C, k),
}

rows = []
for k in k_values(C):
    print(f"K = {k} ...")
    for name, choose in methods.items():
        channels = choose(k)
        oa, kappa = classify(channels)
        rows.append({"Method": name, "Family": FAMILY[name], "K": k, "OA %": oa, "Kappa": kappa,
                     "Channels": " ".join(names[c] for c in channels)})

# Reference: the classifier with every channel
oa_all, kappa_all = classify(list(range(C)))
rows.append({"Method": f"All {C} bands", "Family": "Reference", "K": C, "OA %": oa_all,
             "Kappa": kappa_all, "Channels": "all"})
df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/step8_classification_indian_pines.csv", index=False, float_format="%.4f")

# ---------------------------------------------- 4a. table at K = 10
main = df[df.K == K_MAIN].sort_values("OA %", ascending=False)
lines = [f"## Indian Pines: crop classification with {K_MAIN} of {C} channels", "",
         f"SVM (RBF) trained on {len(train):,} labelled pixels (10%), tested on {len(test):,} (90%).", "",
         f"| Method | Family | OA (%) | Kappa | OA behind all {C} bands (points) |",
         "|---|---|---|---|---|",
         f"| All {C} bands | Reference | {oa_all:.2f} | {kappa_all:.3f} | 0.00 |"]
for _, r in main.iterrows():
    lines.append(f"| {r.Method} | {r.Family} | {r['OA %']:.2f} | {r.Kappa:.3f} | {oa_all - r['OA %']:.2f} |")
best = (df[df.Family.isin(["Ranking", "Search", "Clustering"])]
        .groupby(["Family", "K"])["OA %"].max().unstack())
lines += ["", "Best OA (%) of each family at every K:", "",
          "| Family | " + " | ".join(f"K={k}" for k in best.columns) + " |",
          "|---|" + "---|" * len(best.columns)]
lines += [f"| {f} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |" for f, r in best.iterrows()]
text = "\n".join(lines) + "\n"
with open(f"{OUT_DIR}/step8_summary_indian_pines.md", "w", encoding="utf-8") as f:
    f.write(text)
print("\n" + text)

# ---------------------------------------------- 4b. chart: OA vs K
fig, ax = plt.subplots(figsize=(10, 5.5))
for name in methods:
    d = df[df.Method == name]
    ax.plot(d.K, d["OA %"], STYLE[name], color=COLOUR[FAMILY[name]], lw=2, ms=6,
            label=f"{name} ({FAMILY[name]})")
ax.axhline(oa_all, color="black", ls=":", lw=1.5, label=f"All {C} bands ({oa_all:.1f}%)")
ax.set_xlabel("K = number of channels sent")
ax.set_ylabel("Overall accuracy on test pixels (%), higher = better")
ax.set_title("Indian Pines: crop classification (SVM) using only the sent channels", fontsize=11)
ax.set_xticks(k_values(C))
ax.grid(alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/step8_oa_vs_k_indian_pines.png", dpi=120)
print(f"Saved step8_classification_indian_pines.csv, step8_oa_vs_k_indian_pines.png and "
      f"step8_summary_indian_pines.md in {OUT_DIR}/")
