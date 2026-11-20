"""
STEP 8 - Classification test, on ANY dataset that has labels.

Steps 2-7 asked: "can the ground REBUILD all channels from the K that were sent?"
This step asks: "can a computer still tell the land-cover CLASSES apart using only the
K channels that were sent?" This is how most band-selection papers report results, so these
numbers can be put next to theirs.

Protocol (same as most papers):
  - only labelled pixels are used; 10% of every class trains the classifier, 90% tests it
  - repeated for several random splits (--runs), reported as mean +/- std
  - classifiers: SVM (RBF), and optionally KNN (k=5) and Random Forest (100 trees)
  - metrics: OA (overall accuracy), AA (average of per-class accuracies), Kappa,
    and AOA (OA averaged over all K values)
The channels are NOT chosen again here: they are read from step 6's results, so both steps
use exactly the same bands. Run step 6 on the same dataset first.

Run:
    python darshita/step8_classification.py indian_pines
    python darshita/step8_classification.py pavia_u --runs 10 --classifiers svm knn rf
    python darshita/step8_classification.py data/darshita/MyImage.mat      (needs MyImage_gt.mat)
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import OUT_DIR, dataset_name, default_k, load_dataset, load_labels, to_table

p = argparse.ArgumentParser(description="Classification test of the chosen channels")
p.add_argument("dataset", nargs="?", default="indian_pines")
p.add_argument("--runs", type=int, default=5, help="random train/test splits (papers often use 10)")
p.add_argument("--train", type=float, default=0.10, help="share of labelled pixels used for training")
p.add_argument("--classifiers", nargs="+", default=["svm"], choices=["svm", "knn", "rf"])
args = p.parse_args()

FAMILY_COLOUR = {"Ranking": "#2a78d6", "Search": "#eb6834", "Clustering": "#1baf7a", "Hybrid": "#a5359b",
                 "Baseline": "#555555", "Best possible": "#000000"}
STYLE = {"Variance": "-o", "Entropy": "--s", "MVPCA": ":^", "SFS": "-o", "GA": "--s", "PSO": ":^",
         "WaLuDi": "-o", "WaLuMI": "--s", "AffinityProp": ":^", "SegWard": "-D", "Uniform": "-.", "Exhaustive": ":"}

# ---------------------------------------------- 1. data, labels and step 6's channel choices
cube, names = load_dataset(args.dataset)
DATASET = dataset_name(args.dataset)
H, W, C = cube.shape
labels = load_labels(args.dataset)
if labels is None:
    raise SystemExit(f"{DATASET} has no labels file, so there is nothing to classify. "
                     "(Raichur has no crop labels: use step 6's PSNR results for it.)")
if labels.shape != (H, W):
    raise SystemExit(f"Labels shape {labels.shape} does not match the image {(H, W)}")
choices_file = f"{OUT_DIR}/final_results_{DATASET}.csv"
if not os.path.exists(choices_file):
    raise SystemExit(f"Run step 6 first:  python darshita/step6_compare_all.py {args.dataset}")
choices = pd.read_csv(choices_file)
index_of = {n: i for i, n in enumerate(names)}

raw = to_table(cube)
y_all = labels.ravel()
labelled = np.where(y_all > 0)[0]
classes, counts = np.unique(y_all[labelled], return_counts=True)
print(f"{DATASET}: {len(classes)} classes, {len(labelled):,} labelled pixels, "
      f"smallest class {counts.min()} pixels. {args.runs} runs, {int(args.train * 100)}% training")

splits = [train_test_split(labelled, train_size=args.train, stratify=y_all[labelled], random_state=r)
          for r in range(args.runs)]


def make_model(kind):
    """Same fixed settings for every method (no tuning), so only the channels differ."""
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=100, gamma="scale"))
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)


def score(channels, kind):
    """OA %, AA %, Kappa for every split."""
    out = []
    for train, test in splits:
        model = make_model(kind).fit(raw[np.ix_(train, channels)], y_all[train])
        pred = model.predict(raw[np.ix_(test, channels)])
        out.append((100 * accuracy_score(y_all[test], pred),
                    100 * recall_score(y_all[test], pred, average="macro"),     # = average accuracy
                    cohen_kappa_score(y_all[test], pred)))
    return np.array(out)


# ---------------------------------------------- 2. classify with the chosen channels
rows = []
for kind in args.classifiers:
    for _, r in choices.iterrows():
        if "Indices" in r and isinstance(r.Indices, str):            # step 6 saves the band numbers
            channels = [int(c) for c in r.Indices.split()]
        else:                                                        # older step 6 files: names only
            parts = str(r.Channels).split()
            pairs = [f"{a} {b}" for a, b in zip(parts[::2], parts[1::2])]
            channels = [index_of[n] for n in (parts if parts[0] in index_of else pairs)]
        s = score(channels, kind)
        rows.append({"Classifier": kind.upper(), "Method": r.Method, "Family": r.Family, "K": r.K,
                     "OA": s[:, 0].mean(), "OA std": s[:, 0].std(), "AA": s[:, 1].mean(),
                     "AA std": s[:, 1].std(), "Kappa": s[:, 2].mean(), "Kappa std": s[:, 2].std()})
        print(f"  {kind.upper():3s} K={r.K:<3} {r.Method:13s} OA {s[:, 0].mean():6.2f} +/- {s[:, 0].std():.2f}")
    s = score(list(range(C)), kind)                               # reference: every channel
    rows.append({"Classifier": kind.upper(), "Method": f"All {C} bands", "Family": "Reference", "K": C,
                 "OA": s[:, 0].mean(), "OA std": s[:, 0].std(), "AA": s[:, 1].mean(), "AA std": s[:, 1].std(),
                 "Kappa": s[:, 2].mean(), "Kappa std": s[:, 2].std()})
df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/step8_classification_{DATASET}.csv", index=False, float_format="%.4f")

# ---------------------------------------------- 3. summary tables
K_MAIN = default_k(C)
lines = [f"## {DATASET}: classification with the chosen channels", "",
         f"{len(classes)} classes, {int(args.train * 100)}% of labelled pixels for training, "
         f"{args.runs} random splits (mean ± std).", ""]
for kind in df.Classifier.unique():
    d = df[df.Classifier == kind]
    ref = d[d.Family == "Reference"].iloc[0]
    main = d[d.K == K_MAIN].sort_values("OA", ascending=False)
    aoa = d[d.Family != "Reference"].groupby("Method").OA.mean()
    lines += [f"### {kind}, K = {K_MAIN} of {C} channels", "",
              "| Method | Family | OA (%) | AA (%) | Kappa | AOA over all K (%) |",
              "|---|---|---|---|---|---|",
              f"| All {C} bands | Reference | {ref.OA:.2f} ± {ref['OA std']:.2f} | {ref.AA:.2f} | "
              f"{ref.Kappa:.3f} | – |"]
    for _, r in main.iterrows():
        lines.append(f"| {r.Method} | {r.Family} | {r.OA:.2f} ± {r['OA std']:.2f} | {r.AA:.2f} ± "
                     f"{r['AA std']:.2f} | {r.Kappa:.3f} | {aoa[r.Method]:.2f} |")
    lines.append("")
text = "\n".join(lines)
with open(f"{OUT_DIR}/step8_summary_{DATASET}.md", "w", encoding="utf-8") as f:
    f.write(text)
print("\n" + text)

# ---------------------------------------------- 4. chart: OA vs K (first classifier)
kind = df.Classifier.unique()[0]
d = df[df.Classifier == kind]
fig, ax = plt.subplots(figsize=(10, 5.5))
for method in d[d.Family != "Reference"].Method.unique():
    m = d[d.Method == method]
    ax.errorbar(m.K, m.OA, yerr=m["OA std"], fmt=STYLE.get(method, "-o"), capsize=3, lw=2, ms=6,
                color=FAMILY_COLOUR.get(m.Family.iloc[0], "#888888"), label=f"{method} ({m.Family.iloc[0]})")
ref = d[d.Family == "Reference"].iloc[0]
ax.axhline(ref.OA, color="black", ls=":", lw=1.5, label=f"All {C} bands ({ref.OA:.1f}%)")
ax.set_xlabel("K = number of channels sent")
ax.set_ylabel(f"Overall accuracy (%), {kind}, higher = better")
ax.set_title(f"{DATASET}: classification from the sent channels only", fontsize=11)
ax.set_xticks(sorted(d[d.Family != "Reference"].K.unique()))
ax.grid(alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/step8_oa_vs_k_{DATASET}.png", dpi=200)
print(f"Saved step8_classification_{DATASET}.csv, step8_summary_{DATASET}.md, "
      f"step8_oa_vs_k_{DATASET}.png in {OUT_DIR}/")
