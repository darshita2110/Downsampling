"""
Run all four implemented techniques (Isomap, Isomap+SMACOF, Kernel PCA, LLE)
plus a PCA baseline on the Indian Pines hyperspectral benchmark, using the
SAME shared preprocessing and evaluation code as the Raichur runs.

Settings mirror the Raichur runs: 8,000 random labelled pixels (2,000 for
SMACOF, same cost-driven reduction as Raichur), k=12, 3 components, seed 42.
Kernel PCA uses gamma = 1/n_features = 1/200 (sklearn default, as before).

On top of the shared neighbourhood metrics this adds a CLASSIFICATION check
that Raichur could not offer: a 5-NN classifier on the 3-D embedding, scored
with overall accuracy (OA), average per-class accuracy (AA) and Cohen's kappa
under two cross-validation schemes:
  - random stratified 5-fold (optimistic: neighbouring pixels leak across folds)
  - spatial-block 5-fold (29x29-pixel blocks held out together; fairer)
IMPORTANT CAVEAT: the embeddings are transductive (every method is fitted on
all sampled pixels, unlabelled use only - labels are never used in fitting).
The classifier is then cross-validated on the embedded points. This is a
fair comparison between methods, but not an out-of-sample deployment test.

Run (from the project root):
    python experiments/05_indian_pines_all_techniques.py
    python experiments/05_indian_pines_all_techniques.py --skip-smacof   # quick
    python experiments/05_indian_pines_all_techniques.py --cube PATH --gt PATH
Output: results/indian_pines_<technique>.json, results/indian_pines_all.json,
        visualisations/05_indian_pines_embeddings.png
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for sub in ["", "01_Isomap", "02_IsomapSMACOF", "03_KernelPCA", "04_LLE"]:
    sys.path.insert(0, str(PROJECT_ROOT / sub) if sub else str(PROJECT_ROOT))

from src.evaluation import summarize
from src.indian_pines_loader import load_indian_pines
from src.preprocessing import standardize
from isomap import fit_isomap_pixels
from isomap_smacof import fit_isomap_smacof_pixels
from kernel_pca import explained_variance_ratio, fit_kernel_pca_pixels
from lle import fit_lle_pixels

SAMPLE_SIZE = 8000
SMACOF_SAMPLE_SIZE = 2000
N_NEIGHBORS = 12
N_COMPONENTS = 3
RANDOM_SEED = 42
BLOCK = 29  # 145 / 5 -> 5x5 = 25 spatial blocks
KNN_K = 5
RESULTS_DIR = PROJECT_ROOT / "results"
VIS_DIR = PROJECT_ROOT / "visualisations"


def classification_scores(X, y, groups):
    """5-NN accuracy under random-stratified and spatial-block 5-fold CV."""
    out = {}
    clf = KNeighborsClassifier(n_neighbors=KNN_K)
    schemes = {
        "random_5fold": (StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED), None),
        "spatial_block_5fold": (GroupKFold(5), groups),
    }
    for name, (cv, grp) in schemes.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tiny classes < 5 members
            pred = cross_val_predict(clf, X, y, cv=cv, groups=grp)
        cm = confusion_matrix(y, pred, labels=np.unique(y))
        per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        out[name] = {
            "overall_accuracy": float((pred == y).mean()),
            "average_class_accuracy": float(per_class.mean()),
            "kappa": float(cohen_kappa_score(y, pred)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", default=str(PROJECT_ROOT / "data" / "hsi" / "Indian_pines_corrected.mat"))
    ap.add_argument("--gt", default=str(PROJECT_ROOT / "data" / "hsi" / "Indian_pines_gt.mat"))
    ap.add_argument("--skip-smacof", action="store_true")
    args = ap.parse_args()

    for p in (args.cube, args.gt):
        if not Path(p).exists():
            print(f"Not found: {p}\nPut both Indian Pines .mat files in data/hsi/ "
                  f"or pass --cube/--gt.")
            return

    ip = load_indian_pines(args.cube, args.gt, labelled_only=True)
    n_valid = ip["pixels"].shape[0]
    print(f"Loaded {n_valid} labelled pixels, {ip['n_bands']} bands, grid {ip['grid_shape']}.")

    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(n_valid)
    idx_main = np.sort(order[:SAMPLE_SIZE])
    idx_smacof = np.sort(order[:SMACOF_SAMPLE_SIZE])  # nested subset of the main sample

    def prep(idx):
        pix, _, _ = standardize(ip["pixels"][idx])
        groups = (ip["row"][idx] // BLOCK) * 5 + (ip["col"][idx] // BLOCK)
        return pix, ip["labels"][idx], groups

    X, y, g = prep(idx_main)
    Xs, ys, gs = prep(idx_smacof)

    jobs = []  # (name, X, y, groups, runner)
    jobs.append(("PCA baseline", X, y, g, "pca"))
    jobs.append(("Isomap", X, y, g, "isomap"))
    if not args.skip_smacof:
        jobs.append(("Isomap with SMACOF", Xs, ys, gs, "smacof"))
    jobs.append(("Kernel PCA", X, y, g, "kpca"))
    jobs.append(("LLE", X, y, g, "lle"))

    embeddings, all_results = {}, {}
    # reference: classification on the full 200 bands (upper reference, not a reduction)
    raw_cls = classification_scores(X, y, g)
    all_results["Raw 200 bands (reference)"] = {"n_pixels_used": int(len(y)), "classification": raw_cls}
    print(f"\nReference, 5-NN on all 200 standardised bands: {raw_cls}")

    for name, Xi, yi, gi, kind in jobs:
        print(f"\n=== {name} ({len(yi)} pixels) ===")
        extra = {}
        if kind == "pca":
            import time, tracemalloc
            tracemalloc.start(); t0 = time.perf_counter()
            pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
            emb = pca.fit_transform(Xi)
            diag = {"fit_time_seconds": time.perf_counter() - t0,
                    "peak_memory_mb": tracemalloc.get_traced_memory()[1] / 1e6}
            tracemalloc.stop()
            metrics = summarize(Xi, emb, model=None, diagnostics=diag, n_neighbors=N_NEIGHBORS)
            metrics["explained_variance_ratio"] = float(pca.explained_variance_ratio_.sum())
        elif kind == "isomap":
            emb, model, diag = fit_isomap_pixels(Xi, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS,
                                                  random_state=RANDOM_SEED)
            metrics = summarize(Xi, emb, model=model, diagnostics=diag, n_neighbors=N_NEIGHBORS)
        elif kind == "smacof":
            emb, info, diag = fit_isomap_smacof_pixels(Xi, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS,
                                                       random_state=RANDOM_SEED)
            metrics = summarize(Xi, emb, model=None, diagnostics=diag, n_neighbors=N_NEIGHBORS)
            mds = info["mds_model"]
            extra = {"smacof_n_iter": int(mds.n_iter_), "smacof_stress": float(mds.stress_),
                     "smacof_max_iter": 300, "smacof_converged": bool(mds.n_iter_ < 300)}
        elif kind == "kpca":
            emb, model, diag = fit_kernel_pca_pixels(Xi, n_components=N_COMPONENTS, kernel="rbf",
                                                      gamma=None, random_state=RANDOM_SEED)
            metrics = summarize(Xi, emb, model=None, diagnostics=diag, n_neighbors=N_NEIGHBORS)
            metrics["explained_variance_ratio"] = explained_variance_ratio(
                Xi, n_components=N_COMPONENTS, kernel="rbf", gamma=None, random_state=RANDOM_SEED)
            extra = {"gamma_used": diag["gamma_used"]}
        else:
            emb, model, diag = fit_lle_pixels(Xi, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS,
                                               reg=1e-3, random_state=RANDOM_SEED)
            metrics = summarize(Xi, emb, model=model, diagnostics=diag, n_neighbors=N_NEIGHBORS)

        cls = classification_scores(emb, yi, gi)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"  classification: {cls}")
        embeddings[name] = (emb, yi)
        all_results[name] = {
            "technique": name, "dataset": "indian_pines", "n_pixels_used": int(len(yi)),
            "n_labelled_pixels_available": int(n_valid), "n_bands": int(ip["n_bands"]),
            "n_neighbors": N_NEIGHBORS, "n_components": N_COMPONENTS, "random_seed": RANDOM_SEED,
            "metrics": metrics, "classification_5nn": cls, **extra,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)
    for name, res in all_results.items():
        if name.startswith("Raw"):
            continue
        slug = name.lower().replace(" ", "_")
        (RESULTS_DIR / f"indian_pines_{slug}.json").write_text(json.dumps(res, indent=2))
    (RESULTS_DIR / "indian_pines_all.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {RESULTS_DIR}")

    # figure: first two embedding dims coloured by class
    fig, axes = plt.subplots(1, len(embeddings), figsize=(4.2 * len(embeddings), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, (emb, yi)) in zip(axes, embeddings.items()):
        ax.scatter(emb[:, 0], emb[:, 1], c=yi, cmap="tab20", s=3)
        ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Indian Pines: first two embedding dimensions, coloured by true class")
    fig.tight_layout()
    fig.savefig(VIS_DIR / "05_indian_pines_embeddings.png", dpi=140)
    print(f"Saved {VIS_DIR / '05_indian_pines_embeddings.png'}")


if __name__ == "__main__":
    main()
