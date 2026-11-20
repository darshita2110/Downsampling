"""
Run PCA (baseline), Isomap, Kernel PCA, LLE (and optionally Isomap+SMACOF) on
every hyperspectral benchmark scene found in the data folder, with identical
settings, shared preprocessing and shared evaluation code.

Scenes: Indian Pines, Salinas, Kennedy Space Center (KSC), Botswana, and Pavia
University (if its image cube PaviaU.mat is present). Scenes without ground
truth (Botswana here) get neighbourhood metrics only - no classification.

Settings (same as Raichur): up to 8,000 random pixels per scene (all of them
if the scene has fewer labelled pixels, e.g. KSC), k = 12, 3 components,
seed 42, Kernel PCA gamma = 1/n_bands. SMACOF (2,000-pixel nested subset) is
slow, so it is OFF by default; pass --with-smacof to include it.

Run from the project root:
    python experiments/07_benchmarks_all_techniques.py
    python experiments/07_benchmarks_all_techniques.py --data-dir "C:\\path\\to\\mat_files"
    python experiments/07_benchmarks_all_techniques.py --datasets salinas ksc --with-smacof
Output: results/benchmarks/<dataset>_all.json and
        visualisations/07_<dataset>_embeddings.png
"""
import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for sub in ["", "01_Isomap", "02_IsomapSMACOF", "03_KernelPCA", "04_LLE"]:
    sys.path.insert(0, str(PROJECT_ROOT / sub) if sub else str(PROJECT_ROOT))

from src.classification import classification_scores, spatial_block_groups
from src.evaluation import summarize
from src.hsi_loader import REGISTRY, available_datasets, load_hsi
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
OUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
VIS_DIR = PROJECT_ROOT / "visualisations"


def run_dataset(name, data_dir, with_smacof):
    scene = load_hsi(name, data_dir, labelled_only=True)
    n_avail = scene["pixels"].shape[0]
    print(f"\n########## {name}: {n_avail} usable pixels, {scene['n_bands']} bands, "
          f"grid {scene['grid_shape']}, labels={'yes' if scene['has_labels'] else 'NO'} ##########")

    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(n_avail)
    n_main = min(SAMPLE_SIZE, n_avail)
    idx_main = np.sort(order[:n_main])
    idx_smacof = np.sort(order[: min(SMACOF_SAMPLE_SIZE, n_avail)])

    def prep(idx):
        X, _, _ = standardize(scene["pixels"][idx])
        y = scene["labels"][idx] if scene["has_labels"] else None
        g = spatial_block_groups(scene["row"][idx], scene["col"][idx], scene["grid_shape"]) \
            if scene["has_labels"] else None
        return X, y, g

    X, y, g = prep(idx_main)
    results = {}
    if scene["has_labels"]:
        raw = classification_scores(X, y, g, RANDOM_SEED)
        results["Raw bands (reference)"] = {"n_pixels_used": int(len(X)), "classification": raw}
        print(f"Reference 5-NN on all {scene['n_bands']} bands: {raw}")

    jobs = [("PCA baseline", X, y, g, "pca"), ("Isomap", X, y, g, "isomap")]
    if with_smacof:
        Xs, ys, gs = prep(idx_smacof)
        jobs.append(("Isomap with SMACOF", Xs, ys, gs, "smacof"))
    jobs += [("Kernel PCA", X, y, g, "kpca"), ("LLE", X, y, g, "lle")]

    embeddings = {}
    for tech, Xi, yi, gi, kind in jobs:
        print(f"\n=== {name} / {tech} ({len(Xi)} pixels) ===")
        extra = {}
        try:
            if kind == "pca":
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
                emb, info, diag = fit_isomap_smacof_pixels(Xi, n_neighbors=N_NEIGHBORS,
                                                           n_components=N_COMPONENTS, random_state=RANDOM_SEED)
                metrics = summarize(Xi, emb, model=None, diagnostics=diag, n_neighbors=N_NEIGHBORS)
                mds = info["mds_model"]
                extra = {"smacof_n_iter": int(mds.n_iter_), "smacof_max_iter": 300,
                         "smacof_converged": bool(mds.n_iter_ < 300)}
            elif kind == "kpca":
                emb, model, diag = fit_kernel_pca_pixels(Xi, n_components=N_COMPONENTS, kernel="rbf",
                                                          gamma=None, random_state=RANDOM_SEED)
                metrics = summarize(Xi, emb, model=None, diagnostics=diag, n_neighbors=N_NEIGHBORS)
                metrics["explained_variance_ratio"] = explained_variance_ratio(
                    Xi, n_components=N_COMPONENTS, kernel="rbf", gamma=None, random_state=RANDOM_SEED)
            else:
                emb, model, diag = fit_lle_pixels(Xi, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS,
                                                   reg=1e-3, random_state=RANDOM_SEED, eigen_solver="dense")
                metrics = summarize(Xi, emb, model=model, diagnostics=diag, n_neighbors=N_NEIGHBORS)
        except Exception as exc:  # report, never hide, and keep the other techniques running
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results[tech] = {"technique": tech, "dataset": name, "error": f"{type(exc).__name__}: {exc}"}
            continue

        cls = classification_scores(emb, yi, gi, RANDOM_SEED) if yi is not None else None
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        if cls:
            print(f"  classification: {cls}")
        embeddings[tech] = (emb, yi)
        results[tech] = {"technique": tech, "dataset": name, "n_pixels_used": int(len(Xi)),
                         "n_pixels_available": int(n_avail), "n_bands": int(scene["n_bands"]),
                         "n_neighbors": N_NEIGHBORS, "n_components": N_COMPONENTS,
                         "random_seed": RANDOM_SEED, "metrics": metrics,
                         "classification_5nn": cls, **extra}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{name}_all.json").write_text(json.dumps(results, indent=2))
    if embeddings:
        fig, axes = plt.subplots(1, len(embeddings), figsize=(4.2 * len(embeddings), 4.2))
        axes = np.atleast_1d(axes)
        for ax, (tech, (emb, yi)) in zip(axes, embeddings.items()):
            ax.scatter(emb[:, 0], emb[:, 1], c=yi if yi is not None else "tab:blue",
                       cmap="tab20", s=3)
            ax.set_title(tech, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name}: first two embedding dims" + (" coloured by class" if scene["has_labels"] else " (no labels)"))
        fig.tight_layout()
        fig.savefig(VIS_DIR / f"07_{name}_embeddings.png", dpi=130)
        plt.close(fig)
    print(f"Saved {OUT_DIR / (name + '_all.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "hsi"))
    ap.add_argument("--datasets", nargs="*", default=None, help=f"subset of {list(REGISTRY)}")
    ap.add_argument("--with-smacof", action="store_true")
    args = ap.parse_args()

    found = available_datasets(args.data_dir)
    wanted = args.datasets or list(REGISTRY)
    print(f"Data folder: {args.data_dir}\nFound image cubes for: {found}")
    for name in wanted:
        if name not in found:
            print(f"SKIPPING {name}: image file '{REGISTRY[name]['cube'][0]}' not found in the data folder.")
            continue
        run_dataset(name, args.data_dir, args.with_smacof)


if __name__ == "__main__":
    main()
