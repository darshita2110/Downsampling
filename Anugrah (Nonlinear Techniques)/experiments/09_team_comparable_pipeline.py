"""
Team-comparable pipeline: reduce -> reconstruct -> score, with the SAME metrics and whole-image protocol as
Kartavya (PSNR/SSIM/MSE/RMSE/MAE on [0,1] data, 'full' protocol) and Abhishek (RMSE/PSNR/SAM, ARI/NMI,
RBF-SVM OA/AA/kappa on a random 10%-per-class split), so these rows can be put in their tables.

Why this is a MODIFICATION of the earlier runs (state this in the report):
  * Isomap, Kernel PCA and LLE only learn a map from pixels to a k-D embedding. To get reconstruction scores
    they need a decoder. Here the decoder is learnt AFTER the embedding, on the same fit sample, in two ways:
      - 'linear' : least-squares map k-D -> bands (the same kind of inverse PCA has)
      - 'knn'    : mean of the 5 nearest sample pixels in the embedding (non-linear, but the receiver must
                   store the whole sample, which is counted in the storage numbers)
    This is NOT part of the published Isomap/KPCA/LLE methods.
  * The model is fitted on a random sample (default 8,000 pixels of the WHOLE image, not only labelled ones)
    because Isomap/Kernel PCA need an n x n matrix. All pixels are then embedded with sklearn's out-of-sample
    transform(). Scores are computed on every pixel (the teammates' 'full' protocol) and ALSO on the pixels
    that were not in the fit sample ('heldout' columns) - the honest number for the 'knn' decoder.
  * Isomap with SMACOF is excluded: sklearn's MDS has no out-of-sample transform, so it cannot embed pixels
    it was not fitted on.
  * 'PCA (fit on all pixels)' follows the teammates exactly; compare it with their PCA / PPCA rows to check
    that this pipeline reproduces their numbers before trusting the non-linear rows.
Run (project root):
    python experiments/09_team_comparable_pipeline.py --data-dir data/hsi --datasets indian_pines
    python experiments/09_team_comparable_pipeline.py --data-dir PATH            # all scenes found
Output: results/team_comparable/<dataset>/{team_comparable_all,kartavya_format,abhishek_format}.csv + a PSNR plot
"""
import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for sub in ["", "01_Isomap", "03_KernelPCA", "04_LLE"]:
    sys.path.insert(0, str(PROJECT_ROOT / sub) if sub else str(PROJECT_ROOT))

from src import team_metrics as tm
from src.classification import classification_scores, spatial_block_groups
from src.evaluation import summarize
from src.hsi_loader import REGISTRY, available_datasets, load_hsi
from isomap import fit_isomap_pixels
from kernel_pca import fit_kernel_pca_pixels
from lle import fit_lle_pixels

RANDOM_SEED = 42
N_NEIGHBORS = 12
BATCH = 4000


def embed_batches(model, Z):
    """sklearn out-of-sample transform in batches (a 300k x 8k distance matrix would not fit in memory)."""
    return np.vstack([model.transform(Z[i:i + BATCH]) for i in range(0, len(Z), BATCH)])


def predict_batches(dec, F):
    return np.vstack([dec.predict(F[i:i + 20000]) for i in range(0, len(F), 20000)]).astype(np.float32)


def eig_pca(Zs):
    ev, U = np.linalg.eigh((Zs.T @ Zs) / (len(Zs) - 1))
    return U[:, np.argsort(ev)[::-1]].astype(np.float32)


def run_dataset(name, data_dir, ks, n_sample, repeats, no_ssim, merge=False):
    scene = load_hsi(name, data_dir, labelled_only=False)
    X = scene["pixels"].astype(np.float32)
    N, C = X.shape
    grid = scene["grid_shape"]
    y = scene["labels"] if scene["has_labels"] else None
    print(f"\n########## {name}: {N} pixels x {C} bands, grid {grid}, labels={'yes' if y is not None else 'NO'}")

    low, high = tm.minmax_range(X)
    mu_all, sd_all = X.mean(0), X.std(0)
    sd_all[sd_all < 1e-12] = 1.0
    Xs_all = ((X - mu_all) / sd_all).astype(np.float32)
    rng_val = tm.data_range_p99_p1(X)
    ref_clusters = tm.cluster_labels(Xs_all)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.sort(rng.choice(N, min(n_sample, N), replace=False))
    heldout = np.ones(N, bool); heldout[idx] = False
    mu_s, sd_s = X[idx].mean(0), X[idx].std(0)
    sd_s[sd_s < 1e-12] = 1.0
    Zs_all = ((X - mu_s) / sd_s).astype(np.float32)          # standardised with SAMPLE statistics
    Zs_fit = Zs_all[idx]

    splits, groups_lab = [], None
    if y is not None:
        splits = [tm.make_split(y, {0}, 0.10, 3000, 20000, seed=s) for s in range(repeats)]
        lab = np.where(y > 0)[0]
        groups_lab = spatial_block_groups(scene["row"][lab], scene["col"][lab], grid)

    rows = []

    def score(label, family, k, F, rec_raw, t_fwd, t_inv, peak_mb, model_values, emb_fit=None):
        r = {"dataset": name, "algorithm": label, "family": family, "k": k, "n_pixels": N, "n_fit_pixels": int(len(idx))}
        r.update(tm.kartavya_metrics(X, rec_raw, low, high, grid, with_ssim=not no_ssim))
        if heldout.any():
            ho = tm.kartavya_metrics(X[heldout], rec_raw[heldout], low, high, grid, with_ssim=False)
            r.update({"PSNR_dB_heldout": ho["PSNR_dB"], "RMSE_heldout": ho["RMSE"]})
        r.update(tm.storage_report(N, C, k, model_values))
        r.update(forward_seconds=t_fwd, inverse_seconds=t_inv, forward_peak_mb=peak_mb)
        r.update(tm.abhishek_recon_metrics(X, rec_raw, Xs_all, mu_all, sd_all, rng_val))
        r["ARI"], r["NMI"] = tm.ari_nmi(ref_clusters, F)
        if y is not None:
            r.update(tm.svm_classify(F, y, splits))
            c5 = classification_scores(F[lab], y[lab], groups_lab, RANDOM_SEED)
            r["knn5_OA_spatial_block"] = c5["spatial_block_5fold"]["overall_accuracy"]
            r["knn5_OA_random"] = c5["random_5fold"]["overall_accuracy"]
        if emb_fit is not None:
            s = summarize(Zs_fit, emb_fit, model=None, diagnostics={"fit_time_seconds": 0.0, "peak_memory_mb": 0.0},
                          n_neighbors=N_NEIGHBORS)
            r.update(trustworthiness=s["trustworthiness"], continuity=s["continuity"],
                     neighborhood_preservation=s["neighborhood_preservation"])
        rows.append(r)
        print(f"  {label:34s} k={k:<3d} PSNR {r['PSNR_dB']:6.2f} dB (heldout {r.get('PSNR_dB_heldout', float('nan')):6.2f}) "
              f"SSIM {r['SSIM']:.4f} SAM {r['sam_deg']:.2f} ARI {r['ARI']:.3f}"
              + (f" OA {r['OA']:.1f}%" if y is not None else ""), flush=True)

    # reference: all bands
    if y is not None:
        ref = tm.svm_classify(Xs_all, y, splits)
        rows.append({"dataset": name, "algorithm": "Full bands (reference)", "family": "reference", "k": C, **ref})
        print(f"  Full bands reference: OA {ref['OA']:.1f}% AA {ref['AA']:.1f}%")

    # ---- PCA on all pixels (reproduces the teammates) and PCA on the sample (fair comparator)
    U_all = eig_pca(Xs_all)
    U_s = eig_pca(Zs_fit)
    for k in ks:
        t0 = time.perf_counter(); F = Xs_all @ U_all[:, :k]; t_f = time.perf_counter() - t0
        t0 = time.perf_counter(); rec = (F @ U_all[:, :k].T) * sd_all + mu_all; t_i = time.perf_counter() - t0
        score("PCA (fit on all pixels)", "linear (teammates' protocol)", k, F, rec, t_f, t_i, 0.0, k * C + 3 * C)
        t0 = time.perf_counter(); F = Zs_all @ U_s[:, :k]; t_f = time.perf_counter() - t0
        t0 = time.perf_counter(); rec = (F @ U_s[:, :k].T) * sd_s + mu_s; t_i = time.perf_counter() - t0
        score("PCA (fit on sample)", "linear", k, F, rec, t_f, t_i, 0.0, k * C + 3 * C, emb_fit=F[idx])
        # control: the SAME kNN decoder on plain PCA scores, so a gain from a non-linear DECODER is not
        # mistaken for a gain from a non-linear EMBEDDING
        dec = KNeighborsRegressor(n_neighbors=5).fit(F[idx], Zs_fit)
        t0 = time.perf_counter(); rec = predict_batches(dec, F) * sd_s + mu_s; t_i = time.perf_counter() - t0
        score("PCA (fit on sample) + knn decoder", "linear embedding, non-linear decoder (control)", k, F, rec,
              t_f, t_i, 0.0, len(idx) * (k + C) + 2 * C, emb_fit=F[idx])

    # ---- non-linear methods, each with two decoders
    fitters = {
        "Isomap": lambda k: fit_isomap_pixels(Zs_fit, n_neighbors=N_NEIGHBORS, n_components=k, random_state=RANDOM_SEED),
        "Kernel PCA": lambda k: fit_kernel_pca_pixels(Zs_fit, n_components=k, kernel="rbf", gamma=None, random_state=RANDOM_SEED),
        "LLE": lambda k: fit_lle_pixels(Zs_fit, n_neighbors=N_NEIGHBORS, n_components=k, reg=1e-3, random_state=RANDOM_SEED, eigen_solver="dense"),
    }
    for k in ks:
        for method, fit in fitters.items():
            try:
                t0 = time.perf_counter(); _, model, _ = fit(k); t_fit = time.perf_counter() - t0
                tracemalloc.start(); model.transform(Zs_all[:BATCH]); peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
                t0 = time.perf_counter(); F = embed_batches(model, Zs_all).astype(np.float32); t_f = time.perf_counter() - t0
            except Exception as exc:
                print(f"  {method} k={k} FAILED: {type(exc).__name__}: {exc}")
                rows.append({"dataset": name, "algorithm": method, "k": k, "error": f"{type(exc).__name__}: {exc}"})
                continue
            for dec_name in ("linear", "knn"):
                dec = LinearRegression() if dec_name == "linear" else KNeighborsRegressor(n_neighbors=5)
                dec.fit(F[idx], Zs_fit)
                t0 = time.perf_counter(); rec = predict_batches(dec, F) * sd_s + mu_s; t_i = time.perf_counter() - t0
                mv = (k * C + 3 * C) if dec_name == "linear" else len(idx) * (k + C) + 2 * C
                score(f"{method} + {dec_name} decoder", "non-linear manifold (unsupervised)", k, F, rec,
                      t_f, t_i, peak, mv, emb_fit=F[idx])
            print(f"    ({method} k={k}: fit {t_fit:.1f}s, embed all pixels {t_f:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "team_comparable" / name
    out.mkdir(parents=True, exist_ok=True)
    if merge and (out / "team_comparable_all.csv").exists():
        # keep earlier rows for other k values; new rows replace the same (algorithm, k)
        old = pd.read_csv(out / "team_comparable_all.csv")
        new_keys = set(zip(df["algorithm"], df["k"]))
        old = old[[(a, k) not in new_keys for a, k in zip(old["algorithm"], old["k"])]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out / "team_comparable_all.csv", index=False)
    kcols = ["PSNR_dB", "SSIM", "MSE", "RMSE", "MAE", "dimensionality_reduction_pct", "storage_saving_pct",
             "compression_ratio", "forward_seconds", "inverse_seconds", "forward_peak_mb"]
    ok = df[df["SSIM"].notna()] if "SSIM" in df else df
    kf = ok[["algorithm", "family", "k"] + [c for c in kcols if c in ok]].copy()
    kf.insert(0, "protocol", "full")
    kf.insert(2, "algorithm_id", kf["algorithm"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True))
    kf.to_csv(out / "kartavya_format.csv", index=False)
    acols = ["var_explained", "rmse", "nrmse_pct", "psnr_db", "sam_deg", "ARI", "NMI", "OA", "AA", "kappa", "OA_std"]
    af = df[["algorithm", "k"] + [c for c in acols if c in df]].rename(columns={"algorithm": "method"})
    af.to_csv(out / "abhishek_format.csv", index=False)
    plot_psnr(df, out / "psnr_vs_k.png", name)
    print(f"Saved {out}")


def plot_psnr(df, path, name):
    d = df[df["PSNR_dB"].notna()] if "PSNR_dB" in df else df
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alg, g in d.groupby("algorithm"):
        g = g.sort_values("k")
        ax.plot(g["k"], g["PSNR_dB"], marker="o", label=alg, ls="--" if "knn" in alg else "-")
    ax.set_xlabel("k (reduced channels)"); ax.set_ylabel("PSNR (dB, whole image)"); ax.set_title(f"{name}: reconstruction quality vs k")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "hsi"))
    ap.add_argument("--datasets", nargs="*", default=None, help=f"subset of {list(REGISTRY)}")
    ap.add_argument("--ks", type=int, nargs="*", default=[3, 5, 10])
    ap.add_argument("--sample-size", type=int, default=8000)
    ap.add_argument("--repeats", type=int, default=5, help="random splits averaged for the SVM scores")
    ap.add_argument("--no-ssim", action="store_true", help="skip SSIM (faster)")
    ap.add_argument("--merge", action="store_true", help="add these k values to the existing CSV instead of replacing it")
    a = ap.parse_args()
    found = available_datasets(a.data_dir)
    for name in a.datasets or list(REGISTRY):
        if name not in found:
            print(f"SKIPPING {name}: image file not found in {a.data_dir}")
            continue
        run_dataset(name, a.data_dir, a.ks, a.sample_size, a.repeats, a.no_ssim, a.merge)


if __name__ == "__main__":
    main()
