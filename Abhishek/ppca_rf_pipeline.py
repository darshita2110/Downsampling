"""
PPCA + Recursive Filter pipeline for Sentinel-2 (Raichur) data
================================================================
Goal  : reduce the number of channels (e.g. 6 -> 3) BEFORE downlinking,
        while keeping spatial edges, so it can be compared with the papers.

Pipeline
    1. Load table  (x, y, date, B02, B03, B04, B08, NDVI, NDWI, ...)
    2. Pick one date and rebuild an image cube  (H, W, C)
    3. Standardise channels (zero mean, unit variance)
    4. PPCA   : C channels -> k components  (k = 3 by default)
    5. Reshape (N, k) -> (H, W, k)
    6. Recursive Filter (domain transform, cv2.ximgproc.dtFilter) per component
    7. Save/visualise reduced + smoothed cube

Run (from the project folder, conda env active):
    python ppca_rf_pipeline.py --input data/processed/ML_Dataset/combined/raichur_features.parquet
Install:
    pip install numpy pandas matplotlib pyarrow opencv-contrib-python
(NOTE: needs opencv-contrib-python, NOT plain opencv-python, for cv2.ximgproc)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                     # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Channels we try to use. Any that are empty (e.g. LST right now) are dropped automatically.
CANDIDATE_FEATURES = ["B02", "B03", "B04", "B08", "NDVI", "NDWI", "LST"]


# ----------------------------------------------------------------------
# Step 1: load data
# ----------------------------------------------------------------------
def load_table(path):
    """Read a .parquet or .csv file into a DataFrame."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def pick_features(df, min_valid=0.5):
    """Keep only the candidate channels that exist and are mostly non-empty."""
    feats = [c for c in CANDIDATE_FEATURES
             if c in df.columns and df[c].notna().mean() >= min_valid]
    print(f"[data] usable channels ({len(feats)}): {feats}")
    return feats


# ----------------------------------------------------------------------
# Step 2: table -> image cube
# ----------------------------------------------------------------------
def table_to_cube(df, features, date=None):
    """
    Each row = one pixel (x, y). We rebuild the image grid for ONE date.
    Returns cube (H, W, C) with NaN where a pixel is missing, plus the date used.
    """
    if "date" in df.columns:
        dates = sorted(df["date"].astype(str).unique())
        date = str(date) if date else dates[0]
        print(f"[data] dates available: {dates} -> using {date}")
        df = df[df["date"].astype(str) == date]

    # unique sorted x / y values become column / row indices of the image
    xs, col = np.unique(df["x"].to_numpy(), return_inverse=True)
    ys, row = np.unique(df["y"].to_numpy(), return_inverse=True)
    H, W = len(ys), len(xs)

    cube = np.full((H, W, len(features)), np.nan, dtype=np.float32)
    for k, f in enumerate(features):
        cube[row, col, k] = df[f].to_numpy(dtype=np.float32)
    cube = cube[::-1]                     # y grows northwards, image rows grow downwards
    print(f"[data] cube shape (H, W, C) = {cube.shape}")
    return np.ascontiguousarray(cube), date


# ----------------------------------------------------------------------
# Step 3 + 4: PPCA
# ----------------------------------------------------------------------
def ppca_reduce(cube, n_components=3):
    """
    Probabilistic PCA (Tipping & Bishop 1999), closed-form maximum likelihood.
        x = W z + mu + noise,   noise ~ N(0, sigma^2 I)
        W      = U_k * sqrt(L_k - sigma^2)   (top-k eigenvectors of covariance)
        sigma^2= mean of the discarded eigenvalues
        z      = (W^T W + sigma^2 I)^-1 W^T (x - mu)   <- reduced data
    Returns reduced array (H, W, k) and a dict with model info.
    """
    H, W_, C = cube.shape
    X = cube.reshape(-1, C).astype(np.float64)          # (N, C): N = H*W pixels

    # fill missing values with the channel mean (so PPCA can run)
    nan_mask = np.isnan(X)
    col_mean = np.nanmean(X, axis=0)
    X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    # standardise so big-valued channels (reflectance) don't dominate small ones (NDVI)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd

    # eigen-decomposition of the C x C covariance matrix (small, so it's fast)
    cov = np.cov(Xs, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]                    # largest first
    eigval, eigvec = eigval[order], eigvec[:, order]

    k = min(n_components, C)
    sigma2 = eigval[k:].mean() if k < C else 0.0        # noise variance
    Wm = eigvec[:, :k] * np.sqrt(np.maximum(eigval[:k] - sigma2, 0))
    M = Wm.T @ Wm + sigma2 * np.eye(k)
    Z = Xs @ Wm @ np.linalg.inv(M).T                    # (N, k) latent scores

    info = {"explained_variance_ratio": eigval[:k] / eigval.sum(),
            "sigma2": sigma2, "W": Wm}
    print(f"[ppca] {C} -> {k} channels | variance kept = "
          f"{info['explained_variance_ratio'].sum() * 100:.1f}% | noise sigma^2 = {sigma2:.4f}")

    return Z.reshape(H, W_, k).astype(np.float32), info    # Step 5: reshape to (H, W, k)


# ----------------------------------------------------------------------
# Step 6: Recursive Filter (domain transform)
# ----------------------------------------------------------------------
def to_0_255(band):
    """Scale one band to 0-255 so sigma_color has a consistent meaning. Returns band, lo, hi."""
    lo, hi = float(band.min()), float(band.max())
    return (band - lo) / (hi - lo + 1e-12) * 255.0, lo, hi


def recursive_filter(cube, sigma_spatial=20, sigma_color=30, num_iters=3):
    """
    Edge-preserving smoothing of every component with the recursive domain-transform filter.
      sigma_spatial : bigger = smoother over larger areas
      sigma_color   : bigger = edges are smoothed more (smaller keeps edges sharper)
      num_iters     : passes of the filter
    Each component is its own guide, so its edges are what gets preserved.
    """
    out = np.empty_like(cube)
    for k in range(cube.shape[2]):
        band, lo, hi = to_0_255(cube[:, :, k])
        band = band.astype(np.float32)
        filt = cv2.ximgproc.dtFilter(band, band, sigma_spatial, sigma_color,
                                     mode=cv2.ximgproc.DTF_RF, numIters=num_iters)
        out[:, :, k] = filt / 255.0 * (hi - lo) + lo       # back to the original range
    return out


# ----------------------------------------------------------------------
# Full pipeline + visualisation
# ----------------------------------------------------------------------
def ppca_rf_pipeline(cube, n_components=3, sigma_spatial=20, sigma_color=30, num_iters=3):
    """Returns (reduced_cube, smoothed_cube, info). Both cubes are (H, W, n_components)."""
    reduced, info = ppca_reduce(cube, n_components)
    smoothed = recursive_filter(reduced, sigma_spatial, sigma_color, num_iters)
    return reduced, smoothed, info


def stretch(img):
    """2-98 percentile stretch to 0-1 for display."""
    lo, hi = np.percentile(img, (2, 98))
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)


def visualise(original, reduced, smoothed, features, date, out_dir):
    """Save a figure: original RGB | each reduced component | before/after as 3-band composite."""
    k = reduced.shape[2]
    fig, ax = plt.subplots(3, k + 1, figsize=(4 * (k + 1), 11))

    # row 0: original RGB (B04,B03,B02) if available
    if all(b in features for b in ("B04", "B03", "B02")):
        rgb = np.dstack([stretch(np.nan_to_num(original[:, :, features.index(b)]))
                         for b in ("B04", "B03", "B02")])
        ax[0, 0].imshow(rgb); ax[0, 0].set_title("Original RGB")
    else:
        ax[0, 0].imshow(stretch(np.nan_to_num(original[:, :, 0])), cmap="gray")
        ax[0, 0].set_title(f"Original {features[0]}")

    # rows 0/1: each component, before and after the recursive filter
    for i in range(k):
        ax[0, i + 1].imshow(reduced[:, :, i], cmap="viridis"); ax[0, i + 1].set_title(f"PPCA comp {i + 1}")
        ax[1, i + 1].imshow(smoothed[:, :, i], cmap="viridis"); ax[1, i + 1].set_title(f"PPCA+RF comp {i + 1}")
    ax[1, 0].axis("off")

    # row 2: 3 components shown as a false-colour image
    ax[2, 0].axis("off")
    ax[2, 1].imshow(np.dstack([stretch(reduced[:, :, i]) for i in range(3)])); ax[2, 1].set_title("PPCA (3 comps as RGB)")
    ax[2, 2].imshow(np.dstack([stretch(smoothed[:, :, i]) for i in range(3)])); ax[2, 2].set_title("PPCA+RF (3 comps as RGB)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    for a in ax[2, 3:]:
        a.axis("off")
    fig.suptitle(f"Raichur {date}: {original.shape[2]} -> {k} channels", fontsize=14)
    fig.tight_layout()
    path = Path(out_dir) / f"ppca_rf_{date}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[out] figure saved -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="parquet/csv with x, y, date, bands")
    p.add_argument("--date", default=None, help="e.g. 2026-10-07 (default: first date)")
    p.add_argument("--k", type=int, default=3, help="number of output channels")
    p.add_argument("--sigma_spatial", type=float, default=20)
    p.add_argument("--sigma_color", type=float, default=30)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    df = load_table(a.input)
    print(f"[data] table columns: {list(df.columns)} | rows: {len(df)}")
    features = pick_features(df)
    cube, date = table_to_cube(df, features, a.date)

    reduced, smoothed, info = ppca_rf_pipeline(cube, a.k, a.sigma_spatial, a.sigma_color, a.iters)

    np.save(Path(a.out) / f"reduced_{date}.npy", reduced)
    np.save(Path(a.out) / f"smoothed_{date}.npy", smoothed)
    visualise(cube, reduced, smoothed, features, date, a.out)
    print(f"[done] final cube shape {smoothed.shape} (data volume {cube.shape[2]} -> {smoothed.shape[2]} channels)")
    return smoothed


if __name__ == "__main__":
    main()