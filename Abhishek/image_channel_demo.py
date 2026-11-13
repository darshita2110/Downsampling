"""
image_channel_demo.py  --  "what does reducing channels actually change?"
=========================================================================
Works on ONE image file (e.g. data/processed/sentinel2/NDVI/raichur_ndvi_2026-09-17.png).

Steps
  1. inspect_image()   : print dimensions, channels, dtype, value ranges, #unique colours
  2. auto_crop_map()   : remove title / white margins / colour-bar so only the map is analysed
  3. PPCA with k = 3, 2, 1 components, then RECONSTRUCT the image from only k channels
     -> variance kept, RMSE, PSNR and an error map show exactly what is lost
  4. Recursive filter on the components, with a zoomed before/after on an edge area

Run (needs ppca_rf_pipeline.py in the same folder):
    python image_channel_demo.py --image data/processed/sentinel2/NDVI/raichur_ndvi_2026-09-17.png
    python image_channel_demo.py --folder data/processed/sentinel2        # only list dimensions of every PNG
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppca_rf_pipeline import recursive_filter   # reuse the filter from the main pipeline


# ----------------------------------------------------------------------
# 1. Dimension / channel check
# ----------------------------------------------------------------------
def inspect_image(path, verbose=True):
    """Report H, W, number of channels, dtype, per-channel stats and distinct colours."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)       # keeps alpha channel if present
    if img is None:
        raise FileNotFoundError(path)
    C = 1 if img.ndim == 2 else img.shape[2]
    H, W = img.shape[:2]
    flat = img.reshape(-1, C)
    names = {1: ["gray"], 3: ["B", "G", "R"], 4: ["B", "G", "R", "A"]}.get(C, [f"c{i}" for i in range(C)])
    info = {"file": Path(path).name, "H": H, "W": W, "channels": C, "dtype": str(img.dtype),
            "unique_colours": len(np.unique(flat, axis=0)) if flat.shape[0] < 3e7 else -1}
    if verbose:
        print(f"\n[inspect] {info['file']}\n  size      : {H} x {W} pixels  ({H * W:,} pixels)"
              f"\n  channels  : {C}  ->  shape {img.shape}, dtype {img.dtype}"
              f"\n  distinct colours: {info['unique_colours']:,}")
        for i, n in enumerate(names):
            ch = flat[:, i]
            print(f"  channel {n}: min {ch.min()}  max {ch.max()}  mean {ch.mean():.1f}")
    return img, info


def inspect_folder(folder):
    """Print a one-line size/channel summary for every PNG under a folder."""
    print(f"{'file':55s} {'H':>6s} {'W':>6s} {'ch':>3s}  dtype")
    for p in sorted(Path(folder).rglob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        C = 1 if img.ndim == 2 else img.shape[2]
        print(f"{str(p.relative_to(folder)):55s} {img.shape[0]:6d} {img.shape[1]:6d} {C:3d}  {img.dtype}")


# ----------------------------------------------------------------------
# 2. Cut out the actual map
# ----------------------------------------------------------------------
def _longest_run(mask):
    """(start, end) of the longest stretch of True values in a 1-D boolean array."""
    best, start = (0, 0), None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            best = (start, i) if i - start > best[1] - best[0] else best
            start = None
    return best


def auto_crop_map(rgb, thr=245):
    """
    Saved plots contain a title, white margin and colour-bar. The map is the biggest
    non-white block along each axis, so keep that block and drop the rest.
    """
    nonwhite = (rgb < thr).any(axis=2)
    x0, x1 = _longest_run(nonwhite.any(axis=0))              # columns: map vs colour-bar
    y0, y1 = _longest_run(nonwhite[:, x0:x1].any(axis=1))    # rows: map vs title
    print(f"[crop] map region x {x0}:{x1}, y {y0}:{y1}  ->  {y1 - y0} x {x1 - x0} px")
    return rgb[y0:y1, x0:x1]


# ----------------------------------------------------------------------
# 3. PPCA with reconstruction
# ----------------------------------------------------------------------
def ppca_fit(X, max_k):
    """Standardise, then eigen-decompose the covariance. Returns everything needed for any k <= max_k."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    eigval, eigvec = np.linalg.eigh(np.cov(Xs, rowvar=False))
    order = np.argsort(eigval)[::-1]
    return {"mu": mu, "sd": sd, "Xs": Xs, "eigval": eigval[order], "eigvec": eigvec[:, order]}


def ppca_transform(m, k):
    """
    PPCA scores Z (N,k) as in the main pipeline, plus a reconstruction of the original
    channels from only those k directions (what you'd get back after downlink).
    """
    C = m["Xs"].shape[1]
    ev, U = m["eigval"], m["eigvec"]
    sigma2 = ev[k:].mean() if k < C else 0.0                 # noise variance = mean of dropped eigenvalues
    W = U[:, :k] * np.sqrt(np.maximum(ev[:k] - sigma2, 0))
    Z = m["Xs"] @ W @ np.linalg.inv(W.T @ W + sigma2 * np.eye(k)).T
    recon = (m["Xs"] @ U[:, :k] @ U[:, :k].T) * m["sd"] + m["mu"]   # project onto k directions & undo scaling
    return Z, recon, ev[:k].sum() / ev.sum()


def metrics(orig, recon):
    """RMSE and PSNR on the 0-255 scale (higher PSNR = closer to the original)."""
    rmse = float(np.sqrt(np.mean((orig - recon) ** 2)))
    return rmse, (20 * np.log10(255 / rmse) if rmse > 0 else float("inf"))


# ----------------------------------------------------------------------
# helpers for the figure
# ----------------------------------------------------------------------
def stretch(a):
    lo, hi = np.percentile(a, (2, 98))
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def busiest_window(gray, size=200):
    """Top-left corner of the size x size window with the most edges (good place to judge the filter)."""
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    energy = cv2.boxFilter(np.hypot(gx, gy), -1, (size, size), normalize=False)
    r = size // 2
    sub = energy[r:-r, r:-r]
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return y, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--folder", help="just list dimensions of every PNG in this folder")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="manual crop (overrides auto)")
    ap.add_argument("--k_list", type=int, nargs="+", default=[3, 2, 1])
    ap.add_argument("--sigma_spatial", type=float, default=20)
    ap.add_argument("--sigma_color", type=float, default=30)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.folder:
        inspect_folder(a.folder)
    if not a.image:
        return

    img, _ = inspect_image(a.image)
    rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB) if img.ndim == 3 else np.dstack([img] * 3)
    rgb = rgb[a.crop[1]:a.crop[3], a.crop[0]:a.crop[2]] if a.crop else auto_crop_map(rgb)
    H, W, C = rgb.shape
    X = rgb.reshape(-1, C).astype(np.float32)
    print(f"[data] analysing map of {H} x {W} px with {C} channels")

    model = ppca_fit(X, C)
    results = {}
    print(f"\n{'k':>3s} {'data kept':>10s} {'variance kept':>14s} {'RMSE':>7s} {'PSNR dB':>8s}")
    for k in a.k_list:
        Z, recon, var = ppca_transform(model, k)
        recon = np.clip(recon, 0, 255)
        rmse, psnr = metrics(X, recon)
        results[k] = (Z.reshape(H, W, k), recon.reshape(H, W, C), var, rmse, psnr)
        print(f"{k:3d} {k / C * 100:9.0f}% {var * 100:13.2f}% {rmse:7.2f} {psnr:8.2f}")

    # recursive filter on the 3-component (or largest-k) cube
    kmax = max(a.k_list)
    cube = results[kmax][0]
    smoothed = recursive_filter(cube, a.sigma_spatial, a.sigma_color)

    # ---------------- figure ----------------
    ks = sorted(a.k_list, reverse=True)
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    ax[0, 0].imshow(rgb); ax[0, 0].set_title(f"Original ({C} channels)")
    err_max = max(np.abs(rgb - results[k][1]).max() for k in ks) or 1
    for j, k in enumerate(ks[:3]):
        _, rec, var, rmse, psnr = results[k]
        ax[0, j + 1].imshow(rec.astype(np.uint8))
        ax[0, j + 1].set_title(f"Rebuilt from {k} channel(s)\nvar {var * 100:.1f}%  PSNR {psnr:.1f} dB")
        err = np.abs(rgb.astype(np.float32) - rec).mean(axis=2)
        ax[1, j + 1].imshow(err, cmap="magma", vmin=0, vmax=err_max / 2); ax[1, j + 1].set_title(f"Error map, k={k} (bright = lost)")
    ax[1, 0].axis("off")
    for i in range(min(3, kmax)):
        ax[2, i].imshow(stretch(cube[:, :, i]), cmap="gray"); ax[2, i].set_title(f"PPCA component {i + 1}")
    ax[2, 3].imshow(np.dstack([stretch(smoothed[:, :, i]) for i in range(min(3, kmax))]) if kmax >= 3 else stretch(smoothed[:, :, 0]),
                    cmap="gray")
    ax[2, 3].set_title("Smoothed components\n(as RGB)" if kmax >= 3 else "Smoothed component 1")

    y, x = busiest_window(cube[:, :, 0].astype(np.float32))
    s = 200
    sl = (slice(y, y + s), slice(x, x + s))
    ax[3, 0].imshow(stretch(cube[:, :, 0])[sl], cmap="gray"); ax[3, 0].set_title("Zoom: comp 1 BEFORE filter")
    ax[3, 1].imshow(stretch(smoothed[:, :, 0])[sl], cmap="gray"); ax[3, 1].set_title("Zoom: comp 1 AFTER recursive filter")
    ax[3, 2].imshow(np.abs(cube[:, :, 0] - smoothed[:, :, 0])[sl], cmap="magma"); ax[3, 2].set_title("What the filter removed")
    ax[3, 3].axis("off")
    for aa in ax.ravel():
        aa.set_xticks([]); aa.set_yticks([])
    fig.suptitle(f"{Path(a.image).name}: {C} -> {ks} channels", fontsize=15)
    fig.tight_layout()

    out = Path(a.out); out.mkdir(exist_ok=True)
    stem = Path(a.image).stem
    fig.savefig(out / f"{stem}_channel_demo.png", dpi=120); plt.close(fig)
    np.save(out / f"{stem}_smoothed_k{kmax}.npy", smoothed)
    print(f"\n[out] {out / (stem + '_channel_demo.png')}\n[done] returned cube shape {smoothed.shape}")
    return smoothed


if __name__ == "__main__":
    main()