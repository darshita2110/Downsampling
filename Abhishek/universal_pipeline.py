"""
universal_pipeline.py  --  PPCA + Recursive Filter for ANY (H, W, C) image cube
================================================================================
Works for RGB, multispectral (Raichur, 12 bands) and hyperspectral (Indian Pines, 200 bands).

Order of work (same for every dataset)
  1. LOAD     : cube (H, W, C) from load_data.py (or any .npy / .mat via --file)
  2. INSPECT  : dimensions, data type, ranges, NaNs, band correlation, how many
                components are really needed (written to report.txt)
  3. REDUCE   : PPCA  C -> k channels, reshape to (H, W, k)
  4. SMOOTH   : recursive filter (domain transform) on every component
  5. COMPARE  : PPCA, PPCA+RF and 3 band-selection baselines, at several k, using
                  - variance explained, RMSE / PSNR, SAM  (how well the cube can be rebuilt)
                  - classification OA / AA / kappa if labels exist, clustering ARI / NMI always
                  - noise reduction + edge retention of the recursive filter
  6. SAVE     : everything in  Abhishek/results/<dataset>/

Run from the Downsampling folder:
    python Abhishek\\universal_pipeline.py --dataset raichur --date 2026-10-12
    python Abhishek\\universal_pipeline.py --dataset indian_pines
    python Abhishek\\universal_pipeline.py --dataset both
    python Abhishek\\universal_pipeline.py --dataset file --file some_cube.npy --labels some_labels.npy
Install:  pip install numpy scipy pandas matplotlib opencv-contrib-python scikit-learn
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")      # silences a harmless joblib warning on Windows
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ppca_rf_pipeline import recursive_filter            # the filter you already have

try:                                                      # your loaders
    from load_data import RAICHUR_BANDS, load_indian_pines, load_raichur
except ImportError:
    RAICHUR_BANDS, load_raichur, load_indian_pines = None, None, None


# ======================================================================
# 1. LOADING
# ======================================================================
def load_file(path):
    """Load a cube from .npy or .mat (first 3-D array found)."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        mat = loadmat(path)
        arr = next(v for k, v in mat.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D cube (H, W, C), got shape {arr.shape}")
    if arr.shape[0] <= 32 < arr.shape[1] and arr.shape[2] > 32:   # looks like (C, H, W)
        print("[load] shape looks like (C, H, W) -> moving channels last")
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def get_dataset(a, which):
    """Returns cube, band_names (or None), short name."""
    if which == "raichur":
        return load_raichur(a.date), RAICHUR_BANDS, f"raichur_{a.date}"
    if which == "indian_pines":
        return load_indian_pines(), None, "indian_pines"
    return load_file(a.file), None, Path(a.file).stem


def align_labels(y, cube, names, min_corr=0.15):
    """
    Find where the cube sits inside a LARGER label map (the cube is a crop of the full scene).
    Method: template matching between the cube's NDVI and the label map's "vegetation" mask
    (SCL class 4). Tries the window at 1x and 0.5x scale (10 m cube vs 20 m labels).
    Returns (corr, scale, y0, x0, window_h, window_w, runner_up_corr) or None.
    """
    H, W = cube.shape[:2]
    if not names or "B04" not in names or "B08" not in names:
        return None
    nir, red = cube[:, :, names.index("B08")], cube[:, :, names.index("B04")]
    ndvi = ((nir - red) / (nir + red + 1e-6)).astype(np.float32)
    veg = (y == 4).astype(np.float32)
    best = None
    for scale in (1.0, 0.5):
        th, tw = int(round(H * scale)), int(round(W * scale))
        if th >= y.shape[0] or tw >= y.shape[1]:
            continue
        res = np.nan_to_num(cv2.matchTemplate(veg, cv2.resize(ndvi, (tw, th), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED))
        y0, x0 = np.unravel_index(np.argmax(res), res.shape)
        r2 = res.copy(); r2[max(0, y0 - 25):y0 + 26, max(0, x0 - 25):x0 + 26] = -1
        cand = (float(res[y0, x0]), scale, int(y0), int(x0), th, tw, float(r2.max()))
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def get_labels(a, which, hw, cube=None, names=None):
    """Optional label map (H, W). Raichur: SCL map if present. Indian Pines: ground-truth .mat if present."""
    path = None
    if a.labels and a.dataset != "both":
        path = Path(a.labels)
    elif which == "raichur" and Path(f"data/darshita/raichur_{a.date}_scl.npy").exists():
        path = Path(f"data/darshita/raichur_{a.date}_scl.npy")
    elif which == "indian_pines" and Path("data/darshita/Indian_pines_gt.mat").exists():
        path = Path("data/darshita/Indian_pines_gt.mat")
    if path is None:
        print("[labels] none found -> classification skipped, clustering agreement (ARI/NMI) is used instead")
        return None
    if path.suffix == ".npy":
        y = np.load(path)
    else:
        m = loadmat(path)
        y = next(v for k, v in m.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 2)
    y = np.squeeze(y).astype(np.int32)
    if y.shape != hw:
        done = False
        if a.label_window:                                   # manual crop, in label-pixel coordinates
            y0, y1, x0, x1 = a.label_window
            y = cv2.resize(y[y0:y1, x0:x1].astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            print(f"[labels] manual window y {y0}:{y1}, x {x0}:{x1} -> {hw}")
            done = True
        elif not a.no_align and cube is not None:
            best = align_labels(y, cube, names)
            if best is not None and best[0] >= 0.15:
                corr, sc, y0, x0, th, tw, ru = best
                y = cv2.resize(y[y0:y0 + th, x0:x0 + tw].astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                print(f"[labels] labels are larger than the cube -> cropped window y {y0}:{y0 + th}, x {x0}:{x0 + tw} "
                      f"(scale {sc}, NDVI-vs-vegetation correlation {corr:.2f}, next-best elsewhere {ru:.2f})")
                done = True
            elif best is not None:
                print(f"[labels] WARNING: no convincing match of the cube inside the labels (best correlation {best[0]:.2f})")
        if not done:
            print(f"[labels] WARNING: resizing the WHOLE label map {y.shape} -> {hw}; if the cube is only a crop of the "
                  "scene the labels are misaligned and classification scores are meaningless")
            y = cv2.resize(y.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    ids, cnt = np.unique(y, return_counts=True)
    print(f"[labels] {path.name}: classes {dict(zip(ids.tolist(), cnt.tolist()))}")
    return y


# ======================================================================
# 2. INSPECTION
# ======================================================================
def inspect_cube(cube, name, band_names=None):
    """Print + return a text report; also cleans NaN/inf so later steps work. Returns (clean_cube, info, text)."""
    H, W, C = cube.shape
    lines = [f"DATASET {name}", f"  shape (H, W, C)  : {cube.shape}   dtype {cube.dtype}",
             f"  pixels           : {H * W:,}   |   values: {cube.size:,}   |   memory {cube.nbytes / 1e6:.1f} MB"]
    kind = "RGB / few-band" if C <= 4 else ("multispectral" if C <= 30 else "hyperspectral")
    lines.append(f"  type             : {kind} ({C} channels)")

    bad = ~np.isfinite(cube)
    lines.append(f"  NaN / inf values : {int(bad.sum()):,} ({bad.mean() * 100:.3f}%)")
    if bad.any():                                           # replace by that band's mean
        means = np.nanmean(np.where(bad, np.nan, cube), axis=(0, 1))
        cube = np.where(bad, means[None, None, :], cube)
    X = cube.reshape(-1, C)
    lines.append(f"  value range      : {X.min():.4g} .. {X.max():.4g}   (1-99 percentile: "
                 f"{np.percentile(X[::7], 1):.4g} .. {np.percentile(X[::7], 99):.4g})")
    zero_bands = int((X.std(0) < 1e-9).sum())
    lines.append(f"  constant bands   : {zero_bands}")

    names = band_names or [f"b{i + 1}" for i in range(C)]
    lines.append("  per-band statistics" + (" (first 20 of %d)" % C if C > 20 else ""))
    lines.append(f"    {'band':>6s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s}")
    for i in range(min(C, 20)):
        b = X[:, i]
        lines.append(f"    {names[i]:>6s} {b.min():10.4g} {b.max():10.4g} {b.mean():10.4g} {b.std():10.4g}")

    sub = X[:: max(1, len(X) // 50000)]
    corr = np.corrcoef(sub, rowvar=False)
    corr = np.nan_to_num(corr)
    adj = np.mean([corr[i, i + 1] for i in range(C - 1)]) if C > 1 else 1.0
    off = np.abs(corr[~np.eye(C, dtype=bool)]).mean() if C > 1 else 1.0
    lines.append(f"  band correlation : neighbouring bands {adj:.3f} | mean |all pairs| {off:.3f}  (high = redundant = good for reduction)")

    ev = np.sort(np.linalg.eigvalsh(corr))[::-1]
    ev = np.clip(ev, 0, None)
    cum = np.cumsum(ev) / ev.sum()
    need = {p: int(np.searchsorted(cum, p) + 1) for p in (0.90, 0.95, 0.99)}
    lines.append(f"  components needed: 90% variance -> {need[0.90]}, 95% -> {need[0.95]}, 99% -> {need[0.99]}  (of {C})")
    text = "\n".join(lines)
    print("\n" + text)
    return cube, {"H": H, "W": W, "C": C, "kind": kind, "corr": corr, "cum": cum}, text


# ======================================================================
# 3. PPCA (closed-form maximum likelihood)
# ======================================================================
def ppca_fit(X):
    """Standardise channels and eigen-decompose the covariance. Reusable for every k."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Xs = ((X - mu) / sd).astype(np.float32)
    ev, U = np.linalg.eigh((Xs.T @ Xs) / (len(Xs) - 1))
    o = np.argsort(ev)[::-1]
    return {"mu": mu, "sd": sd, "Xs": Xs, "ev": ev[o], "U": U[:, o].astype(np.float32)}


def ppca_scores(m, k):
    """
    PPCA latent scores Z (N,k):   W = U_k sqrt(L_k - s2),  s2 = mean(dropped eigenvalues),
    Z = (W'W + s2 I)^-1 W' x.  Also returns the factor that maps Z back to orthonormal PCA scores.
    """
    C = m["Xs"].shape[1]
    lam = m["ev"][:k]
    s2 = m["ev"][k:].mean() if k < C else 0.0
    amp = np.sqrt(np.maximum(lam - s2, 1e-9))
    Z = (m["Xs"] @ m["U"][:, :k]) * (amp / lam)
    return Z.astype(np.float32), (lam / amp).astype(np.float32), s2


def ppca_reconstruct(m, Z, back, k):
    """Rebuild the standardised channels from k scores (what the ground station would do)."""
    return (Z * back) @ m["U"][:, :k].T


# ======================================================================
# band-selection baselines (reduce channels by KEEPING k original bands)
# ======================================================================
def select_uniform(C, k, **_):
    return np.unique(np.linspace(0, C - 1, k).round().astype(int))


def select_variance(C, k, Xs=None, X=None, **_):
    return np.argsort(X.var(0))[::-1][:k]


def select_cluster(C, k, Xs=None, **_):
    """Group-wise method from the review: cluster bands, keep the band nearest each cluster centre."""
    rng = np.random.default_rng(0)
    pts = Xs[rng.choice(len(Xs), min(len(Xs), 4000), replace=False)].T       # (C, n) one row per band
    km = KMeans(k, n_init=3, random_state=0).fit(pts)
    return np.array(sorted({int(np.argmin(((pts - c) ** 2).sum(1))) for c in km.cluster_centers_}))


def band_recon(Xs, idx):
    """Rebuild all bands from the kept ones with least squares (fair comparison with PPCA)."""
    sub = np.random.default_rng(1).choice(len(Xs), min(len(Xs), 50000), replace=False)
    A = np.c_[Xs[sub][:, idx], np.ones(len(sub))]
    coef = np.linalg.lstsq(A, Xs[sub], rcond=None)[0]
    return np.c_[Xs[:, idx], np.ones(len(Xs))] @ coef


# ======================================================================
# metrics
# ======================================================================
def sam_deg(a, b):
    """Mean spectral angle (degrees) between original and rebuilt spectra."""
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float(np.degrees(np.arccos(np.clip((a * b).sum(1) / den, -1, 1))).mean())


def recon_metrics(X, Xs, recon_s, m, rng_val):
    """variance explained (standardised), RMSE/PSNR in original units, SAM."""
    var_exp = 1 - float(((Xs - recon_s) ** 2).sum() / (Xs ** 2).sum())
    rec = recon_s * m["sd"] + m["mu"]
    rmse = float(np.sqrt(np.mean((X - rec) ** 2)))
    return {"var_explained": var_exp, "rmse": rmse, "nrmse_pct": rmse / rng_val * 100,
            "psnr_db": 20 * np.log10(rng_val / max(rmse, 1e-12)), "sam_deg": sam_deg(X, rec)}


def cluster_labels(F, n_clusters, seed=0):
    Fs = F        # features are already in the common standardised-band scale: no per-column whitening
    idx = np.random.default_rng(seed).choice(len(Fs), min(len(Fs), 20000), replace=False)
    km = MiniBatchKMeans(n_clusters, n_init=3, random_state=seed, batch_size=4096).fit(Fs[idx])
    return km.predict(Fs)


def make_split(y, ignore, frac, max_train, max_test, seed=0):
    """Same train/test pixels for every method, so the comparison is fair."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if c in ignore or len(idx) < 10:
            continue
        rng.shuffle(idx)
        n = max(3, int(round(frac * len(idx))))
        tr.append(idx[:n]); te.append(idx[n:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    if len(tr) > max_train: tr = rng.choice(tr, max_train, replace=False)
    if len(te) > max_test: te = rng.choice(te, max_test, replace=False)
    return tr, te


def make_split_blocked(y_map, ignore, block, buffer, train_block_frac, max_train, max_test, seed=0):
    """
    Spatially separated split. The image is cut into square blocks; whole blocks are train OR test.
    Pixels within `buffer` px of a block edge are dropped, so the recursive filter cannot carry
    training-pixel information into test pixels (a random pixel split would leak it).
    """
    rng = np.random.default_rng(seed)
    H, W = y_map.shape
    r, c = np.arange(H), np.arange(W)
    nbx = int(np.ceil(W / block))
    bid = (r // block)[:, None] * nbx + (c // block)[None, :]
    nb = int(bid.max()) + 1
    is_train = rng.random(nb) < train_block_frac
    is_train[rng.integers(nb)] = True                       # at least one block each way
    if is_train.all(): is_train[rng.integers(nb)] = False
    dy, dx = r % block, c % block
    edge = np.minimum(np.minimum(dy, block - 1 - dy)[:, None], np.minimum(dx, block - 1 - dx)[None, :])
    keep = (edge >= buffer).reshape(-1)
    y = y_map.reshape(-1)
    ok = keep & ~np.isin(y, list(ignore))
    tr = np.where(ok & is_train[bid.reshape(-1)])[0]
    te = np.where(ok & ~is_train[bid.reshape(-1)])[0]
    te = te[np.isin(y[te], np.unique(y[tr]))]               # only classes the model has seen
    if len(tr) > max_train: tr = rng.choice(tr, max_train, replace=False)
    if len(te) > max_test: te = rng.choice(te, max_test, replace=False)
    return tr, te


def classify(F, y, tr, te):
    """
    RBF-SVM (the classifier the papers use). Features are used AS GIVEN: every method lives in the
    same standardised-band scale, so weak (noisy) components are not artificially boosted.
    Returns ({OA, AA, kappa} in %, per-class recall in %).
    """
    clf = SVC(kernel="rbf", C=100, gamma="scale").fit(F[tr], y[tr])
    pred = clf.predict(F[te])
    rec = {int(c): 100 * float(np.mean(pred[y[te] == c] == c)) for c in np.unique(y[te])}
    return {"OA": 100 * float(np.mean(pred == y[te])), "AA": float(np.mean(list(rec.values()))),
            "kappa": 100 * float(cohen_kappa_score(y[te], pred))}, rec


def classify_multi(F, y, splits):
    """Run the classifier on every split (different seeds) and average. OA_std shows the spread."""
    res, recs = [], None
    for tr, te in splits:
        met, rec = classify(F, y, tr, te)
        res.append(met)
        recs = recs or rec
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    if len(res) > 1:
        out["OA_std"] = float(np.std([r["OA"] for r in res]))
    return out, recs


def spatial_stats(raw, sm):
    """Noise reduction (%) and edge retention (correlation of edge maps) of the recursive filter."""
    nr, er = [], []
    for i in range(raw.shape[2]):
        a, b = raw[:, :, i], sm[:, :, i]
        s = a.std() + 1e-12
        a, b = a / s, b / s
        mad = lambda z: np.median(np.abs(cv2.Laplacian(z, cv2.CV_32F) - np.median(cv2.Laplacian(z, cv2.CV_32F)))) * 1.4826
        nr.append((1 - mad(b) / (mad(a) + 1e-12)) * 100)
        ga = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
        gb = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
        strong = ga > np.percentile(ga, 90)
        er.append(np.corrcoef(ga[strong], gb[strong])[0, 1])
    return float(np.mean(nr)), float(np.mean(er))


# ======================================================================
# display helpers
# ======================================================================
def stretch(a):
    lo, hi = np.percentile(a[::3, ::3], (2, 98))
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def display_rgb(cube, names):
    """True colour for Sentinel-2 (B04,B03,B02); otherwise bands at 15%, 10%, 5% of the spectrum."""
    C = cube.shape[2]
    if names and all(b in names for b in ("B04", "B03", "B02")):
        idx = [names.index(b) for b in ("B04", "B03", "B02")]
        title = "Original (true colour B04/B03/B02)"
    else:
        idx = [int(.15 * C), int(.10 * C), int(.05 * C)]
        title = f"Original (false colour, bands {idx})"
    return np.dstack([stretch(cube[:, :, i]) for i in idx]), title


def composite(c):
    return np.dstack([stretch(c[:, :, i]) if i < c.shape[2] else np.zeros(c.shape[:2], np.float32) for i in range(3)])


def busiest_window(gray, size):
    g = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    e = cv2.boxFilter(g, -1, (size, size), normalize=False)
    r = size // 2
    y, x = np.unravel_index(np.argmax(e[r:-r, r:-r]), e[r:-r, r:-r].shape)
    return int(y), int(x)


# ======================================================================
# main per-dataset run
# ======================================================================
def build_splits(a, y, y_map, H, W, sig_sp):
    """One train/test split per repeat (seeds 0..repeats-1). Returns (list of (train_idx, test_idx), description)."""
    ign = set(a.ignore_labels)
    if a.split == "auto":
        a.split = "blocked" if min(H, W) >= 256 else "random"
        print(f"[split] auto -> {a.split} (image {H}x{W}; blocked splits need a large image with many separate fields)")
    if a.rf_sweep:                                            # filter sizes must be small compared with the image
        a.sweep_spatial = [v for v in a.sweep_spatial if v <= min(H, W) / 12] or [3.0]
    desc, splits = "", []
    if a.split == "blocked":
        buf = a.buffer if a.buffer is not None else int(np.ceil(max([sig_sp] + (list(a.sweep_spatial) if a.rf_sweep else []))))
        block = max(a.block or max(24, min(H, W) // 6), 4 * buf)
        splits = [make_split_blocked(y_map, ign, block, buf, a.train_block_frac, a.max_train, a.max_test, seed=s) for s in range(a.repeats)]
        desc = f"blocked ({block}px blocks, {buf}px buffer, {a.train_block_frac:.0%} of blocks for training)"
        if any(len(tr) == 0 or len(te) == 0 for tr, te in splits):
            print("[split] WARNING: blocked split left no train or test pixels (image too small) -> using a random split instead")
            a.split = "random"
    if a.split == "random":
        splits = [make_split(y, ign, a.train_frac, a.max_train, a.max_test, seed=s) for s in range(a.repeats)]
        desc = f"random pixels ({a.train_frac:.0%} per class; OPTIMISTIC for PPCA+RF because the filter mixes neighbouring pixels)"
    return splits, desc


def run(a, which):
    a = argparse.Namespace(**vars(a))                      # private copy: settings below are per dataset
    cube, names, name = get_dataset(a, which)
    out = HERE / "results" / (name + (f"_{a.tag}" if a.tag else ""))   # --tag keeps runs from overwriting each other
    out.mkdir(parents=True, exist_ok=True)

    # ---- 2. inspect ----
    cube, info, report = inspect_cube(cube, name, names)
    H, W, C = cube.shape
    X = cube.reshape(-1, C).astype(np.float32)
    rng_val = float(np.percentile(X[::7], 99) - np.percentile(X[::7], 1))

    # ---- choose k values from the data type ----
    default_k = [1, 2] if C <= 4 else ([2, 3, 4, 6] if C <= 30 else [3, 5, 10, 20])
    ks = sorted({k for k in (a.k_list or default_k) + [a.primary_k] if k < C})
    sig_sp = a.sigma_spatial or max(3.0, round(min(H, W) / 80))
    report += f"\n  settings         : k values {ks}, primary k {a.primary_k}, sigma_spatial {sig_sp}, sigma_color {a.sigma_color}"

    # ---- labels / task ----
    y_map = get_labels(a, which, (H, W), cube, names)
    y = y_map.reshape(-1) if y_map is not None else None
    splits, maj = [], None
    if y is not None:
        splits, desc = build_splits(a, y, y_map, H, W, sig_sp)
        split = splits[0]
        ids, cnt = np.unique(y[split[1]], return_counts=True)
        n_cls = len(ids)
        n_all = len([c for c in np.unique(y) if c not in set(a.ignore_labels) and (y == c).sum() >= 10])
        maj = 100 * cnt.max() / len(split[1])
        msg = (f"[split] {a.split}: train {len(split[0]):,} px, test {len(split[1]):,} px, {n_cls} of {n_all} classes in the test set, "
               f"majority-class baseline OA = {maj:.1f}%, {a.repeats} repeat(s)")
        print(msg); report += f"\n  split            : {desc}\n  " + msg
        if n_cls < 0.6 * n_all:
            print(f"[split] WARNING: only {n_cls} of {n_all} classes are in the test set -> classification scores will be unstable")

    # ---- 3. fit PPCA once ----
    m = ppca_fit(X)
    Xs = m["Xs"]
    ref_clusters = cluster_labels(Xs, a.n_clusters)      # reference = clusters of the full standardised cube
    rows, kept = [], {}

    def evaluate(method, k, feats, recon_s):
        r = {"method": method, "k": k, **recon_metrics(X, Xs, recon_s, m, rng_val)}
        cl = cluster_labels(feats, a.n_clusters)
        r["ARI"], r["NMI"] = adjusted_rand_score(ref_clusters, cl), normalized_mutual_info_score(ref_clusters, cl)
        if y is not None:
            r.update(classify_multi(feats, y, splits)[0])
        rows.append(r)
        print(f"  {method:16s} k={k:<3d} var {r['var_explained'] * 100:6.2f}%  RMSE {r['rmse']:.4g}  SAM {r['sam_deg']:.2f}  "
              + (f"OA {r['OA']:.1f}%" if y is not None else f"ARI {r['ARI']:.3f}"))

    print("\n[compare] evaluating methods (this is the slow part)...")
    ref_recalls = None
    if y is not None:                                            # reference: all bands
        met, ref_recalls = classify_multi(Xs, y, splits)
        ref = {"method": "Full bands", "k": C, "var_explained": 1.0, "rmse": 0.0, "nrmse_pct": 0.0,
               "psnr_db": np.inf, "sam_deg": 0.0, "ARI": 1.0, "NMI": 1.0, **met}
        rows.append(ref)
        print(f"  {'Full bands':16s} k={C:<3d} OA {ref['OA']:.1f}%  (upper reference)")

    for k in ks:
        Z, back, s2 = ppca_scores(m, k)
        evaluate("PPCA", k, Z * back, ppca_reconstruct(m, Z, back, k))
        Zs = recursive_filter(Z.reshape(H, W, k), sig_sp, a.sigma_color, a.iters)
        Zsf = Zs.reshape(-1, k)
        evaluate("PPCA+RF", k, Zsf * back, ppca_reconstruct(m, Zsf, back, k))
        for label, fn in (("Band-uniform", select_uniform), ("Band-variance", select_variance), ("Band-cluster", select_cluster)):
            idx = fn(C, k, Xs=Xs, X=X)
            evaluate(label, len(idx), Xs[:, idx], band_recon(Xs, idx))
        if k == a.primary_k:
            kept = {"Z": Z.reshape(H, W, k), "Zs": Zs, "s2": s2, "back": back}

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    if y is not None:
        diag = ("\n\nLABEL DIAGNOSTICS (full-band classifier, per-class recall % on the test pixels)\n  "
                + ", ".join(f"class {c}: {v:.0f}%" for c, v in ref_recalls.items())
                + f"\n  majority-class baseline OA {maj:.1f}%  |  full-band OA {df.OA.iloc[0]:.1f}%  AA {df.AA.iloc[0]:.1f}%  kappa {df.kappa.iloc[0]:.1f}%"
                + "\n  If full-band OA is close to the baseline and AA/kappa are tiny, the labels do not follow the pixels "
                  "(misaligned, wrong date, or too coarse) and OA says nothing about the reduction.")
        if df.kappa.iloc[0] < 20:
            why = (f"only {n_cls} of {n_all} classes ended up in the test set, so the scores are unstable (the labels may be fine: "
                   "check fig4_labels.png)" if n_cls < 0.6 * n_all else
                   "the labels may not follow the pixels (misaligned, wrong date, too coarse)")
            diag += (f"\n  >>> CLASSIFICATION SCORES UNRELIABLE: full-band kappa is below 20% -- {why}. Do not use OA/AA/kappa "
                     "to compare methods; rely on reconstruction metrics (variance, RMSE, SAM).")
        if names and "B04" in names and "B08" in names:
            nir, red = cube[:, :, names.index("B08")], cube[:, :, names.index("B04")]
            nd = (nir - red) / (nir + red + 1e-6)
            diag += "\n  mean NDVI per label class (SCL: 4 vegetation should be highest, 6 water lowest): " + ", ".join(
                f"{int(c)}: {nd[y_map == c].mean():.2f}" for c in np.unique(y_map) if c not in a.ignore_labels and (y_map == c).sum() > 0)
        print(diag); report += diag

    # ---- spatial effect of the recursive filter ----
    noise_red, edge_keep = spatial_stats(kept["Z"], kept["Zs"])
    report += (f"\n\nRESULT (primary k={a.primary_k}): variance explained by PPCA "
               f"{df[(df.method == 'PPCA') & (df.k == a.primary_k)].var_explained.iloc[0] * 100:.2f}%, "
               f"PPCA noise sigma^2 {kept['s2']:.4f}\n"
               f"recursive filter: fine detail (noise + texture) removed {noise_red:.1f}%, edge retention (correlation) {edge_keep:.3f}\n\n"
               f"METRICS\n{df.to_string(index=False, float_format=lambda v: f'{v:.4g}')}")
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(f"\n[spatial] recursive filter: fine detail (noise + texture) removed {noise_red:.1f}% | edge retention {edge_keep:.3f}")

    # ---- 6. save arrays + images ----
    pk = a.primary_k
    np.save(out / f"ppca_k{pk}.npy", kept["Z"])
    np.save(out / f"ppca_rf_k{pk}.npy", kept["Zs"])
    for tag, c in (("ppca", kept["Z"]), ("ppca_rf", kept["Zs"])):
        cv2.imwrite(str(out / f"{tag}_k{pk}_composite.png"), (composite(c)[:, :, ::-1] * 255).astype(np.uint8))
    make_figures(a, out, name, names, cube, info, m, df, kept, ks, noise_red, edge_keep, sig_sp, y is not None, maj)
    if y is not None:
        fig_labels(out, names, cube, y_map, splits[0], set(a.ignore_labels))
    if a.rf_sweep:
        rf_sweep(a, out, kept, m, X, Xs, rng_val, H, W, y, splits, ref_clusters)
    print(f"[saved] {out}\n  report.txt, metrics.csv, ppca_k{pk}.npy, ppca_rf_k{pk}.npy, composites, fig1/2/3 PNGs (+ fig4_labels, rf_sweep if available)")
    return kept["Zs"]


# ======================================================================
# figures
# ======================================================================
def make_figures(a, out, name, names, cube, info, m, df, kept, ks, noise_red, edge_keep, sig_sp, has_labels, maj=None):
    pk, H, W, C = a.primary_k, *cube.shape
    Z, Zs = kept["Z"], kept["Zs"]

    # ---- fig 1: what the reduced channels look like ----
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    rgb, title = display_rgb(cube, names)
    ax[0, 0].imshow(rgb); ax[0, 0].set_title(title)
    ax[0, 1].imshow(composite(Z)); ax[0, 1].set_title(f"PPCA: {C} -> {pk} channels (comps 1-3 as RGB)")
    ax[0, 2].imshow(composite(Zs)); ax[0, 2].set_title("PPCA + Recursive Filter")
    for i in range(3):
        if i < pk:
            ax[1, i].imshow(stretch(Zs[:, :, i]), cmap="gray"); ax[1, i].set_title(f"Smoothed component {i + 1}")
        else:
            ax[1, i].axis("off")
    size = max(40, min(H, W) // 4)
    y0, x0 = busiest_window(Z[:, :, 0].astype(np.float32), size)
    sl = (slice(y0, y0 + size), slice(x0, x0 + size))
    ax[2, 0].imshow(stretch(Z[:, :, 0])[sl], cmap="gray"); ax[2, 0].set_title("Zoom comp 1: before filter")
    ax[2, 1].imshow(stretch(Zs[:, :, 0])[sl], cmap="gray"); ax[2, 1].set_title("Zoom comp 1: after filter")
    ax[2, 2].imshow(np.abs(Z[:, :, 0] - Zs[:, :, 0])[sl], cmap="magma"); ax[2, 2].set_title("Removed by filter (noise + fine detail)")
    for z in ax.ravel(): z.set_xticks([]); z.set_yticks([])
    fig.suptitle(f"{name}: {C} -> {pk} channels | filter: fine detail -{noise_red:.0f}%, edge retention {edge_keep:.2f}", fontsize=14)
    fig.tight_layout(); fig.savefig(out / "fig1_overview.png", dpi=110); plt.close(fig)

    # ---- fig 2: data analysis ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    cum = info["cum"][:min(C, 40)]
    ax[0].plot(range(1, len(cum) + 1), cum * 100, "o-", ms=3)
    for k in ks: ax[0].axvline(k, color="grey", ls=":", lw=.8)
    ax[0].axhline(99, color="r", ls="--", lw=.8)
    ax[0].set_xlabel("components kept"); ax[0].set_ylabel("cumulative variance (%)"); ax[0].set_title("How many components are needed?")
    im = ax[1].imshow(info["corr"], cmap="coolwarm", vmin=-1, vmax=1); ax[1].set_title("Band correlation (red = redundant)")
    fig.colorbar(im, ax=ax[1], fraction=.046)
    w = (m["U"][:, :pk] ** 2 * m["ev"][:pk]).sum(1); w = w / w.sum()
    ax[2].bar(range(C), w * 100); ax[2].set_xlabel("band index" + (" (" + ", ".join(names) + ")" if names else ""), fontsize=7)
    ax[2].set_ylabel("share of first %d components (%%)" % pk); ax[2].set_title("Which bands drive the components?")
    fig.tight_layout(); fig.savefig(out / "fig2_data_analysis.png", dpi=110); plt.close(fig)

    # ---- fig 3: comparison of methods ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    task, tlabel = ("OA", "Classification OA (%)") if has_labels else ("ARI", "Clustering agreement with full bands (ARI)")
    methods = [mm for mm in df.method.unique() if mm != "Full bands"]
    for mm in methods:
        d = df[df.method == mm].sort_values("k")
        style = dict(marker="o", lw=2.5 if mm.startswith("PPCA") else 1.3)
        ax[0, 0].plot(d.k, d.nrmse_pct, label=mm, **style)
        ax[0, 1].plot(d.k, d.sam_deg, label=mm, **style)
        if task == "OA" and "OA_std" in d.columns and d.OA_std.notna().any():
            ax[1, 0].errorbar(d.k, d[task], yerr=d.OA_std, capsize=3, label=mm, **style)
        else:
            ax[1, 0].plot(d.k, d[task], label=mm, **style)
    if has_labels:
        ax[1, 0].axhline(df[df.method == "Full bands"].OA.iloc[0], color="k", ls="--", label="all bands")
        ax[1, 0].axhline(maj, color="r", ls=":", label="majority-class baseline")
    ax[0, 0].set_title("Reconstruction error (lower = better)"); ax[0, 0].set_ylabel("NRMSE (% of data range)")
    ax[0, 1].set_title("Spectral angle error (lower = better)"); ax[0, 1].set_ylabel("SAM (degrees)")
    ax[1, 0].set_title(f"Task quality (higher = better)"); ax[1, 0].set_ylabel(tlabel)
    for z in (ax[0, 0], ax[0, 1], ax[1, 0]):
        z.set_xlabel("channels kept (k)"); z.legend(fontsize=8); z.grid(alpha=.3)
    ax[1, 1].bar(["fine detail removed\n(%)", "edge retention\n(x100)"], [noise_red, edge_keep * 100], color=["tab:blue", "tab:green"])
    ax[1, 1].set_title(f"Recursive filter effect (k={pk}, sigma_spatial={sig_sp})"); ax[1, 1].set_ylim(0, 105)
    fig.suptitle(f"{name}: method comparison", fontsize=14)
    fig.tight_layout(); fig.savefig(out / "fig3_comparison.png", dpi=110); plt.close(fig)


def fig_labels(out, names, cube, y_map, split, ignore):
    """Sanity figure: do the labels line up with the image, and which pixels were used for train / test?"""
    H, W = y_map.shape
    rgb, _ = display_rgb(cube, names)
    ids = [int(c) for c in np.unique(y_map) if c not in ignore]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.5))
    ax[0].imshow(rgb); ax[0].set_title("Original")
    ax[1].imshow(np.ma.masked_where(np.isin(y_map, list(ignore)), y_map), cmap="tab20", interpolation="nearest")
    ax[1].set_title("Label map (do labels follow the image?)")
    sm = np.zeros(H * W, np.uint8); sm[split[0]] = 1; sm[split[1]] = 2
    ax[2].imshow(sm.reshape(H, W), cmap=ListedColormap(["#cccccc", "tab:blue", "tab:orange"]), interpolation="nearest")
    ax[2].set_title("Train (blue) / test (orange) / unused (grey)")
    ax[3].bar([str(c) for c in ids], [int((y_map == c).sum()) for c in ids]); ax[3].set_yscale("log")
    ax[3].set_title("Pixels per class (log scale)"); ax[3].set_xlabel("class id")
    for z in ax[:3]: z.set_xticks([]); z.set_yticks([])
    fig.tight_layout(); fig.savefig(out / "fig4_labels.png", dpi=110); plt.close(fig)


def rf_sweep(a, out, kept, m, X, Xs, rng_val, H, W, y, splits, ref_clusters):
    """Try several recursive-filter strengths at the primary k. Row sigma=0 is the unfiltered PPCA."""
    Z, back, k = kept["Z"], kept["back"], a.primary_k
    rows = []
    print("\n[rf_sweep] sigma_spatial x sigma_color at k =", k)
    combos = [(0, 0)] + [(ss, sc) for ss in a.sweep_spatial for sc in a.sweep_color]
    for ss, sc in combos:
        Zs = Z if ss == 0 else recursive_filter(Z, ss, sc, a.iters)
        flat = Zs.reshape(-1, k)
        r = {"sigma_spatial": ss, "sigma_color": sc, **recon_metrics(X, Xs, ppca_reconstruct(m, flat, back, k), m, rng_val)}
        r["noise_reduced_pct"], r["edge_retention"] = spatial_stats(Z, Zs) if ss else (0.0, 1.0)
        r["ARI"] = adjusted_rand_score(ref_clusters, cluster_labels(flat * back, a.n_clusters))
        if y is not None:
            r.update(classify_multi(flat * back, y, splits)[0])
        rows.append(r)
        print(f"  ss={ss:<5g} sc={sc:<4g} var {r['var_explained'] * 100:6.2f}%  SAM {r['sam_deg']:.2f}  "
              f"detail -{r['noise_reduced_pct']:.0f}%  edges {r['edge_retention']:.3f}" + (f"  kappa {r['kappa']:.1f}" if y is not None else ""))
    d = pd.DataFrame(rows)
    d.to_csv(out / "rf_sweep.csv", index=False)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    task = "kappa" if y is not None else "ARI"
    for z, (col, lab) in zip(ax.ravel(), [("var_explained", "variance explained (rebuilt cube)"), ("noise_reduced_pct", "fine detail removed (%)"),
                                           ("edge_retention", "edge retention (1 = perfect)"), (task, "task: " + task)]):
        for sc in a.sweep_color:
            dd = d[d.sigma_color == sc]
            z.plot(dd.sigma_spatial, dd[col], "o-", label=f"sigma_color {sc:g}")
        z.axhline(d[col].iloc[0], color="k", ls="--", label="no filter")
        z.set_xlabel("sigma_spatial"); z.set_title(lab); z.grid(alpha=.3); z.legend(fontsize=8)
    fig.suptitle(f"Recursive filter strength sweep (PPCA k={k})"); fig.tight_layout()
    fig.savefig(out / "fig5_rf_sweep.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["raichur", "indian_pines", "both", "file"], default="both")
    ap.add_argument("--date", default="2026-10-12", help="Raichur date (data/darshita/raichur_<date>_crop.npy)")
    ap.add_argument("--file", help=".npy/.mat cube for --dataset file")
    ap.add_argument("--labels", help=".npy/.mat label map (H,W); 0 = unlabelled")
    ap.add_argument("--label_window", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"), help="crop the label map manually (label pixels)")
    ap.add_argument("--no_align", action="store_true", help="do not search for the cube's position inside a larger label map")
    ap.add_argument("--ignore_labels", type=int, nargs="*", default=[0])
    ap.add_argument("--primary_k", type=int, default=3)
    ap.add_argument("--k_list", type=int, nargs="*")
    ap.add_argument("--sigma_spatial", type=float, default=None, help="default: image size / 80")
    ap.add_argument("--sigma_color", type=float, default=10)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n_clusters", type=int, default=8)
    ap.add_argument("--split", choices=["auto", "blocked", "random"], default="auto",
                    help="auto = blocked for images >= 256 px, random for small ones; blocked = spatially separated "
                         "train/test (honest for the filter); random = pixel split (what the papers use)")
    ap.add_argument("--tag", default="", help="suffix for the results folder, e.g. --tag k37")
    ap.add_argument("--repeats", type=int, default=1, help="number of different train/test splits to average (OA_std = spread)")
    ap.add_argument("--block", type=int, default=None, help="block size in px (default image/6)")
    ap.add_argument("--buffer", type=int, default=None, help="px dropped at block edges (default = filter sigma_spatial)")
    ap.add_argument("--train_block_frac", type=float, default=0.4, help="share of blocks used for training")
    ap.add_argument("--rf_sweep", action="store_true", help="also test several recursive-filter strengths")
    ap.add_argument("--sweep_spatial", type=float, nargs="*", default=[3, 6, 10, 20])
    ap.add_argument("--sweep_color", type=float, nargs="*", default=[10, 20, 30])
    ap.add_argument("--train_frac", type=float, default=0.10, help="only for --split random")
    ap.add_argument("--max_train", type=int, default=3000)
    ap.add_argument("--max_test", type=int, default=20000)
    a = ap.parse_args()
    for which in (["raichur", "indian_pines"] if a.dataset == "both" else [a.dataset]):
        print("\n" + "=" * 78 + f"\n{which.upper()}\n" + "=" * 78)
        run(a, which)


if __name__ == "__main__":
    main()