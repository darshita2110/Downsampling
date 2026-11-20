"""
Metrics and protocols copied from the teammates' code so that this folder's results can sit in the SAME
tables as theirs. Each block names its source file; run experiments/09 on Indian Pines and compare the
'PCA (fit on all pixels)' row with the teammates' PCA/PPCA rows to confirm the reproduction.

Kartavya (Kartavya/common/metrics.py, preprocessing.py, evaluation.py)
    data are first mapped to [0,1] per channel (min/max over the pixels used), reconstructions are clipped
    to [0,1]; MSE, RMSE, MAE, PSNR (MAX = 1), SSIM (7x7 window, per channel, mean over channels).
Abhishek (Abhishek/universal_pipeline.py)
    work in the original units: RMSE, PSNR = 20*log10(range / RMSE) with range = p99 - p1 of the data,
    SAM in degrees, variance explained on standardised channels; k-means clusters of the features compared
    with clusters of the full standardised cube (ARI, NMI); RBF-SVM (C=100) classification with a random
    split of 10 % of each class for training (OA, AA, kappa in %), averaged over repeats.
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score
from sklearn.svm import SVC


# ----------------------------------------------------------------------------- Kartavya
def minmax_range(X):
    low = X.min(axis=0).astype(np.float64)
    high = X.max(axis=0).astype(np.float64)
    span = high - low
    high = np.where(span <= 0, low + 1.0, high)  # constant channel maps to 0
    return low, high


def to_unit(X, low, high):
    out = (np.asarray(X, dtype=np.float64) - low) / (high - low)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def k_mse(x, y):
    return float(np.mean((np.asarray(x, np.float64) - np.asarray(y, np.float64)) ** 2))


def k_psnr(x, y, data_range=1.0):
    m = k_mse(x, y)
    return float("inf") if m == 0 else float(10 * np.log10(data_range ** 2 / m))


def k_mae(x, y):
    return float(np.mean(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))


_SSIM_WARNED = False


def ssim_mean(orig_unit, rec_unit, grid_shape):
    """Mean over channels of SSIM (7x7 window), computed as in Kartavya/common/metrics.py: skimage
    structural_similarity with full=True, averaging the WHOLE SSIM map (borders included).
    Needs scikit-image; if it is missing, returns NaN and prints how to install it."""
    global _SSIM_WARNED
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        if not _SSIM_WARNED:
            print("  [SSIM skipped: scikit-image is not installed. Run: pip install scikit-image]")
            _SSIM_WARNED = True
        return float("nan")
    h, w = grid_shape
    vals = []
    for c in range(orig_unit.shape[1]):
        a = orig_unit[:, c].reshape(h, w).astype(np.float64)
        b = rec_unit[:, c].reshape(h, w).astype(np.float64)
        _, smap = structural_similarity(a, b, data_range=1.0, full=True, win_size=7)
        vals.append(float(smap.mean()))
    return float(np.nanmean(vals))


def kartavya_metrics(X_raw, rec_raw, low, high, grid_shape, with_ssim=True):
    xo, xr = to_unit(X_raw, low, high), to_unit(rec_raw, low, high)
    m = k_mse(xo, xr)
    out = {"MSE": m, "RMSE": float(np.sqrt(m)), "MAE": k_mae(xo, xr), "PSNR_dB": k_psnr(xo, xr)}
    out["SSIM"] = ssim_mean(xo, xr, grid_shape) if with_ssim else float("nan")
    return out


def storage_report(n_pixels, n_channels, k, model_values, bytes_per_value=4):
    original = n_pixels * n_channels * bytes_per_value
    reduced = n_pixels * k * bytes_per_value
    model = model_values * bytes_per_value
    return {"dimensionality_reduction_pct": 100.0 * (1.0 - k / n_channels),
            "storage_saving_pct": 100.0 * (1 - (reduced + model) / original),
            "compression_ratio": original / (reduced + model)}


# ----------------------------------------------------------------------------- Abhishek
def data_range_p99_p1(X):
    return float(np.percentile(X[::7], 99) - np.percentile(X[::7], 1))


def sam_deg(a, b):
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float(np.degrees(np.arccos(np.clip((a * b).sum(1) / den, -1, 1))).mean())


def abhishek_recon_metrics(X_raw, rec_raw, Xs_all, mu_all, sd_all, rng_val):
    """var_explained on standardised channels (all-pixel statistics), RMSE/PSNR in original units, SAM."""
    rec_s = (rec_raw - mu_all) / sd_all
    var_exp = 1 - float(((Xs_all - rec_s) ** 2).sum() / (Xs_all ** 2).sum())
    rmse = float(np.sqrt(np.mean((X_raw - rec_raw) ** 2)))
    return {"var_explained": var_exp, "rmse": rmse, "nrmse_pct": rmse / rng_val * 100,
            "psnr_db": 20 * np.log10(rng_val / max(rmse, 1e-12)), "sam_deg": sam_deg(X_raw, rec_raw)}


def cluster_labels(F, n_clusters=8, seed=0):
    idx = np.random.default_rng(seed).choice(len(F), min(len(F), 20000), replace=False)
    km = MiniBatchKMeans(n_clusters, n_init=3, random_state=seed, batch_size=4096).fit(F[idx])
    return km.predict(F)


def ari_nmi(ref_clusters, F, n_clusters=8):
    cl = cluster_labels(F, n_clusters)
    return adjusted_rand_score(ref_clusters, cl), normalized_mutual_info_score(ref_clusters, cl)


def make_split(y, ignore, frac, max_train, max_test, seed=0):
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
    if len(tr) > max_train:
        tr = rng.choice(tr, max_train, replace=False)
    if len(te) > max_test:
        te = rng.choice(te, max_test, replace=False)
    return tr, te


def svm_classify(F, y, splits):
    """Mean OA / AA / kappa (in %) of an RBF-SVM over the given (train, test) splits, plus OA spread."""
    res = []
    for tr, te in splits:
        clf = SVC(kernel="rbf", C=100, gamma="scale").fit(F[tr], y[tr])
        pred = clf.predict(F[te])
        rec = [100 * float(np.mean(pred[y[te] == c] == c)) for c in np.unique(y[te])]
        res.append({"OA": 100 * float(np.mean(pred == y[te])), "AA": float(np.mean(rec)),
                    "kappa": 100 * float(cohen_kappa_score(y[te], pred))})
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    out["OA_std"] = float(np.std([r["OA"] for r in res])) if len(res) > 1 else 0.0
    return out
