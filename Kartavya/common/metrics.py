"""
Evaluation metrics (unchanged definitions from the original repository). All reconstruction metrics expect
data already normalised to [0, 1].

    MSE  = mean((x - x_hat)^2)              smaller is better, 0 = perfect
    RMSE = sqrt(MSE)                        same unit as the data (fraction of full range)
    MAE  = mean(|x - x_hat|)                less sensitive to a few large errors
    PSNR = 10 * log10(MAX^2 / MSE)  [dB]    MAX = 1 here; higher is better, infinite if MSE = 0
    SSIM = structural similarity in [-1, 1] compares local mean, contrast and structure in
           7x7 windows; 1 = identical. Needs the 2-D image layout, so it is computed per channel.
           (Images smaller than 7 pixels use the largest odd window that fits; below 3 pixels SSIM is NaN.)
"""
import numpy as np
from skimage.metrics import structural_similarity


def mse(x, y):
    return float(np.mean((np.asarray(x, np.float64) - np.asarray(y, np.float64)) ** 2))


def rmse(x, y):
    return float(np.sqrt(mse(x, y)))


def mae(x, y):
    return float(np.mean(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))


def psnr(x, y, data_range=1.0):
    m = mse(x, y)
    return float("inf") if m == 0 else float(10 * np.log10(data_range ** 2 / m))


def ssim_channel(img_a, img_b, valid, data_range=1.0):
    """
    SSIM of one 2-D channel, averaged only over valid pixels.
    Invalid pixels are set to 0 in both images before the window computation (so they agree there),
    and the SSIM map is then averaged over the valid mask only.
    """
    side = min(img_a.shape)
    win = 7 if side >= 7 else (side if side % 2 == 1 else side - 1)
    if win < 3 or not valid.any():
        return float("nan")
    a = np.where(valid, img_a, 0).astype(np.float64)
    b = np.where(valid, img_b, 0).astype(np.float64)
    _, smap = structural_similarity(a, b, data_range=data_range, full=True, win_size=win)
    return float(smap[valid].mean())


def all_metrics(orig_img, recon_img, valid, names, groups=None):
    """
    Compare an original and a reconstructed (H, W, C) image on the valid pixels.
    Returns {"overall": {...}, "per_channel": {name: {...}}, "per_group": {group: {...}}}.
    groups: {"RGB": [channel indices], ...} to also report metrics per product / band region.
    """
    xo, xr = orig_img[valid], recon_img[valid]
    out = {"overall": {"MSE": mse(xo, xr), "RMSE": rmse(xo, xr), "MAE": mae(xo, xr), "PSNR_dB": psnr(xo, xr)},
           "per_channel": {}, "per_group": {}}
    ssims = []
    for c, name in enumerate(names):
        s = ssim_channel(orig_img[..., c], recon_img[..., c], valid)
        ssims.append(s)
        out["per_channel"][name] = {"MSE": mse(xo[:, c], xr[:, c]), "RMSE": rmse(xo[:, c], xr[:, c]),
                                    "MAE": mae(xo[:, c], xr[:, c]), "PSNR_dB": psnr(xo[:, c], xr[:, c]),
                                    "SSIM": s}
    out["overall"]["SSIM"] = float(np.nanmean(ssims)) if np.isfinite(ssims).any() else float("nan")
    for g, idx in (groups or {}).items():
        gs = [ssims[i] for i in idx]
        out["per_group"][g] = {"MSE": mse(xo[:, idx], xr[:, idx]), "RMSE": rmse(xo[:, idx], xr[:, idx]),
                               "MAE": mae(xo[:, idx], xr[:, idx]), "PSNR_dB": psnr(xo[:, idx], xr[:, idx]),
                               "SSIM": float(np.nanmean(gs)) if np.isfinite(gs).any() else float("nan")}
    return out


# ------------------------------------------------------------- component statistics
def component_statistics(s):
    """
    How independent are the columns of s (N samples x K components)?

    mean_abs_excess_kurtosis : average |kurtosis - 3|. A Gaussian has 0; larger = less Gaussian.
                               ICA maximises non-Gaussianity, so this should rise vs PCA.
    mean_abs_linear_corr     : average |corr(s_i, s_j)|, i != j. Both PCA and ICA make this ~0.
    mean_abs_square_corr     : average |corr(s_i^2, s_j^2)|, i != j. A NON-linear dependence
                               measure: uncorrelated-but-dependent components (typical for PCA)
                               show up here. Independent components give ~0.
    """
    s = np.asarray(s, dtype=np.float64)
    z = (s - s.mean(0)) / np.maximum(s.std(0), 1e-12)
    kurt = (z ** 4).mean(0) - 3.0
    out = {"excess_kurtosis": kurt.tolist(), "mean_abs_excess_kurtosis": float(np.abs(kurt).mean())}
    k = s.shape[1]
    if k > 1:
        off = ~np.eye(k, dtype=bool)
        out["mean_abs_linear_corr"] = float(np.abs(np.corrcoef(z.T))[off].mean())
        out["mean_abs_square_corr"] = float(np.abs(np.corrcoef((z ** 2).T))[off].mean())
    else:
        out["mean_abs_linear_corr"] = float("nan")
        out["mean_abs_square_corr"] = float("nan")
    return out


# ------------------------------------------------------------- classification metrics
def classification_report(y_true, y_pred, classes):
    """
    Accuracy, per-class precision/recall/F1, their macro and support-weighted averages, and the
    confusion matrix (rows = true class, columns = predicted class), computed from scratch.
        precision_c = TP / (TP + FP)   of pixels predicted c, how many really are c
        recall_c    = TP / (TP + FN)   of pixels that are c, how many were found
        F1_c        = 2 P R / (P + R)
    A class with no predictions gets precision 0.

    Macro averages are taken over the classes PRESENT in y_true (support > 0) only. Averaging over every
    trained class, as an earlier version did, counts each class that has no test pixels as F1 = 0: with a
    spatial split where only 2 of 10 trained classes occur in the test region, macro-F1 fell to ~0.2 although
    both present classes were classified well. The old value is still reported as "*_macro_all_classes".
    """
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (np.vectorize(idx.get)(y_true), np.vectorize(idx.get)(y_pred)), 1)
    tp = np.diag(cm).astype(float)
    pred_tot, true_tot = cm.sum(0).astype(float), cm.sum(1).astype(float)
    prec = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    rec = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    w = true_tot / true_tot.sum()
    present = true_tot > 0
    return {"accuracy": float(tp.sum() / cm.sum()),
            "precision_macro": float(prec[present].mean()), "recall_macro": float(rec[present].mean()),
            "f1_macro": float(f1[present].mean()),
            "n_classes_evaluated": int(present.sum()),
            "classes_without_test_pixels": [str(c) for c, p in zip(classes, present) if not p],
            "precision_macro_all_classes": float(prec.mean()), "recall_macro_all_classes": float(rec.mean()),
            "f1_macro_all_classes": float(f1.mean()),
            "precision_weighted": float((prec * w).sum()), "recall_weighted": float((rec * w).sum()),
            "f1_weighted": float((f1 * w).sum()),
            "per_class": {str(c): {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                                   "support": int(true_tot[i])} for i, c in enumerate(classes)},
            "confusion_matrix": cm.tolist(), "classes": [str(c) for c in classes]}


def fisher_ratio(z, y):
    """
    Class separability of a feature space: trace(S_between) / trace(S_within) (higher = classes further
    apart relative to their spread). Used to compare the original channels with LDA components.
    """
    z = np.asarray(z, dtype=np.float64)
    mu = z.mean(0)
    sb = sw = 0.0
    for c in np.unique(y):
        zc = z[y == c]
        sb += len(zc) * ((zc.mean(0) - mu) ** 2).sum()
        sw += ((zc - zc.mean(0)) ** 2).sum()
    return float(sb / sw) if sw > 0 else float("inf")
