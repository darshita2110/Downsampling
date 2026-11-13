"""
Reconstruction-quality metrics. All functions expect data already normalised to [0, 1].

    MSE  = mean((x - x_hat)^2)              smaller is better, 0 = perfect
    RMSE = sqrt(MSE)                        same unit as the data (fraction of full range)
    MAE  = mean(|x - x_hat|)                less sensitive to a few large errors
    PSNR = 10 * log10(MAX^2 / MSE)  [dB]    MAX = 1 here; higher is better, infinite if MSE = 0
    SSIM = structural similarity in [-1, 1] compares local mean, contrast and structure in
           7x7 windows; 1 = identical. Needs the 2-D image layout, so it is computed per channel.
"""
import numpy as np
from skimage.metrics import structural_similarity


def mse(x, y):
    return float(np.mean((x - y) ** 2))


def rmse(x, y):
    return float(np.sqrt(mse(x, y)))


def mae(x, y):
    return float(np.mean(np.abs(x - y)))


def psnr(x, y, data_range=1.0):
    m = mse(x, y)
    return float("inf") if m == 0 else float(10 * np.log10(data_range ** 2 / m))


def ssim_channel(img_a, img_b, valid, data_range=1.0):
    """
    SSIM of one 2-D channel, averaged only over valid pixels.
    Invalid pixels are set to 0 in both images before the window computation (so they agree there),
    and the SSIM map is then averaged over the valid mask only.
    """
    a = np.where(valid, img_a, 0).astype(np.float64)
    b = np.where(valid, img_b, 0).astype(np.float64)
    _, smap = structural_similarity(a, b, data_range=data_range, full=True)
    return float(smap[valid].mean())


def all_metrics(orig_img, recon_img, valid, names, groups=None):
    """
    Compare an original and a reconstructed (H, W, C) image on the valid pixels.
    Returns {"overall": {...}, "per_channel": {name: {...}}, "per_group": {group: {...}}}.
    groups: {"RGB": [channel indices], ...} to also report metrics per product.
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
    out["overall"]["SSIM"] = float(np.mean(ssims))          # mean of per-channel SSIM
    for g, idx in (groups or {}).items():
        out["per_group"][g] = {"MSE": mse(xo[:, idx], xr[:, idx]), "RMSE": rmse(xo[:, idx], xr[:, idx]),
                               "MAE": mae(xo[:, idx], xr[:, idx]), "PSNR_dB": psnr(xo[:, idx], xr[:, idx]),
                               "SSIM": float(np.mean([ssims[i] for i in idx]))}
    return out
