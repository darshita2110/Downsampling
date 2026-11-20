"""
wave_ae.py  --  Wavelet-domain convolutional autoencoder ("WaveAE")
====================================================================
2-D discrete wavelet transform (DWT)  +  deep learning, for reducing the number of channels of a
multispectral / hyperspectral cube before downlinking.

WHY (links to the two papers)
  * Deep-learning review (2026), sec. 7.3 and 7.5: wavelet decomposition has "inherent advantages in
    small-sample scenarios" and is "white-box"; the authors propose using wavelet features as the PRIOR INPUT
    of deep band-selection / reduction networks. Only Islam et al. (2024, Hybrid-2DNET, a 2-D wavelet CNN fed
    with already-selected bands) is cited as an early step, so this direction is largely open.
  * Vaddi et al. (2024): wavelet-enabled band selection + 3-D spatial features is cited (Vaddi & Manoharan 2021),
    and PCA-type feature extraction + spatial features is named as the classic (moderate-accuracy) baseline.

IDEA
  1. Standardise every band, then apply a one-level Haar 2-D DWT to every band:
         LL (smooth image), LH / HL (horizontal / vertical edges), HH (diagonal detail).
     Each is half the height and width, so the cube becomes (H/2, W/2, 4*C). It is exactly invertible.
  2. A small convolutional autoencoder compresses these 4*C wavelet channels to k_ae latent channels
     (1x1 conv = spectral mixing, 3x3 conv = spatial context). Only the latent is "downlinked".
  3. On the ground: decoder -> inverse DWT -> reconstructed (H, W, C) cube.

FAIR COMPARISON = EQUAL DATA VOLUME
  The latent has half the resolution, so k_ae latent channels cost k_ae/4 full-resolution channels.
  For a budget of k_eq full-resolution channels we use k_ae = 4*k_eq and compare against
     PCA (k_eq ch, full res)              the classic per-pixel linear reduction (same subspace as PPCA)
     PCA + downsample (k_ae ch, half res) spectral PCA, then 2x2 averaging (no wavelets, no deep learning)
     Wavelet-PCA (k_ae ch, half res)      the LINEAR version of WaveAE  -> isolates the deep-learning gain
     WaveAE (k_ae ch, half res)           the new method

This file is self-contained: it does not modify any earlier file. For labels, train/test splits and the SVM it
only READS helper functions from universal_pipeline.py so the classification numbers are directly comparable
with the PPCA results. Needs PyTorch:   pip install torch

    python Abhishek\\wave_ae.py --dataset indian_pines --steps 600 --k_eq 3 5 10 --split random --repeats 5
    python Abhishek\\wave_ae.py --dataset raichur --date 2026-10-12 --k_eq 1 2 3
    python Abhishek\\wave_ae.py --dataset indian_pines --k_eq 7 --tag k7         # any budget you like
    python Abhishek\\wave_ae.py --dataset raichur --fit_date 2026-09-27 --date 2026-10-12   # CROSS-DATE: train on one date, compress another
Results: Abhishek/results/wave_ae_<dataset>[_tag]/
"""
import argparse
import sys
import time
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from universal_pipeline import build_splits, classify_multi, get_labels, sam_deg, ssim_cube    # read-only helpers

RAICHUR_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


# ======================================================================
# 1. data
# ======================================================================
def load_dataset(a):
    """Returns cube (H, W, C) float32, band names or None, short name."""
    if a.dataset == "raichur":
        return np.load(f"data/darshita/raichur_{a.date}_crop.npy").astype(np.float32), RAICHUR_BANDS, f"raichur_{a.date}"
    if a.dataset == "indian_pines":
        return loadmat("data/darshita/Indian_pines_corrected.mat")["indian_pines_corrected"].astype(np.float32), None, "indian_pines"
    p = Path(a.file)
    if p.suffix == ".npy":
        arr = np.load(p)
    else:
        m = loadmat(p)
        arr = next(v for k, v in m.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3)
    return arr.astype(np.float32), None, p.stem


def inspect(cube, name, ks):
    """Print dimensions and what each data budget means."""
    H, W, C = cube.shape
    print(f"\nDATASET {name}\n  shape (H, W, C): {cube.shape}  dtype {cube.dtype}  ->  {H * W:,} pixels, {C} channels"
          f"\n  value range: {cube.min():.4g} .. {cube.max():.4g}   odd size handled by 1-pixel reflect padding: {H % 2 == 1 or W % 2 == 1}"
          f"\n  wavelet cube (half resolution): ({(H + 1) // 2}, {(W + 1) // 2}, {4 * C}) = LL, LH, HL, HH of every band")
    print(f"  {'budget k_eq':>11s} {'latent ch (half res)':>21s} {'values/pixel':>13s} {'compression':>12s}")
    for k in ks:
        print(f"  {k:11d} {min(4 * k, C):21d} {min(4 * k, C) / 4:13.2f} {C / (min(4 * k, C) / 4):11.1f}x")


# ======================================================================
# 2. Haar 2-D DWT (orthonormal, exactly invertible)
# ======================================================================
def pad_even(x):
    ph, pw = x.shape[0] % 2, x.shape[1] % 2
    return np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else x


def haar_dwt2(x):
    """x (H, W, C), even H and W  ->  LL, LH, HL, HH, each (H/2, W/2, C)."""
    a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    return (a + b + c + d) / 2, (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2


def haar_idwt2(ll, lh, hl, hh):
    out = np.empty((ll.shape[0] * 2, ll.shape[1] * 2, ll.shape[2]), ll.dtype)
    out[0::2, 0::2] = (ll + lh + hl + hh) / 2
    out[0::2, 1::2] = (ll + lh - hl - hh) / 2
    out[1::2, 0::2] = (ll - lh + hl - hh) / 2
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2
    return out


def split_bands(w, C):
    """(h, w, 4C) -> LL, LH, HL, HH."""
    return [w[:, :, i * C:(i + 1) * C] for i in range(4)]


def wave_to_cube(w, C, H, W):
    """(h, w, 4C) wavelet cube -> (H, W, C) cube (inverse DWT, crop padding)."""
    return haar_idwt2(*split_bands(w, C))[:H, :W]


def up2(z, H, W):
    """Nearest-neighbour 2x upsampling of (h, w, k) to (H, W, k)."""
    return np.repeat(np.repeat(z, 2, axis=0), 2, axis=1)[:H, :W]


# ======================================================================
# 3. the network
# ======================================================================
class WaveAE(nn.Module):
    """
    encoder: 1x1 conv (spectral mix) -> 3x3 conv (spatial context) -> 1x1 conv to k latent channels; decoder mirrors it.
    Optional lin_init=(mean, U) from the Wavelet-PCA fit: adds a LINEAR encoder/decoder path that starts exactly at the
    Wavelet-PCA solution while the non-linear branches start at zero, so training can only improve on the linear twin.
    """

    def __init__(self, c_in, k, width=128, lin_init=None):
        super().__init__()
        g = lambda ch: nn.GroupNorm(8, ch)
        self.enc = nn.Sequential(nn.Conv2d(c_in, width, 1), g(width), nn.GELU(),
                                 nn.Conv2d(width, width // 2, 3, padding=1), g(width // 2), nn.GELU(),
                                 nn.Conv2d(width // 2, k, 1))
        self.dec = nn.Sequential(nn.Conv2d(k, width // 2, 1), g(width // 2), nn.GELU(),
                                 nn.Conv2d(width // 2, width, 3, padding=1), g(width), nn.GELU(),
                                 nn.Conv2d(width, c_in, 1))
        self.lin_enc = self.lin_dec = None
        if lin_init is not None:
            mu_w, U_w = lin_init                                   # (c_in,), (c_in, k)
            self.lin_enc, self.lin_dec = nn.Conv2d(c_in, k, 1), nn.Conv2d(k, c_in, 1)
            mu_t = torch.from_numpy(np.ascontiguousarray(mu_w)).float()
            U_t = torch.from_numpy(np.ascontiguousarray(U_w)).float()
            with torch.no_grad():
                self.lin_enc.weight.copy_(U_t.T[:, :, None, None]); self.lin_enc.bias.copy_(-(U_t.T @ mu_t))
                self.lin_dec.weight.copy_(U_t[:, :, None, None]); self.lin_dec.bias.copy_(mu_t)
                for m in (self.enc[-1], self.dec[-1]):             # non-linear branches start at zero
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x):
        z = self.enc(x)
        return z + self.lin_enc(x) if self.lin_enc is not None else z

    def decode(self, z):
        r = self.dec(z)
        return r + self.lin_dec(z) if self.lin_dec is not None else r

    def forward(self, x):
        return self.decode(self.encode(x))

    def n_params(self, part):
        mods = [self.enc, self.lin_enc] if part == "enc" else [self.dec, self.lin_dec]
        return sum(p.numel() for m in mods if m is not None for p in m.parameters())


def train_ae(wv, C, k, a, device, lin_init=None):
    """Train on random patches of the wavelet cube wv (h, w, 4C). Unsupervised: target = input."""
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    h, w, c = wv.shape
    T = torch.from_numpy(np.ascontiguousarray(wv.transpose(2, 0, 1)))
    model = WaveAE(c, k, a.width, lin_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    wt = torch.ones(1, c, 1, 1)
    wt[:, C:] = a.detail_weight                              # channels C.. are the LH/HL/HH detail subbands
    wt = wt.to(device)
    p = min(a.patch, h, w)
    losses = []
    t0 = time.time()
    for s in range(a.steps):
        ys, xs = rng.integers(0, h - p + 1, a.batch), rng.integers(0, w - p + 1, a.batch)
        x = torch.stack([T[:, y:y + p, xx:xx + p] for y, xx in zip(ys, xs)]).to(device)
        loss = (((model(x) - x) ** 2) * wt).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if (s + 1) % max(1, a.steps // 5) == 0:
            print(f"      step {s + 1:5d}/{a.steps}  loss {np.mean(losses[-20:]):.4f}")
    return model, losses, time.time() - t0


def ae_encode_decode(model, wv, device):
    """Latent, reconstruction and the measured time (s) of the encoder on the whole image."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(wv.transpose(2, 0, 1)))[None].to(device)
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        model.encode(x); sync()                              # warm-up
        t0 = time.perf_counter(); z = model.encode(x); sync()
        t_enc = time.perf_counter() - t0
        rec = model.decode(z)
    return z[0].cpu().numpy().transpose(1, 2, 0), rec[0].cpu().numpy().transpose(1, 2, 0), t_enc


# ======================================================================
# 4. linear baselines
# ======================================================================
def pca_fit(X):
    mu = X.mean(0)
    Xc = X - mu
    ev, U = np.linalg.eigh((Xc.T @ Xc) / (len(Xc) - 1))
    o = np.argsort(ev)[::-1]
    return mu, U[:, o].astype(np.float32), ev[o]


# ======================================================================
# 5. metrics
# ======================================================================
def recon_metrics(X, Xs, rec_s, mu, sd, rng_val, hw=None):
    """Rebuild quality in standardised space (variance explained) and original units (RMSE, PSNR, SAM).
    Give hw = (H, W) to also get SSIM."""
    rec_s = rec_s.reshape(Xs.shape)
    rec = rec_s * sd + mu
    rmse = float(np.sqrt(np.mean((X - rec) ** 2)))
    out = {"var_explained": 1 - float(((Xs - rec_s) ** 2).sum() / ((Xs - Xs.mean(0)) ** 2).sum()), "rmse": rmse,
           "nrmse_pct": rmse / rng_val * 100, "psnr_db": 20 * np.log10(rng_val / max(rmse, 1e-12)),
           "sam_deg": sam_deg(X, rec)}
    if hw is not None:
        out["ssim"] = ssim_cube(X, rec, hw, rng_val)
    return out


# ======================================================================
# 6. figures
# ======================================================================
def display_idx(C, names):
    if names and all(b in names for b in ("B04", "B03", "B02")):
        return [names.index(b) for b in ("B04", "B03", "B02")], "true colour"
    return [int(.15 * C), int(.10 * C), int(.05 * C)], "false colour"


def show(cube, idx, lims):
    return np.dstack([np.clip((cube[:, :, i] - lo) / (hi - lo + 1e-12), 0, 1) for i, (lo, hi) in zip(idx, lims)])


def stretch(a):
    lo, hi = np.percentile(a, (2, 98))
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def fig_overview(out, name, cube, names, bands, latents, losses, k_show):
    """What the DWT does and what the latents look like."""
    H, W, C = cube.shape
    b = names.index("B08") if names and "B08" in names else int(.3 * C)
    ll, lh, hl, hh = haar_dwt2(pad_even(cube[:, :, b:b + 1]))
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.8))
    for z, img, t in zip(ax[0], [pad_even(cube[:, :, b:b + 1])[:, :, 0], ll[:, :, 0], np.abs(lh[:, :, 0]), np.abs(hl[:, :, 0]), np.abs(hh[:, :, 0])],
                         ["one band (original)", "LL: smooth image", "|LH|: edges", "|HL|: edges", "|HH|: diagonal detail"]):
        z.imshow(stretch(img), cmap="gray"); z.set_title(f"{t}" + (f"\n(band {names[b]})" if names and t.startswith("one") else ""), fontsize=10)
    for z, key, t in zip(ax[1][:3], ["WaveAE", "Wavelet-PCA", "PCA"], ["WaveAE latent (3 of %d ch)", "Wavelet-PCA latent (3 of %d ch)", "PCA scores (3 of %d ch, full res)"]):
        l = latents[key]
        z.imshow(np.dstack([stretch(l[:, :, i]) if i < l.shape[2] else np.zeros(l.shape[:2]) for i in range(3)]))
        z.set_title(t % l.shape[2], fontsize=10)
    ax[1, 3].plot(losses); ax[1, 3].set_yscale("log"); ax[1, 3].set_title(f"WaveAE training loss (budget k_eq={k_show})", fontsize=10)
    ax[1, 3].set_xlabel("step"); ax[1, 3].grid(alpha=.3)
    ax[1, 4].axis("off")
    for z in list(ax[0]) + list(ax[1][:3]): z.set_xticks([]); z.set_yticks([])
    fig.suptitle(f"{name}: 2-D wavelet transform of one band and the learned compressed representations", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "fig1_wavelet_and_latent.png", dpi=105); plt.close(fig)


def fig_recon(out, name, cube, names, recs, k_show):
    """Original vs reconstructions at equal data volume, plus error maps."""
    H, W, C = cube.shape
    idx, kind = display_idx(C, names)
    lims = [tuple(np.percentile(cube[:, :, i], (2, 98))) for i in idx]
    keys = list(recs)
    fig, ax = plt.subplots(2, len(keys) + 1, figsize=(4.6 * (len(keys) + 1), 9.4))
    ax[0, 0].imshow(show(cube, idx, lims)); ax[0, 0].set_title(f"Original ({C} channels, {kind})", fontsize=10); ax[1, 0].axis("off")
    errs = {k: np.abs(cube - recs[k]).mean(axis=2) for k in keys}
    vmax = np.percentile(np.concatenate([e.ravel() for e in errs.values()]), 99)
    for j, k in enumerate(keys, 1):
        ax[0, j].imshow(show(recs[k], idx, lims)); ax[0, j].set_title(k, fontsize=9)
        ax[1, j].imshow(errs[k], cmap="magma", vmin=0, vmax=vmax); ax[1, j].set_title(f"mean |error| = {errs[k].mean():.3g}", fontsize=9)
    for z in ax.ravel(): z.set_xticks([]); z.set_yticks([])
    fig.suptitle(f"{name}: reconstructions from the SAME data volume (budget = {k_show} full-resolution channels)", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "fig2_reconstruction.png", dpi=105); plt.close(fig)


def fig_metrics(out, name, df, has_labels):
    cols = [("nrmse_pct", "Reconstruction error NRMSE (%) - lower is better"), ("sam_deg", "Spectral angle error (deg) - lower is better"),
            ("OA" if has_labels else "var_explained", "Classification OA (%) - higher is better" if has_labels else "Variance explained"),
            ("psnr_db", "PSNR (dB) - higher is better")]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    for z, (c, t) in zip(ax.ravel(), cols):
        for m in df.method.unique():
            d = df[df.method == m].sort_values("k_eq")
            if c == "OA" and "OA_std" in d.columns and d.OA_std.notna().any():
                z.errorbar(d.k_eq, d[c], yerr=d.OA_std, marker="o", capsize=3, label=m, lw=2.4 if m == "WaveAE" else 1.4)
            else:
                z.plot(d.k_eq, d[c], marker="o", label=m, lw=2.4 if m == "WaveAE" else 1.4)
        z.set_title(t, fontsize=10); z.set_xlabel("data budget k_eq (full-resolution channels worth of data per pixel)"); z.grid(alpha=.3); z.legend(fontsize=7)
    fig.suptitle(f"{name}: equal-data-volume comparison", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "fig3_metrics_vs_budget.png", dpi=105); plt.close(fig)


# ======================================================================
# 7. main
# ======================================================================
def drop_block_leaks(splits, W):
    """
    Features of the half-resolution methods are constant inside each 2x2 block, so a test pixel that shares a block with
    a training pixel has an IDENTICAL feature vector in the training set. Remove such test pixels (same test set for
    every method, so the comparison stays fair). Neighbouring-block leakage remains (as in any random pixel split).
    """
    nbx = (W + 1) // 2
    bid = lambda idx: (idx // W // 2) * nbx + (idx % W) // 2
    out, removed = [], []
    for tr, te in splits:
        bad = np.isin(bid(te), np.unique(bid(tr)))
        out.append((tr, te[~bad])); removed.append(int(bad.sum()))
    return out, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["raichur", "indian_pines", "file"], default="indian_pines")
    ap.add_argument("--date", default="2026-10-12", help="date that is compressed and evaluated")
    ap.add_argument("--fit_date", help="CROSS-DATE test: fit standardisation, PCA and train WaveAE on THIS date, then compress --date")
    ap.add_argument("--file")
    ap.add_argument("--k_eq", type=int, nargs="*", help="budgets: full-resolution channels worth of data (latent = 4 x this)")
    ap.add_argument("--primary", type=int, help="budget used in the figures (default: middle one)")
    ap.add_argument("--steps", type=int, default=3000, help="training steps per budget")
    ap.add_argument("--patch", type=int, default=32, help="training patch size (half-resolution pixels)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=128, help="try 256 for a bigger network")
    ap.add_argument("--pca_init", action="store_true",
                    help="start from the Wavelet-PCA solution (linear path + zero-initialised non-linear branches)")
    ap.add_argument("--detail_weight", type=float, default=1.0, help=">1 makes the network care more about edges/detail subbands")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    # labels / splits (same options and meaning as universal_pipeline.py)
    ap.add_argument("--labels")
    ap.add_argument("--label_window", type=int, nargs=4)
    ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--ignore_labels", type=int, nargs="*", default=[0])
    ap.add_argument("--split", choices=["auto", "blocked", "random"], default="auto")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--train_frac", type=float, default=0.10)
    ap.add_argument("--train_block_frac", type=float, default=0.4)
    ap.add_argument("--max_train", type=int, default=3000)
    ap.add_argument("--max_test", type=int, default=20000)
    ap.add_argument("--block", type=int)
    ap.add_argument("--buffer", type=int)
    ap.add_argument("--keep_block_leaks", action="store_true",
                    help="do NOT remove test pixels that share a 2x2 block with a training pixel (old behaviour)")
    a = ap.parse_args()
    a.rf_sweep, a.sweep_spatial = False, []
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))

    # ---- load + inspect ----
    cube, names, name = load_dataset(a)
    H, W, C = cube.shape
    ks = sorted({k for k in (a.k_eq or ([1, 2, 3] if C <= 30 else [3, 5, 10])) if 1 <= k and 4 * k <= 4 * C})
    inspect(cube, name, ks)
    primary = a.primary if a.primary in ks else ks[len(ks) // 2]
    out = HERE / "results" / (f"wave_ae_{name}" + (f"_fit{a.fit_date}" if a.fit_date else "") + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    # ---- standardise, DWT ----
    X = cube.reshape(-1, C)
    if a.fit_date:                                                          # cross-date: everything is FITTED on another date
        if a.dataset != "raichur":
            raise SystemExit("--fit_date needs --dataset raichur (Indian Pines is a single scene)")
        fit_cube, _, _ = load_dataset(types.SimpleNamespace(dataset=a.dataset, date=a.fit_date, file=a.file))
        if fit_cube.shape != cube.shape:
            raise SystemExit(f"fit cube {fit_cube.shape} and target cube {cube.shape} must have the same shape")
        Xf = fit_cube.reshape(-1, C)
        print(f"\n[cross-date] model fitted on {a.fit_date}, compressing and evaluating {a.date}")
    else:
        Xf = X
    mu, sd = X.mean(0), X.std(0)                                            # the target's OWN per-band mean/std (only 2*C numbers
    sd[sd < 1e-12] = 1.0                                                    # of side information, so a date-wide brightness shift
    Xs = ((X - mu) / sd).astype(np.float32)                                 # does not count against the model)
    if a.fit_date:
        muf, sdf = Xf.mean(0), Xf.std(0)
        sdf[sdf < 1e-12] = 1.0
        Xs_fit = ((Xf - muf) / sdf).astype(np.float32)                      # training data, standardised with its own statistics
    else:
        Xs_fit = Xs
    rng_val = float(np.percentile(X[::7], 99) - np.percentile(X[::7], 1))
    to_wave = lambda z: np.concatenate(haar_dwt2(pad_even(z.reshape(H, W, C))), axis=2).astype(np.float32)
    wv = to_wave(Xs)                                                        # (h, w, 4C)
    wv_fit = wv if not a.fit_date else to_wave(Xs_fit)
    h, w = wv.shape[:2]
    print(f"\n[dwt] cube {cube.shape} -> wavelet cube {wv.shape}; round-trip error "
          f"{np.abs(wave_to_cube(wv, C, H, W) - Xs.reshape(H, W, C)).max():.2e} (must be ~0)")

    # ---- labels / splits (identical to the PPCA experiments) ----
    largs = types.SimpleNamespace(labels=a.labels, dataset=a.dataset, date=a.date, label_window=a.label_window, no_align=a.no_align)
    y_map = get_labels(largs, a.dataset, (H, W), cube, names)
    y, splits, removed = None, [], []
    split_desc = "no labels found -> classification skipped"
    if y_map is not None:
        y = y_map.reshape(-1)
        splits, split_desc = build_splits(a, y, y_map, H, W, 6)             # 6 px buffer: covers the network's receptive field
        if not a.keep_block_leaks:
            splits, removed = drop_block_leaks(splits, W)
            print(f"[split] removed test pixels sharing a 2x2 block with a training pixel (per repeat): {removed}")
        print(f"[split] {a.split}: train {len(splits[0][0]):,}, test {len(splits[0][1]):,} px, {a.repeats} repeat(s)")

    # ---- linear fits (once) ----
    mu0, U, ev = pca_fit(Xs_fit)
    Wflat = wv.reshape(-1, 4 * C)                                           # target, to be encoded
    muw, Uw, evw = pca_fit(wv_fit.reshape(-1, 4 * C))
    cube_MB = H * W * C * 4 / 1e6

    rows, latents_primary, recs_primary, losses_primary = [], {}, {}, []

    def evaluate(method, k_eq, sent, res, rec_s, feats, lat_vals, side_vals, macs, extra=None):
        mb = (lat_vals + side_vals) * 4 / 1e6                                 # float32 values, latent + side information
        r = {"method": method, "k_eq": k_eq, "channels_sent": sent, "resolution": res, "compression_x": round(C / k_eq, 1),
             "transmit_MB": round(mb, 3), "effective_compression_x": round(cube_MB / mb, 1),
             "encoder_macs_per_pixel": int(macs), **recon_metrics(X, Xs, rec_s, mu, sd, rng_val, (H, W)), **(extra or {})}
        if y is not None:
            r.update(classify_multi(feats.reshape(H * W, -1), y, splits)[0])
        rows.append(r)
        print(f"    {method:34s} var {r['var_explained'] * 100:6.2f}%  NRMSE {r['nrmse_pct']:5.2f}%  SAM {r['sam_deg']:5.2f}  SSIM {r['ssim']:.4f}  "
              f"eff.compr {r['effective_compression_x']:6.1f}x" + (f"  OA {r['OA']:5.1f}%" if y is not None else ""))

    for k_eq in ks:
        k_ae = min(4 * k_eq, C)
        print(f"\n[budget k_eq={k_eq}] full-res channels = {k_eq}; half-res latent channels = {k_ae}")

        # 1. PCA, full resolution, k_eq channels
        Z1 = (Xs - mu0) @ U[:, :k_eq]
        rec1 = Z1 @ U[:, :k_eq].T + mu0
        evaluate(f"PCA ({k_eq} ch, full res)", k_eq, k_eq, "full", rec1, Z1, H * W * k_eq, C * (k_eq + 1), C * k_eq)

        # 2. PCA with k_ae channels, then 2x2 averaging (half res) - spatial downsampling without wavelets/DL
        Zk = ((Xs - mu0) @ U[:, :k_ae]).reshape(H, W, k_ae)
        Zd = pad_even(Zk).reshape(h, 2, w, 2, k_ae).mean(axis=(1, 3))
        Zdu = up2(Zd, H, W)
        rec2 = Zdu.reshape(-1, k_ae) @ U[:, :k_ae].T + mu0
        evaluate(f"PCA+downsample ({k_ae} ch, half res)", k_eq, k_ae, "half", rec2, Zdu, h * w * k_ae, C * (k_ae + 1), C * k_ae)

        # 3. Wavelet-PCA: linear counterpart of WaveAE
        Zw = (Wflat - muw) @ Uw[:, :k_ae]
        recw = wave_to_cube((Zw @ Uw[:, :k_ae].T + muw).reshape(h, w, 4 * C), C, H, W)
        evaluate(f"Wavelet-PCA ({k_ae} ch, half res)", k_eq, k_ae, "half", recw, up2(Zw.reshape(h, w, k_ae), H, W),
                 h * w * k_ae, 4 * C * (k_ae + 1), C * k_ae)

        # 4. WaveAE
        print(f"    training WaveAE ({a.steps} steps, device {device}{', PCA-initialised' if a.pca_init else ''}) ...")
        model, losses, secs = train_ae(wv_fit, C, k_ae, a, device, (muw, Uw[:, :k_ae]) if a.pca_init else None)
        z, recwv, t_enc = ae_encode_decode(model, wv, device)
        reca = wave_to_cube(recwv, C, H, W)
        n_enc, n_dec = model.n_params("enc"), model.n_params("dec")
        macs = (4 * C * a.width + a.width * (a.width // 2) * 9 + (a.width // 2) * k_ae) / 4 + (C * k_ae if a.pca_init else 0)
        evaluate(f"WaveAE ({k_ae} ch, half res)", k_eq, k_ae, "half", reca, up2(z, H, W), h * w * k_ae, n_dec, macs,
                 {"train_seconds": round(secs, 1), "parameters": n_enc + n_dec, "enc_params": n_enc, "dec_params": n_dec,
                  "decoder_MB": round(n_dec * 4 / 1e6, 3), "encode_ms": round(t_enc * 1000, 1)})
        np.save(out / f"waveae_latent_k{k_ae}.npy", z.astype(np.float32))
        torch.save(model.state_dict(), out / f"waveae_k{k_ae}.pt")

        if k_eq == primary:
            latents_primary = {"WaveAE": z, "Wavelet-PCA": Zw.reshape(h, w, k_ae), "PCA": Z1.reshape(H, W, k_eq)}
            unstd = lambda r_: r_.reshape(H, W, C) * sd + mu
            recs_primary = {f"PCA ({k_eq} ch, full res)": unstd(rec1), f"PCA+downsample ({k_ae} ch, half res)": unstd(rec2),
                            f"Wavelet-PCA ({k_ae} ch, half res)": unstd(recw), f"WaveAE ({k_ae} ch, half res)": unstd(reca)}
            losses_primary = losses

    # ---- save + report ----
    df = pd.DataFrame(rows)
    df["method"] = df.method.str.replace(r" \(.*\)", "", regex=True)
    df.to_csv(out / "metrics.csv", index=False)
    fig_overview(out, name, cube, names, None, latents_primary, losses_primary, primary)
    fig_recon(out, name, cube, names, recs_primary, primary)
    fig_metrics(out, name, df, y is not None)
    header = (f"{name}  budgets k_eq={ks}\n"
              f"WaveAE settings: steps {a.steps}, width {a.width}, lr {a.lr}, batch {a.batch}, patch {a.patch}, seed {a.seed}, "
              f"detail_weight {a.detail_weight}, pca_init {a.pca_init}, device {device}\n"
              + (f"CROSS-DATE: standardisation, PCA bases and WaveAE fitted on {a.fit_date}; compressed and evaluated on {a.date}. "
                 "The target's per-band mean/std (2*C numbers) are sent as side information. Decoder weights are reusable, so the latent-only compression_x applies.\n" if a.fit_date else "")
              + f"Split: {split_desc}; repeats {a.repeats}; test pixels removed (share a 2x2 block with a training pixel): "
              f"{removed if removed else 'not applied'}\n"
              "Columns: compression_x counts latent values only. effective_compression_x also counts side information "
              "(PCA basis / decoder weights) at 4 bytes per value. encoder_macs_per_pixel = on-board multiply-adds per "
              "full-resolution pixel; encode_ms = measured encoder time for the whole image on this device. "
              "'Equal data volume' means equal NUMBER OF VALUES, not equal bits.\n")
    caveat = ("\nNOTE: the half-resolution latent and the 3x3 convolutions mix neighbouring pixels, like a spatial filter. With a random "
              "split this is optimistic for WaveAE / Wavelet-PCA / PCA+downsample even after removing same-block duplicates. The blocked "
              "split (default for images >= 256 px) is the honest test. Raichur labels are Sen2Cor SCL, computed from the same bands.") if y is not None else ""
    (out / "report.txt").write_text(f"{header}\n{df.to_string(index=False, float_format=lambda v: f'{v:.4g}')}{caveat}\n", encoding="utf-8")
    print(f"\n[saved] {out}\n  metrics.csv, report.txt, fig1_wavelet_and_latent.png, fig2_reconstruction.png, fig3_metrics_vs_budget.png,"
          f"\n  waveae_latent_k*.npy (the compressed data), waveae_k*.pt (trained weights)")
    print(df[["method", "k_eq", "channels_sent", "compression_x", "effective_compression_x", "var_explained", "nrmse_pct", "sam_deg", "ssim",
              "encoder_macs_per_pixel"] + (["OA"] if y is not None else [])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()