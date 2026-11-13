"""
view_cube.py  --  SEE what is inside a multispectral / hyperspectral cube
==========================================================================
A cube has C layers (bands) but a screen can only show 3 colours at once, so we look at it in
four ways:  (1) false-colour composites  (2) a gallery of single bands  (3) spectra  (4) labels.

    python Abhishek\\view_cube.py --describe data\\darshita\\Indian_pines_corrected.mat   # what is in the file?
    python Abhishek\\view_cube.py --dataset indian_pines
    python Abhishek\\view_cube.py --dataset raichur --date 2026-10-12
    python Abhishek\\view_cube.py --dataset file --file cube.npy --labels labels.npy

Output folder:  Abhishek/results/<dataset>/view/   (nothing else is touched)
"""
import argparse
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from universal_pipeline import display_rgb, get_dataset, get_labels, inspect_cube, stretch   # reuse loaders


def describe_mat(path):
    """List every variable stored in a MATLAB .mat file: name, shape, type, value range."""
    m = loadmat(path)
    print(f"\n{path}")
    print("  header :", m.get("__header__", b"").decode(errors="ignore").strip())
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.size > 0 and np.issubdtype(v.dtype, np.number):
            print(f"  variable '{k}': shape {v.shape}, dtype {v.dtype}, min {v.min()}, max {v.max()}, "
                  f"{'<- looks like an image cube (H, W, bands)' if v.ndim == 3 else '<- 2-D map / table'}")
        else:
            print(f"  variable '{k}': {type(v).__name__} {getattr(v, 'shape', '')}")


def composites(cube, names):
    """Several 3-band false-colour views. Different band choices highlight different materials."""
    C = cube.shape[2]
    out = [display_rgb(cube, names)]
    if C > 30:                                              # hyperspectral: add a NIR-heavy composite
        idx = [int(.45 * C), int(.25 * C), int(.10 * C)]
        out.append((np.dstack([stretch(cube[:, :, i]) for i in idx]), f"False colour, bands {idx} (NIR/red/green)"))
    elif names and all(b in names for b in ("B08", "B04", "B03")):
        idx = [names.index(b) for b in ("B08", "B04", "B03")]
        out.append((np.dstack([stretch(cube[:, :, i]) for i in idx]), "Colour infrared (B08/B04/B03): vegetation = red"))
        idx = [names.index(b) for b in ("B12", "B08", "B04")]
        out.append((np.dstack([stretch(cube[:, :, i]) for i in idx]), "SWIR composite (B12/B08/B04)"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--describe", help="print the contents of a .mat file and stop")
    ap.add_argument("--dataset", choices=["raichur", "indian_pines", "file"], default="indian_pines")
    ap.add_argument("--date", default="2026-10-12")
    ap.add_argument("--file")
    ap.add_argument("--labels")
    ap.add_argument("--n_gallery", type=int, default=16, help="number of single bands shown")
    a = ap.parse_args()

    if a.describe:
        describe_mat(a.describe)
        return

    cube, names, name = get_dataset(a, a.dataset)
    cube, info, _ = inspect_cube(cube, name, names)
    H, W, C = cube.shape
    out = HERE / "results" / name / "view"
    out.mkdir(parents=True, exist_ok=True)

    largs = types.SimpleNamespace(labels=a.labels, dataset=a.dataset, date=a.date, label_window=None, no_align=False)
    y = get_labels(largs, a.dataset, (H, W), cube, names)

    # ---- 1. composites (+ labels) ----
    comps = composites(cube, names)
    n = len(comps) + (1 if y is not None else 0)
    fig, ax = plt.subplots(1, n, figsize=(6 * n, 6))
    ax = np.atleast_1d(ax)
    for z, (img, title) in zip(ax, comps):
        z.imshow(img); z.set_title(title, fontsize=9)
    if y is not None:
        ax[len(comps)].imshow(np.ma.masked_equal(y, 0), cmap="tab20", interpolation="nearest")
        ax[len(comps)].set_title("Label map (0 = unlabelled, masked)", fontsize=9)
    for z in ax: z.set_xticks([]); z.set_yticks([])
    fig.suptitle(f"{name}: {H} x {W} pixels, {C} bands -- only 3 can be shown per picture", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "1_false_colour.png", dpi=110); plt.close(fig)

    # ---- 2. single-band gallery ----
    idx = np.unique(np.linspace(0, C - 1, min(a.n_gallery, C)).round().astype(int))
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    for z in np.atleast_1d(ax).ravel():
        z.axis("off")
    for z, i in zip(np.atleast_1d(ax).ravel(), idx):
        z.imshow(stretch(cube[:, :, i]), cmap="gray")
        z.set_title(f"band {i + 1}" + (f" ({names[i]})" if names else ""), fontsize=9)
    fig.suptitle(f"{name}: single bands (each is one wavelength; bright = high reflectance)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "2_band_gallery.png", dpi=100); plt.close(fig)

    # ---- 3. spectra ----
    X = cube.reshape(-1, C)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2 if y is not None else 1, figsize=(15 if y is not None else 8, 5))
    ax = np.atleast_1d(ax)
    x = np.arange(1, C + 1)
    mean, std = X.mean(0), X.std(0)
    ax[0].fill_between(x, mean - std, mean + std, alpha=.25, label="mean +/- 1 std (all pixels)")
    ax[0].plot(x, mean, lw=2, label="mean spectrum")
    for j, p in enumerate(rng.choice(len(X), 6, replace=False)):
        ax[0].plot(x, X[p], lw=.8, label="random pixels" if j == 0 else None, color="grey")
    ax[0].set_xlabel("band number" + (" (" + ", ".join(names) + ")" if names and C < 20 else ""), fontsize=8)
    ax[0].set_ylabel("value"); ax[0].set_title("Spectra: how reflectance changes with wavelength"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    if y is not None:
        yf = y.reshape(-1)
        for c in [int(c) for c in np.unique(yf) if c != 0 and (yf == c).sum() >= 10][:16]:
            ax[1].plot(x, X[yf == c].mean(0), label=f"class {c}")
        ax[1].set_xlabel("band number"); ax[1].set_title("Mean spectrum per label class (different materials = different curves)")
        ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "3_spectra.png", dpi=110); plt.close(fig)
    print(f"\n[saved] {out}\n  1_false_colour.png, 2_band_gallery.png, 3_spectra.png")


if __name__ == "__main__":
    main()