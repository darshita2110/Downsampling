"""
SHOW ABHISHEK - the picture before and after sending only K channels' worth of data,
for Abhishek's methods. Self-contained: needs only the Abhishek/ folder and the data in data/darshita/.

Abhishek's methods are FEATURE EXTRACTION: they do not keep original bands, they send K NEW
channels made by mixing all bands. Two families, plus references:

  Screen 1  PPCA family      PPCA (K components), PPCA + Recursive Filter (edge-preserving smoothing)
  Screen 2  WAVELET family   Wavelet-PCA (linear), WaveAE (wavelet + autoencoder, deep learning)
                             both send 4K channels at HALF resolution = the same data as K full channels
  Screen 3  FINAL            best of each family + Uniform bands (baseline)
  Screen 4  SCOREBOARD       PSNR, SAM and time of every method

Every method row shows 5 things:
  original | rebuilt on the ground | difference map + PSNR | what was SENT (first 3 new channels
  as colour) | one pixel's spectrum, original vs rebuilt

Scores: every band scaled to 0..1, same
90% test pixels, same PSNR (peak = 1) and SAM, same K (equal amount of data sent).

Run (from the repo folder):
    python Abhishek/show_abhishek.py raichur                    # Sentinel-2, K = 4 of 12
    python Abhishek/show_abhishek.py indian_pines               # K = 10 of 200
    python Abhishek/show_abhishek.py pavia_u --k 5
    python Abhishek/show_abhishek.py raichur --steps 1500       # train WaveAE longer (better, slower)
    python Abhishek/show_abhishek.py indian_pines --no-ae       # skip WaveAE (no PyTorch / no time)
    python Abhishek/show_abhishek.py raichur --only final --no-show
Needs: pip install opencv-contrib-python torch     (torch only for WaveAE)
Saved in Abhishek/results/show_<dataset>_K<k>/
"""
import argparse
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

_BACKEND = matplotlib.get_backend()   # the window backend; Abhishek's files switch to 'Agg' (no windows) when imported
import numpy as np

from universal_pipeline import ppca_fit, ppca_reconstruct, ppca_scores       # Abhishek's PPCA
from ppca_rf_pipeline import recursive_filter                                  # Abhishek's recursive filter

# ======================================================================
# small helpers (self-contained: works with only the Abhishek/ folder + data/darshita/)
# ======================================================================
DATA_DIR = "data/darshita"
BENCHMARKS = {"indian_pines": "Indian_pines_corrected.mat", "pavia_u": "PaviaU.mat", "salinas": "Salinas_corrected.mat",
              "ksc": "KSC.mat", "botswana": "Botswana.mat"}
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def _largest_3d(path):
    """Largest 3-D array inside a .npy / .mat file (MATLAB v7.3 files via h5py)."""
    if path.endswith(".npy"):
        return np.load(path)
    try:
        from scipy.io import loadmat
        arrays = [v for k, v in loadmat(path).items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            arrays = [np.array(f[k]).T for k in f.keys() if getattr(f[k], "ndim", 0) == 3]
    if not arrays:
        raise SystemExit(f"No 3-D image array found in {path}")
    return max(arrays, key=lambda a: a.size)


def load_dataset(name):
    """'indian_pines', 'pavia_u', 'salinas', 'ksc', 'botswana', 'raichur' (best date), 'raichur_<date>',
    or a path to any .mat / .npy cube.  Returns (cube (H, W, C) float32, channel names)."""
    if name in BENCHMARKS:
        path = os.path.join(DATA_DIR, BENCHMARKS[name])
    elif name == "raichur" or name.startswith("raichur_"):
        crop = (open(os.path.join(DATA_DIR, "raichur_best.txt")).read().strip() if name == "raichur"
                else f"{name}_crop.npy")
        path = os.path.join(DATA_DIR, crop)
    else:
        path = name
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}")
    cube = _largest_3d(path).astype(np.float32)
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:          # (bands, H, W) -> (H, W, bands)
        cube = np.moveaxis(cube, 0, -1)
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.where(cube.reshape(-1, cube.shape[2]).std(axis=0) > 0)[0]          # drop constant bands
    cube = cube[:, :, keep]
    names = S2_BANDS if name.startswith("raichur") and cube.shape[2] == 12 else [f"ch {i}" for i in keep]
    print(f"{dataset_name(name)}: {cube.shape[0]} x {cube.shape[1]} pixels, {cube.shape[2]} bands")
    return cube, names


def dataset_name(name):
    return os.path.splitext(os.path.basename(name))[0] if os.path.isfile(name) else name


def default_k(C):
    """10 channels for hyperspectral, about a third for few bands (4 of 12)."""
    return 10 if C > 40 else max(1, round(C / 3))


def to_table(cube):
    return cube.reshape(-1, cube.shape[2])


def scale_01(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def split_pixels(n, seed=0):
    """10% calibration pixels (learn the rebuild), 90% test pixels (score it). Same split as the team's scripts."""
    order = np.random.default_rng(seed).permutation(n)
    return order[:n // 10], order[n // 10:]


def rebuild(p01, channels, calib):
    """Rebuild all bands from the kept ones with least squares."""
    X = np.c_[p01[:, channels], np.ones(len(p01))]
    w, *_ = np.linalg.lstsq(X[calib], p01[calib], rcond=None)
    return np.clip(X @ w, 0, 1)


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def sam_degrees(a, b):
    den = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1) / den, -1, 1))).mean()


def uniform(C, k):
    return np.linspace(0, C - 1, k).round().astype(int).tolist()

p = argparse.ArgumentParser()
p.add_argument("dataset", nargs="?", default="raichur")
p.add_argument("--k", type=int, help="data budget in full-resolution channels (default: 4 for 12 bands, 10 for hyperspectral)")
p.add_argument("--steps", type=int, default=600, help="WaveAE training steps (his default is 3000; 600 is a quick demo)")
p.add_argument("--no-ae", action="store_true", help="skip WaveAE")
p.add_argument("--sigma-color", type=float, default=10, help="recursive filter: smaller keeps edges sharper")
p.add_argument("--only", choices=["ppca", "wavelet", "final"], help="show only this screen")
p.add_argument("--no-show", action="store_true", help="only save the pictures, open no window")
args = p.parse_args()
if args.no_show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                 # noqa: E402

COLOUR = {"PPCA": "#8a3ffc", "Wavelet": "#d12771", "Baseline": "#555555"}

# ------------------------------------------------------------ 1. data 
cube, names = load_dataset(args.dataset)
DATASET = dataset_name(args.dataset)
H, W, C = cube.shape
K = args.k or default_k(C)
raw = to_table(cube)
p01 = scale_01(raw)
calib, test = split_pixels(len(p01))
OUT = HERE / "results" / f"show_{DATASET}_K{K}"
OUT.mkdir(parents=True, exist_ok=True)
size_all, size_k = H * W * C * 2 / 1e6, H * W * K * 2 / 1e6

if DATASET.startswith("raichur"):
    RGB, rgb_note = [names.index("B04"), names.index("B03"), names.index("B02")], "true colour B04 B03 B02"
elif DATASET == "indian_pines":
    RGB, rgb_note = [29, 19, 9], "bands 29, 19, 9"
else:
    RGB = [int(C * 0.145), int(C * 0.095), int(C * 0.045)]
    rgb_note = f"bands {RGB[0]}, {RGB[1]}, {RGB[2]}"
orig = p01.reshape(H, W, C)
LO = [np.percentile(orig[:, :, b], 1) for b in RGB]
HI = [np.percentile(orig[:, :, b], 99) for b in RGB]


def colour(table01):
    c = table01.reshape(H, W, C)
    return np.dstack([np.clip((c[:, :, b] - lo) / (hi - lo + 1e-12), 0, 1) for b, lo, hi in zip(RGB, LO, HI)])


def stretch3(z):
    """First 3 sent channels as a colour picture (what actually goes down to the ground)."""
    out = [z[:, :, i] if i < z.shape[2] else np.zeros(z.shape[:2]) for i in range(3)]
    return np.dstack([np.clip((a - np.percentile(a, 1)) / (np.percentile(a, 99) - np.percentile(a, 1) + 1e-12), 0, 1)
                      for a in out])


# ------------------------------------------------------------ 2. Abhishek's methods
print(f"\n{DATASET}: {H} x {W} pixels, {C} channels. Data budget K = {K} channels "
      f"({100 * K / C:.0f}% of the data, {size_all:.1f} MB -> {size_k:.1f} MB)\n")
R = {}


def add(name, family, rebuilt01, sent, secs, note):
    rebuilt01 = np.clip(rebuilt01.reshape(-1, C), 0, 1)
    R[name] = {"family": family, "rebuilt": rebuilt01, "sent": sent, "time": secs, "note": note,
               "psnr": psnr(p01[test], rebuilt01[test]), "sam": sam_degrees(p01[test], rebuilt01[test]),
               "error": np.abs(p01 - rebuilt01).mean(axis=1).reshape(H, W)}
    print(f"  {name:28s} PSNR {R[name]['psnr']:6.2f} dB   SAM {R[name]['sam']:5.2f}°   {secs:6.2f} s")


# --- PPCA family (Abhishek's own functions) ---
t = time.perf_counter()
m = ppca_fit(p01)                                   # standardises every band, eigen-decomposition
Z, back, s2 = ppca_scores(m, K)                     # K new channels per pixel = what is sent
rec_s = ppca_reconstruct(m, Z, back, K)             # ground: rebuild the standardised bands
t_ppca = time.perf_counter() - t
side = lambda n: f" + basis {100 * n / (H * W * K):.1f}% extra"     # numbers the ground needs to decode
add("PPCA", "PPCA", rec_s * m["sd"] + m["mu"], Z.reshape(H, W, K), t_ppca,
    f"{K} new channels, full resolution" + side(C * (K + 2)))

t = time.perf_counter()
sig_sp = max(3.0, round(min(H, W) / 80))            # his default
Zs = recursive_filter(Z.reshape(H, W, K).astype(np.float32), sig_sp, args.sigma_color, 3).reshape(-1, K)
rec_s = ppca_reconstruct(m, Zs, back, K)
add("PPCA + Recursive Filter", "PPCA", rec_s * m["sd"] + m["mu"], Zs.reshape(H, W, K),
    t_ppca + time.perf_counter() - t, f"{K} smoothed channels (sigma_spatial {sig_sp:g})" + side(C * (K + 2)))

# --- Wavelet family: half resolution, 4K channels = same data as K full-resolution channels ---
try:
    import torch
    from wave_ae import haar_dwt2, pad_even, pca_fit, train_ae, ae_encode_decode, up2, wave_to_cube
    have_torch = True
except ImportError:
    have_torch = False
    print("  (PyTorch not installed: WaveAE skipped. pip install torch)")
if have_torch:
    k_ae = min(4 * K, 4 * C)
    mu, sd = m["mu"], m["sd"]
    Xs = m["Xs"]
    h2 = lambda z: np.concatenate(haar_dwt2(pad_even(z.reshape(H, W, C))), axis=2).astype(np.float32)
    t = time.perf_counter()
    wv = h2(Xs)                                                 # (H/2, W/2, 4C): LL, LH, HL, HH of every band
    hh, ww = wv.shape[:2]
    Wflat = wv.reshape(-1, 4 * C)
    muw, Uw, _ = pca_fit(Wflat)
    Zw = (Wflat - muw) @ Uw[:, :k_ae]                           # sent: k_ae channels at half resolution
    recw = wave_to_cube((Zw @ Uw[:, :k_ae].T + muw).reshape(hh, ww, 4 * C), C, H, W)
    add("Wavelet-PCA", "Wavelet", recw.reshape(-1, C) * sd + mu, up2(Zw.reshape(hh, ww, k_ae), H, W),
        time.perf_counter() - t, f"{k_ae} ch at half resolution (linear)" + side(4 * C * (k_ae + 1) + 2 * C))
    if not args.no_ae:
        print(f"  training WaveAE ({args.steps} steps on CPU/GPU, starts from Wavelet-PCA) ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = types.SimpleNamespace(seed=0, width=128, lr=2e-3, steps=args.steps, detail_weight=1.0, patch=32, batch=16)
        t = time.perf_counter()
        model, losses, _ = train_ae(wv, C, k_ae, a, device, (muw, Uw[:, :k_ae]))
        z, recwv, _ = ae_encode_decode(model, wv, device)
        reca = wave_to_cube(recwv, C, H, W)
        add("WaveAE", "Wavelet", reca.reshape(-1, C) * sd + mu, up2(z, H, W), time.perf_counter() - t,
            f"{k_ae} ch at half res, deep learning, {args.steps} steps" + f" + decoder {100 * model.n_params('dec') / (H * W * K):.0f}% (send once)")

# --- reference: Uniform bands (K evenly spaced original bands, no thinking) ---
group_u = uniform(C, K)
add("Uniform bands", "Baseline", rebuild(p01, group_u, calib), orig[:, :, group_u], 0.0,
    "sent " + (" ".join(names[c] for c in group_u) if K <= 8 else f"{K} evenly spaced bands"))

ERR_MAX = np.percentile(R["Uniform bands"]["error"], 99.5) * 1.5
PIXEL = test[np.argmax(R["Uniform bands"]["error"].ravel()[test])]


# ------------------------------------------------------------ 3. screens
if not args.no_show:                         # undo the 'Agg' set by universal_pipeline / ppca_rf_pipeline / wave_ae
    try:
        plt.switch_backend(_BACKEND if _BACKEND.lower() != "agg" else "TkAgg")
    except Exception as e:
        print(f"  (cannot open windows: {e}. The pictures are still saved.)")
def screen(rows, title, filename, note):
    fig, axes = plt.subplots(len(rows), 5, figsize=(21, 3.7 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1.1, 1, 1.6]})
    for (a0, a1, a2, a3, a4), name in zip(axes, rows):
        r = R[name]
        col = COLOUR[r["family"]]
        a0.imshow(colour(p01))
        a0.set_title(f"ORIGINAL, all {C} channels ({size_all:.1f} MB)", fontsize=9.5)
        a1.imshow(colour(r["rebuilt"]))
        a1.set_title(f"{name}: REBUILT on ground", fontsize=9.5, color=col, fontweight="bold")
        im = a2.imshow(r["error"], cmap="magma", vmin=0, vmax=ERR_MAX)
        a2.set_title(f"DIFFERENCE: PSNR {r['psnr']:.2f} dB, SAM {r['sam']:.2f}°\n(brighter = more lost)", fontsize=9.5)
        fig.colorbar(im, ax=a2, fraction=0.046, pad=0.02)
        a3.imshow(stretch3(r["sent"]))
        a3.set_title(f"SENT ({size_k:.1f} MB): first 3 channels as colour\n{r['note']}", fontsize=8.5)
        for ax in (a0, a1, a2, a3):
            ax.set_xticks([]), ax.set_yticks([])
        x = np.arange(C)
        a4.plot(x, p01[PIXEL], color="black", lw=1.8, label="original")
        a4.plot(x, r["rebuilt"][PIXEL], color=col, lw=1.8, ls="--", label="rebuilt")
        a4.set_title(f"One pixel's spectrum, {name}\nencode + rebuild in {r['time']:.2f} s", fontsize=9.5)
        if C <= 20:
            a4.set_xticks(x, names, rotation=90, fontsize=8)
        a4.set_ylim(-0.05, 1.05)
        a4.grid(alpha=0.3)
        a4.legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"{title}   |   {DATASET}: {K} of {C} channels' worth of data = {100 * K / C:.0f}%\n{note}",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, f"Pictures: {rgb_note}, same colour stretch in all. Error maps: same scale in all. "
             "PSNR/SAM on the same 90% test pixels as the team's results.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.95))
    fig.savefig(OUT / f"{filename}.png", dpi=100)
    print(f"  saved {OUT / filename}.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)


def table(title, rows):
    print(f"\n{title}\n  {'Method':28s} {'PSNR dB':>8s} {'SAM °':>7s} {'Time s':>7s}  Sent")
    for n in rows:
        r = R[n]
        print(f"  {n:28s} {r['psnr']:8.2f} {r['sam']:7.2f} {r['time']:7.2f}  {r['note']}")


by_psnr = lambda ns: sorted(ns, key=lambda n: -R[n]["psnr"])
fam = {f: by_psnr([n for n in R if R[n]["family"] == f]) for f in ("PPCA", "Wavelet")}
NOTES = {"PPCA": "PPCA mixes ALL bands into K new channels (the main 'stories' of the data). The recursive filter then "
                 "smooths each channel but keeps edges: less noise, slightly lower PSNR.",
         "Wavelet": "Each band is split into a half-size smooth image + 3 edge images (Haar wavelet). 4K channels at half "
                    "size = same data as K full channels. WaveAE learns a non-linear version of Wavelet-PCA."}
for key, file, title in [("ppca", "1_ppca", "PPCA family"), ("wavelet", "2_wavelet", "WAVELET family")]:
    rows = fam[title.split()[0].upper() if key == "ppca" else "Wavelet"]
    if not rows:
        continue
    table(f"{title} (best first)", rows)
    if args.only in (None, key):
        screen(rows, title, file, NOTES["PPCA" if key == "ppca" else "Wavelet"])

final = by_psnr([f[0] for f in fam.values() if f] + ["Uniform bands"])
table("FINAL: best of each family vs Uniform bands", final)
if args.only in (None, "final"):
    screen(final, "FINAL: best of each family vs Uniform bands", "3_final",
           "Uniform = K evenly spaced original bands. Same amount of data for all.")
    order = by_psnr(list(R))
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.5 * len(order) + 2), sharey=True)
    cols = [COLOUR[R[n]["family"]] for n in order]
    for ax, key, label in zip(axes, ["psnr", "sam", "time"],
                              ["PSNR (dB), higher = better", "SAM (degrees), lower = better", "Time (s), lower = better"]):
        vals = [R[n][key] for n in order]
        ax.barh(order, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(min(vals) * 0.9 if key == "psnr" else 0, max(vals) * 1.18 + 1e-9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    used = [f for f in COLOUR if any(R[n]["family"] == f for n in R)]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=COLOUR[f]) for f in used], used, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"SCOREBOARD   |   {DATASET}: {K} of {C} channels' worth of data", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "4_scoreboard.png", dpi=100)
    print(f"  saved {OUT / '4_scoreboard.png'}")
    if not args.no_show:
        plt.show()
    plt.close(fig)

import csv                                                                            # noqa: E402
with open(OUT / "scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Method", "Family", "K", "PSNR", "SAM", "Time (s)", "Sent"])
    for n in by_psnr(list(R)):
        w.writerow([n, R[n]["family"], K, f"{R[n]['psnr']:.4f}", f"{R[n]['sam']:.4f}", f"{R[n]['time']:.3f}", R[n]["note"]])
print(f"\nSaved in {OUT}/ : 1_ppca.png, 2_wavelet.png, 3_final.png, 4_scoreboard.png, scores.csv")