"""
DATA STEP 2 - Check the downloaded Raichur images and cut a clean window from each.

For every date it:
  1. prints the size, value range and how much of the scene is cloud / shadow / no data
  2. finds the cleanest 512 x 512 window (least cloud, and a mix of fields, bare land and water)
  3. saves that window as data/darshita/raichur_<date>_crop.npy  <- the algorithms run on this
  4. saves a picture darshita/outputs/data_raichur_<date>.png to look at

Run (after data step 1):
    python darshita/data_2_check_s2.py
"""
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

DATA_DIR = "data/darshita"
OUT_DIR = "darshita/outputs"
WINDOW = 512                                    # size of the clean window, pixels (512 x 20 m = 10 km)
os.makedirs(OUT_DIR, exist_ok=True)

# SCL = Sentinel-2 "scene classification": one class number per pixel
BAD = [0, 1, 3, 8, 9, 10]                       # no data, defective, cloud shadow, clouds, cirrus
NAMES = {0: "no data", 1: "defective", 2: "dark area", 3: "cloud shadow", 4: "vegetation",
         5: "bare land", 6: "water", 7: "unclassified", 8: "cloud (medium)", 9: "cloud (high)",
         10: "thin cirrus", 11: "snow"}

info = json.load(open(os.path.join(DATA_DIR, "raichur_bands.json")))
BANDS, WAVES = info["bands"], info["wavelength_nm"]


def true_colour(cube):
    """Red, green, blue bands (B04, B03, B02) stretched to look like a normal photo."""
    rgb = cube[:, :, [BANDS.index("B04"), BANDS.index("B03"), BANDS.index("B02")]]
    lo, hi = np.percentile(rgb[rgb > 0], [1, 99]) if (rgb > 0).any() else (0, 1)
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)


def window_sums(mask, size, step):
    """Number of True pixels in every size x size window (fast, using a summed-area table)."""
    s = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.unique(np.r_[np.arange(0, mask.shape[0] - size + 1, step), mask.shape[0] - size])
    xs = np.unique(np.r_[np.arange(0, mask.shape[1] - size + 1, step), mask.shape[1] - size])
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return ys, xs, s[Y + size, X + size] - s[Y, X + size] - s[Y + size, X] + s[Y, X]


def best_window(scl, size):
    """Cleanest window; among equally clean ones, the one with the best mix of land types."""
    size = min(size, *scl.shape)
    ys, xs, bad = window_sums(np.isin(scl, BAD), size, step=32)
    clean_enough = bad <= bad.min() + 0.005 * size * size
    mix = np.zeros(bad.shape)
    shares = [window_sums(scl == c, size, 32)[2] / (size * size) for c in (4, 5, 6)]
    for sh in shares:                                    # entropy of vegetation / bare / water shares
        mix -= np.where(sh > 0, sh * np.log(np.maximum(sh, 1e-12)), 0)
    score = np.where(clean_enough, mix, -np.inf)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return int(ys[i]), int(xs[j]), size, bad[i, j] / (size * size)


images = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "raichur_*.npy"))
                if not f.endswith(("_scl.npy", "_crop.npy")))
if not images:
    raise SystemExit("No data yet. Run: python darshita/data_1_download_s2.py")

summary = []
for path in images:
    day = os.path.basename(path)[len("raichur_"):-len(".npy")]
    cube = np.load(path)
    scl = np.load(path.replace(".npy", "_scl.npy"))
    H, W, C = cube.shape
    valid = scl != 0
    print(f"\n=== {day} ===")
    print(f"  size        : {H} x {W} pixels, {C} bands ({', '.join(BANDS)})")
    print(f"  reflectance : {cube[valid].min():.3f} to {cube[valid].max():.3f} (should be roughly 0 to 1)")
    counts = {NAMES[c]: 100 * np.mean(scl == c) for c in np.unique(scl)}
    for name, pct in sorted(counts.items(), key=lambda kv: -kv[1]):
        if pct >= 0.5:
            print(f"  {name:15s} {pct:5.1f}% of the scene")
    bad_pct = 100 * np.mean(np.isin(scl, BAD))

    y, x, size, bad_in_window = best_window(scl, WINDOW)
    crop = cube[y:y + size, x:x + size]
    crop_scl = scl[y:y + size, x:x + size]
    np.save(os.path.join(DATA_DIR, f"raichur_{day}_crop.npy"), crop)
    print(f"  clean window: rows {y}-{y + size}, cols {x}-{x + size}, "
          f"{100 * bad_in_window:.2f}% cloud/no-data inside")
    print(f"  saved       : data/darshita/raichur_{day}_crop.npy  shape {crop.shape}")
    summary.append((day, bad_pct, 100 * bad_in_window))

    # ------------------------------------------------------------ picture
    fig = plt.figure(figsize=(17, 9.5))
    fig.suptitle(f"Raichur, Sentinel-2 L2A, {day}: {C} real bands at 20 m", fontsize=15, fontweight="bold")
    a = fig.add_axes([0.01, 0.42, 0.30, 0.50])
    a.imshow(true_colour(cube))
    a.add_patch(Rectangle((x, y), size, size, fill=False, ec="yellow", lw=2))
    a.set_title(f"Whole scene (true colour)  -  yellow = clean window used", fontsize=10)
    a.axis("off")
    a = fig.add_axes([0.33, 0.42, 0.30, 0.50])
    bad_map = np.isin(scl, BAD)
    a.imshow(np.where(bad_map[..., None], [1.0, 0.2, 0.2], true_colour(cube) * 0.5))
    a.set_title(f"Red = cloud, shadow or no data ({bad_pct:.1f}% of scene)", fontsize=10)
    a.axis("off")
    a = fig.add_axes([0.685, 0.42, 0.305, 0.50])
    for cls, colour in [(4, "green"), (5, "saddlebrown"), (6, "royalblue")]:
        m = crop_scl == cls
        if m.sum() > 50:
            a.plot(WAVES, crop[m].mean(0), marker="o", color=colour, lw=2,
                   label=f"{NAMES[cls]} ({100 * m.mean():.0f}% of window)")
    a.set_xlabel("wavelength (nm)")
    a.set_ylabel("reflectance")
    a.set_title("Average spectrum of each land type in the window", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=9)
    for i, b in enumerate(BANDS):
        ax = fig.add_axes([0.01 + i * 0.0825, 0.05, 0.078, 0.30])
        band = crop[:, :, i]
        lo, hi = np.percentile(band, [1, 99])
        ax.imshow(band, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"{b}\n{WAVES[i]} nm", fontsize=9)
        ax.axis("off")
    fig.text(0.5, 0.37, "The 12 bands of the clean window: same place, 12 colours of light",
             ha="center", fontsize=11)
    out = os.path.join(OUT_DIR, f"data_raichur_{day}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  picture     : {out}")

print("\nSummary (lower is cleaner):")
for day, scene, window in summary:
    print(f"  {day}: {scene:5.1f}% cloud/no-data in the scene, {window:.2f}% in the window used")
best = min(summary, key=lambda s: (s[2], s[1]))[0]
print(f"\nCleanest date: {best}. The algorithm scripts use data/darshita/raichur_{best}_crop.npy")
with open(os.path.join(DATA_DIR, "raichur_best.txt"), "w") as f:
    f.write(f"raichur_{best}_crop.npy\n")
