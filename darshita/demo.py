"""
DEMO for the teacher - ranking-based channel selection, told as a story in 5 screens.

Run from the repo folder:
    python darshita/demo.py

Each screen opens in a window. Explain it, close the window, and the next one opens.
All screens are also saved in darshita/outputs/demo_*.png (use those if a window does not open).
"""
import os
import sys
import urllib.request

import matplotlib
if "--save-only" in sys.argv:          # no windows, just save the pictures
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from ranking import entropy_scores, mvpca_scores, top_k, variance_scores   # our own file

K = 10                                        # channels we are allowed to send
IMAGE_FILE = "data/darshita/Indian_pines_corrected.mat"
IMAGE_URL = "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines_corrected.mat"
OUT = "darshita/outputs"
RGB = (29, 19, 9)                             # channels that look like red, green, blue light
COLORS = {"Variance": "#2a78d6", "Entropy": "#eb6834", "MVPCA": "#1baf7a", "Uniform": "#555555"}
SHOW = "--save-only" not in sys.argv
os.makedirs(OUT, exist_ok=True)


def screen(fig, name, text):
    """Print what to say, save the picture, show it and wait until the window is closed."""
    path = f"{OUT}/demo_{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(text)
    print(f"   (picture saved: {path})")
    if SHOW:
        print("   -> close the window to go to the next screen\n")
        plt.show()
    plt.close(fig)


def colour_picture(cube):
    """Make a normal-looking colour photo from 3 channels, stretched for display."""
    out = []
    for c in RGB:
        ch = cube[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        out.append(np.clip((ch - lo) / (hi - lo), 0, 1))
    return np.dstack(out)


# ------------------------------------------------------------------ load
if not os.path.exists(IMAGE_FILE):
    os.makedirs(os.path.dirname(IMAGE_FILE), exist_ok=True)
    print("Downloading the image (6 MB) ...")
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_FILE)
cube = loadmat(IMAGE_FILE)["indian_pines_corrected"].astype(np.float32)
H, W, C = cube.shape
raw = cube.reshape(-1, C)
size_mb = cube.astype(np.int16).nbytes / 1e6     # the satellite stores 16-bit numbers

print("=" * 70)
print(" CHANNEL SELECTION FOR SATELLITE DOWNLINKING  -  ranking-based methods")
print("=" * 70 + "\n")

# ======================================================= SCREEN 1: the image
fig = plt.figure(figsize=(15, 6.5))
fig.suptitle(f"SCREEN 1 - What the satellite captured: one place, {C} channels", fontsize=15, fontweight="bold", y=1.02)
ax = fig.add_axes([0.02, 0.04, 0.30, 0.78])
ax.imshow(colour_picture(cube))
ax.set_title(f"Colour picture of a farm area (from 3 channels)\n{H} x {W} pixels", fontsize=11)
ax.axis("off")
shown = list(range(0, C, 9))[:22]                                 # 22 of the 200 channels
for i, c in enumerate(shown):
    a = fig.add_axes([0.35 + (i % 11) * 0.059, 0.44 - (i // 11) * 0.34, 0.055, 0.28])
    a.imshow(cube[:, :, c], cmap="gray")
    a.set_title(f"ch {c}", fontsize=9)
    a.axis("off")
fig.text(0.66, 0.84, "Each small grey picture is ONE channel (also called a band):\n"
         "the same place photographed in one colour of light.  22 of the 200 are shown.",
         ha="center", fontsize=11)
screen(fig, "1_image", f"""SCREEN 1 - THE IMAGE
   Our eyes see 3 channels (red, green, blue). This satellite sees {C} channels:
   the same place photographed in {C} different colours of light, from visible to infrared.
   Each grey picture on the right is one channel. Total data: {size_mb:.1f} MB for one small image.
""")

# ======================================================= SCREEN 2: the problem
corr_near = np.corrcoef(raw[:, 50], raw[:, 51])[0, 1]
corr_far = np.corrcoef(raw[:, 10], raw[:, 180])[0, 1]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
fig.suptitle("SCREEN 2 - Why we can throw channels away: neighbouring channels are copies",
             fontsize=15, fontweight="bold")
for ax, c in zip(axes, [50, 51, 10, 180]):
    ax.imshow(cube[:, :, c], cmap="gray")
    ax.set_title(f"channel {c}", fontsize=12)
    ax.axis("off")
fig.text(0.30, 0.02, f"channel 50 vs 51: {100 * corr_near:.1f}% similar  ->  sending both is a waste",
         ha="center", fontsize=12, color="#b00020")
fig.text(0.72, 0.02, f"channel 10 vs 180: {100 * corr_far:.1f}% similar  ->  different information",
         ha="center", fontsize=12, color="#1a7f37")
screen(fig, "2_problem", f"""SCREEN 2 - THE PROBLEM AND THE IDEA
   The satellite's downlink (radio link to the ground) is slow: it cannot send all {C} channels.
   But channel 50 and 51 are {100 * corr_near:.1f}% the same. So we send only K = {K} channels
   ({100 * K / C:.0f}% of the data) and rebuild the other {C - K} on the ground.
   QUESTION: which {K} channels should we send?
""")

# ======================================================= SCREEN 3: ranking picks
scores = {"Variance": variance_scores(raw), "Entropy": entropy_scores(raw), "MVPCA": mvpca_scores(raw, K)}
picked = {m: top_k(s, K) for m, s in scores.items()}
picked["Uniform"] = np.linspace(0, C - 1, K).round().astype(int).tolist()

fig = plt.figure(figsize=(15, 8))
fig.suptitle(f"SCREEN 3 - Ranking: score every channel alone, keep the top {K}",
             fontsize=15, fontweight="bold")
strip = fig.add_axes([0.08, 0.62, 0.88, 0.26])
for row, m in enumerate(picked):
    strip.barh(row, C, left=0, height=0.6, color="#eeeeee")
    strip.vlines(picked[m], row - 0.3, row + 0.3, color=COLORS[m], lw=4)
strip.set_yticks(range(len(picked)))
strip.set_yticklabels([f"{m}" for m in picked], fontsize=11)
strip.invert_yaxis()
strip.set_xlim(0, C)
strip.set_xlabel(f"channel number (0 to {C - 1})   -   each coloured line = one channel that will be sent")
for row, m in enumerate(["Variance", "Uniform"]):
    fig.text(0.02, 0.43 - row * 0.25, f"{m}\nsends\nthese {K}:", fontsize=11, color=COLORS[m],
             fontweight="bold", va="center")
    for i, c in enumerate(picked[m]):
        a = fig.add_axes([0.09 + i * 0.089, 0.32 - row * 0.25, 0.08, 0.2])
        a.imshow(cube[:, :, c], cmap="gray")
        a.set_title(f"ch {c}", fontsize=9)
        a.axis("off")
screen(fig, "3_selection", f"""SCREEN 3 - WHICH CHANNELS EACH METHOD SENDS
   Ranking gives each channel a score ON ITS OWN and keeps the top {K}:
     Variance = how much the channel's values change      -> sends {picked['Variance']}
     Entropy  = how much information the channel holds    -> sends {picked['Entropy']}
     MVPCA    = how much it feeds the main PCA patterns   -> sends {picked['MVPCA']}
     Uniform  = no thinking, every 20th channel (baseline) -> sends {picked['Uniform']}
   Look: the ranking methods pick channels that sit next to each other.
   Variance's {K} pictures all look the same; Uniform's {K} pictures are all different.
""")

# ======================================================= SCREEN 4: before vs after
low, high = raw.min(0), raw.max(0)
norm = (raw - low) / (high - low)                            # every channel 0..1
order = np.random.default_rng(0).permutation(len(norm))
calib, test = order[: len(norm) // 10], order[len(norm) // 10:]


def rebuild(chans):
    X = np.c_[norm[:, chans], np.ones(len(norm))]
    w, *_ = np.linalg.lstsq(X[calib], norm[calib], rcond=None)
    return np.clip(X @ w, 0, 1)


results = {}
for m, chans in picked.items():
    r = rebuild(chans)
    mse = np.mean((norm[test] - r[test]) ** 2)
    results[m] = (r, 10 * np.log10(1 / mse))

fig, axes = plt.subplots(2, 5, figsize=(16, 7.4))
fig.suptitle(f"SCREEN 4 - BEFORE vs AFTER: send {K} channels, rebuild all {C} on the ground",
             fontsize=15, fontweight="bold")
orig = colour_picture(norm.reshape(H, W, C))
axes[0, 0].imshow(orig)
axes[0, 0].set_title(f"BEFORE: original\n{C} channels, {size_mb:.1f} MB", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")
axes[1, 0].text(0.5, 0.5, "Bottom row = error\n(bright = wrong,\ndark = correct)", ha="center",
                va="center", fontsize=11, transform=axes[1, 0].transAxes)
for i, (m, (r, p)) in enumerate(results.items(), start=1):
    pic = colour_picture(r.reshape(H, W, C))
    axes[0, i].imshow(pic)
    axes[0, i].set_title(f"AFTER: rebuilt from {m}'s\n{K} channels ({size_mb * K / C:.2f} MB)  {p:.1f} dB",
                         fontsize=10, color=COLORS[m], fontweight="bold")
    err = np.abs(norm - r).mean(1).reshape(H, W)
    axes[1, i].imshow(err, cmap="magma", vmin=0, vmax=0.12)
    axes[1, i].set_title(f"error of {m}", fontsize=10)
for a in axes.ravel():
    a.set_xticks([]), a.set_yticks([])
fig.tight_layout()
screen(fig, "4_before_after", f"""SCREEN 4 - BEFORE AND AFTER
   Only {K} channels are sent ({size_mb:.1f} MB -> {size_mb * K / C:.2f} MB). On the ground, all {C} channels
   are rebuilt from them with linear regression. Top row: rebuilt colour pictures.
   Bottom row: where the rebuild is wrong (bright = big error). Uniform's error map is the darkest.
""")

# ======================================================= SCREEN 5: scoreboard
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]})
fig.suptitle("SCREEN 5 - Scoreboard: how close is the rebuilt image to the original?",
             fontsize=15, fontweight="bold")
names = sorted(results, key=lambda m: results[m][1], reverse=True)
vals = [results[m][1] for m in names]
bars = a1.barh(names, vals, color=[COLORS[m] for m in names])
a1.invert_yaxis()
a1.set_xlim(0, max(vals) + 4)
for b, v in zip(bars, vals):
    a1.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.2f} dB", va="center", fontsize=12)
a1.set_xlabel("PSNR in dB  (higher = rebuilt image closer to the original)")
a1.set_title(f"Sending {K} of {C} channels", fontsize=12)

redund = {}
for m, chans in picked.items():
    cc = np.abs(np.corrcoef(raw[:, chans], rowvar=False))
    redund[m] = (cc.sum() - K) / (K * (K - 1))
a2.barh(names, [100 * redund[m] for m in names], color=[COLORS[m] for m in names])
a2.invert_yaxis()
for i, m in enumerate(names):
    a2.text(100 * redund[m] + 1, i, f"{100 * redund[m]:.1f}%", va="center", fontsize=12)
a2.set_xlim(0, 115)
a2.set_xlabel("how similar the sent channels are to each other (lower = better)")
a2.set_title("Repetition inside the sent channels", fontsize=12)
fig.tight_layout()
best = names[0]
screen(fig, "5_scoreboard", f"""SCREEN 5 - RESULT
   PSNR: {', '.join(f'{m} {results[m][1]:.2f} dB' for m in names)}
   Winner: {best}. All three ranking methods lose to simply taking every 20th channel.
   WHY: ranking judges each channel alone, so it sends near-copies
   ({100 * min(redund[m] for m in ['Variance', 'Entropy', 'MVPCA']):.1f}-{100 * max(redund[m] for m in ['Variance', 'Entropy', 'MVPCA']):.1f}% similar to each other, Uniform only {100 * redund['Uniform']:.1f}%).
   NEXT: search-based (GA, PSO) and clustering-based methods, which avoid sending copies.
""")
print("=" * 70 + "\n DEMO FINISHED\n" + "=" * 70)
