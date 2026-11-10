"""
STEP 1 - Load a hyperspectral image and look at it.

What you learn here:
  - a hyperspectral image is a 3-D array: height x width x bands
  - each band is a grey-scale picture of the same place in one narrow colour
  - each pixel has a "spectrum": its value in every band

Run (from the repo folder):
    python darshita/step1_load_and_look.py

The first run downloads Indian Pines (about 6 MB), the standard practice image.
"""
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------- settings
DATA_DIR = "data/darshita"                 # data/ is in .gitignore, so this never goes to GitHub
OUT_DIR = "darshita/outputs"               # pictures from each step are saved here
IMAGE_FILE = os.path.join(DATA_DIR, "Indian_pines_corrected.mat")
IMAGE_URL = "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines_corrected.mat"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------- 1. get the image
if not os.path.exists(IMAGE_FILE):
    print("Downloading Indian Pines ...")
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_FILE)

# A .mat file is a MATLAB file. loadmat gives a dictionary; the image is under this key.
image = loadmat(IMAGE_FILE)["indian_pines_corrected"].astype(np.float32)

# ------------------------------------------------------- 2. basic facts
height, width, n_bands = image.shape
print("Shape of the image :", image.shape)
print(f"  {height} x {width} pixels, {n_bands} bands")
print(f"  smallest value {image.min():.0f}, largest {image.max():.0f}")
print(f"  memory: {image.nbytes / 1e6:.1f} MB")

# ------------------------------------------------------- 3. look at 3 bands
# image[:, :, b] means: all rows, all columns, band number b -> one grey picture
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, b in zip(axes, [10, 100, 180]):
    ax.imshow(image[:, :, b], cmap="gray")
    ax.set_title(f"Band {b}")
    ax.axis("off")
plt.suptitle("The same place seen in three different bands")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/step1_three_bands.png", dpi=120)
print(f"Saved {OUT_DIR}/step1_three_bands.png")

# ------------------------------------------------------- 4. look at spectra
# image[row, col, :] means: one pixel, all bands -> that pixel's spectrum (a curve)
plt.figure(figsize=(8, 4))
for row, col in [(20, 20), (70, 70), (120, 40)]:
    plt.plot(image[row, col, :], label=f"pixel ({row}, {col})")
plt.xlabel("Band number")
plt.ylabel("Value")
plt.title("Spectra of three pixels")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/step1_spectra.png", dpi=120)
print(f"Saved {OUT_DIR}/step1_spectra.png")

# ------------------------------------------------------- 5. the key observation
# Neighbouring bands are almost the same picture. This is WHY we can throw most bands away.
corr = np.corrcoef(image[:, :, 50].ravel(), image[:, :, 51].ravel())[0, 1]
print(f"\nCorrelation between band 50 and band 51: {corr:.3f}  (1.0 = identical)")
corr = np.corrcoef(image[:, :, 10].ravel(), image[:, :, 180].ravel())[0, 1]
print(f"Correlation between band 10 and band 180: {corr:.3f}")
