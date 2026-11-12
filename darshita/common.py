"""
common.py - small helpers used by every step: load the image, rebuild it, score it.

The downlink test used everywhere in this project:
  1. the satellite sends only the chosen channels
  2. on the ground, every channel is rebuilt from them with linear regression
     (weights learned on 10% of the pixels = "calibration" pixels)
  3. the rebuilt image is compared with the original on the other 90% of the pixels
"""
import json
import os
import urllib.request

import numpy as np
from scipy.io import loadmat

IMAGE_FILE = "data/darshita/Indian_pines_corrected.mat"
IMAGE_URL = "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines_corrected.mat"
OUT_DIR = "darshita/outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_image():
    """Return the image cube, shape (height, width, channels). Downloads it the first time."""
    if not os.path.exists(IMAGE_FILE):
        os.makedirs(os.path.dirname(IMAGE_FILE), exist_ok=True)
        print("Downloading Indian Pines (6 MB) ...")
        urllib.request.urlretrieve(IMAGE_URL, IMAGE_FILE)
    return loadmat(IMAGE_FILE)["indian_pines_corrected"].astype(np.float32)


def load_dataset(name):
    """
    name = "indian_pines" -> practice image, 200 channels (aircraft, real)
    name = "raichur"      -> our real Sentinel-2 satellite data, 12 bands (run data steps 1 and 2 first)
    Returns (cube, channel_names).
    """
    if name == "indian_pines":
        cube = load_image()
        return cube, [f"ch {i}" for i in range(cube.shape[2])]
    if name == "raichur":
        best = os.path.join("data/darshita", "raichur_best.txt")
        if not os.path.exists(best):
            raise SystemExit("No Raichur data yet. Run data_1_download_s2.py, then data_2_check_s2.py")
        crop = open(best).read().strip()
        cube = np.load(os.path.join("data/darshita", crop)).astype(np.float32)
        info = json.load(open("data/darshita/raichur_bands.json"))
        print(f"Raichur data: {crop}")
        return cube, info["bands"]
    raise SystemExit(f"Unknown dataset '{name}'. Use indian_pines or raichur")


def default_k(n_channels):
    """How many channels to send by default: 10 of 200 (5%), or 4 of 12 for Sentinel-2."""
    return 10 if n_channels > 40 else 4


def k_values(n_channels):
    """K values for the 'quality vs number of channels' graph."""
    return [5, 10, 15, 20, 25, 30] if n_channels > 40 else [2, 3, 4, 5, 6, 8]


def to_table(cube):
    """(height, width, channels) -> table with one row per pixel, one column per channel."""
    return cube.reshape(-1, cube.shape[2])


def scale_01(pixels):
    """Scale every channel to 0..1 so every channel counts equally in the error."""
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    return (pixels - low) / (high - low)


def split_pixels(n_pixels, seed=0):
    """10% calibration pixels (to learn the rebuild), 90% test pixels (to score it)."""
    order = np.random.default_rng(seed).permutation(n_pixels)
    n_cal = n_pixels // 10
    return order[:n_cal], order[n_cal:]


def rebuild(pixels01, channels, calib):
    """Rebuild ALL channels from the chosen ones with linear regression (least squares)."""
    X = np.c_[pixels01[:, channels], np.ones(len(pixels01))]     # chosen channels + constant
    weights, *_ = np.linalg.lstsq(X[calib], pixels01[calib], rcond=None)
    return np.clip(X @ weights, 0, 1)


def psnr(original, rebuilt):
    """Peak signal-to-noise ratio in dB. Higher = rebuilt image closer to the original."""
    mse = np.mean((original - rebuilt) ** 2)
    return 10 * np.log10(1.0 / mse)


def sam_degrees(original, rebuilt):
    """Spectral angle: how much each pixel's spectrum curve is bent, averaged. Lower = better."""
    dot = np.sum(original * rebuilt, axis=1)
    norms = np.linalg.norm(original, axis=1) * np.linalg.norm(rebuilt, axis=1)
    return np.degrees(np.arccos(np.clip(dot / np.maximum(norms, 1e-12), -1, 1))).mean()


def redundancy(pixels, channels):
    """Mean |correlation| between the chosen channels. 1 = copies of each other. Lower = better."""
    k = len(channels)
    c = np.abs(np.corrcoef(pixels[:, channels], rowvar=False))
    return (c.sum() - k) / (k * (k - 1))


def uniform(n_channels, k):
    """Baseline: k evenly spaced channels, no thinking at all."""
    return np.linspace(0, n_channels - 1, k).round().astype(int).tolist()


def evaluate(pixels01, raw_pixels, channels, calib, test):
    """All scores for one set of chosen channels."""
    r = rebuild(pixels01, channels, calib)
    return {"PSNR": psnr(pixels01[test], r[test]),
            "SAM": sam_degrees(pixels01[test], r[test]),
            "Redundancy": redundancy(raw_pixels, channels)}
