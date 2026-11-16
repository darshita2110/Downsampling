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


# ------------------------------------------------------------------ all datasets in one place
DATA_DIR = "data/darshita"

# Standard benchmark datasets (files from the EHU "Hyperspectral Remote Sensing Scenes" page).
# The variable name inside each .mat file is found automatically (largest 3-D array = image,
# largest 2-D array = labels), so only the file names matter.
BENCHMARKS = {
    "indian_pines": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),   # AVIRIS, 200 bands, 16 classes
    "pavia_u": ("PaviaU.mat", "PaviaU_gt.mat"),                             # ROSIS, 103 bands, 9 classes
    "salinas": ("Salinas_corrected.mat", "Salinas_gt.mat"),                 # AVIRIS, 204 bands, 16 classes
    "ksc": ("KSC.mat", "KSC_gt.mat"),                                       # AVIRIS, 176 bands, 13 classes
    "botswana": ("Botswana.mat", "Botswana_gt.mat"),                        # Hyperion satellite, 145 bands, 14 classes
}


def _largest_array(path, ndim):
    """Open a .mat / .npy file and return its largest array with `ndim` dimensions."""
    if path.endswith(".npy"):
        a = np.load(path)
        return a if a.ndim == ndim else None
    try:
        content = loadmat(path)
        arrays = [v for k, v in content.items() if not k.startswith("__") and getattr(v, "ndim", 0) == ndim]
    except NotImplementedError:                      # MATLAB v7.3 files are HDF5 inside
        import h5py
        with h5py.File(path, "r") as f:
            arrays = [np.array(f[k]).T for k in f.keys() if getattr(f[k], "ndim", 0) == ndim]
    return max(arrays, key=lambda a: a.size) if arrays else None


def _bands_last(cube):
    """Make sure the shape is (height, width, bands). Benchmark .mat files already are; a file saved
    as (bands, height, width), like a GeoTIFF, has a first axis smaller than both others: move it."""
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
        return np.moveaxis(cube, 0, -1)
    return cube


def _clean(cube, name):
    """Replace NaN/inf with 0 and clip a band only if it has absurd outliers (broken pixels)."""
    cube = np.nan_to_num(cube.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    flat = cube.reshape(-1, cube.shape[2])
    hi = np.percentile(flat, 99.9, axis=0)
    lo = np.percentile(flat, 0.1, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    broken = (flat.max(axis=0) > hi + 10 * span) | (flat.min(axis=0) < lo - 10 * span)
    if broken.any():
        print(f"{name}: clipping extreme outliers in {int(broken.sum())} band(s)")
        flat[:, broken] = np.clip(flat[:, broken], lo[broken], hi[broken])
    return flat.reshape(cube.shape)


def _drop_constant(cube, name):
    """A band with the same value everywhere carries no information (and breaks scaling): drop it.
    Channel names keep the ORIGINAL band numbers, so results still point to the right band."""
    keep = np.where(cube.reshape(-1, cube.shape[2]).std(axis=0) > 0)[0]
    if len(keep) < cube.shape[2]:
        print(f"{name}: dropping {cube.shape[2] - len(keep)} constant band(s)")
    return cube[:, :, keep], [f"ch {i}" for i in keep]


def available_datasets():
    """Names of every dataset whose files are present in data/darshita/."""
    names = [n for n, (img, _) in BENCHMARKS.items() if os.path.exists(os.path.join(DATA_DIR, img))]
    if os.path.exists(os.path.join(DATA_DIR, "raichur_best.txt")):
        names.append("raichur")
    return names


def dataset_name(name):
    """Short name used in output file names (a file path becomes its file name without extension)."""
    return os.path.splitext(os.path.basename(name))[0] if os.path.isfile(name) else name


def load_labels(name):
    """Ground-truth map (height, width), 0 = no label. None if the dataset has no labels."""
    if name in BENCHMARKS:
        path = os.path.join(DATA_DIR, BENCHMARKS[name][1])
    elif os.path.isfile(name):                        # any file: look for <name>_gt.mat / .npy beside it
        stem = os.path.splitext(name)[0].replace("_corrected", "")
        path = next((p for p in (stem + "_gt.mat", stem + "_gt.npy") if os.path.exists(p)), "")
    else:
        return None
    if not os.path.exists(path):
        return None
    labels = _largest_array(path, 2)
    return None if labels is None else labels.astype(np.int64)


def load_dataset(name):
    """
    Any of these works:
      "indian_pines", "pavia_u", "salinas", "ksc", "botswana"  -> benchmark .mat files in data/darshita/
      "raichur"            -> our Sentinel-2 data, cleanest date (run data steps 1 and 2 first)
      "raichur_2026-09-27" -> the clean crop of one date
      "path/to/any_file.mat" or ".npy" -> any image cube (height x width x bands)
    Returns (cube, channel_names), cube shape (height, width, bands), float32.
    """
    if name == "indian_pines" and not os.path.exists(IMAGE_FILE):
        load_image()                                   # downloads it the first time
    if name in BENCHMARKS:
        path = os.path.join(DATA_DIR, BENCHMARKS[name][0])
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Download '{BENCHMARKS[name][0]}' (see darshita/README.md, "
                             f"or run python darshita/data_3_download_benchmarks.py)")
        cube, names = _drop_constant(_clean(_bands_last(_largest_array(path, 3)), name), name)
        print(f"{name}: {cube.shape[0]} x {cube.shape[1]} pixels, {cube.shape[2]} bands")
        return cube, names
    if os.path.isfile(name):
        cube = _largest_array(name, 3)
        if cube is None:
            raise SystemExit(f"No 3-D image array found in {name}")
        cube, names = _drop_constant(_clean(_bands_last(cube), dataset_name(name)), dataset_name(name))
        print(f"{dataset_name(name)}: {cube.shape[0]} x {cube.shape[1]} pixels, {cube.shape[2]} bands")
        return cube, names
    if name == "raichur" or name.startswith("raichur_"):
        if name == "raichur":                      # the cleanest date, chosen by data_2_check_s2.py
            best = os.path.join("data/darshita", "raichur_best.txt")
            if not os.path.exists(best):
                raise SystemExit("No Raichur data yet. Run data_1_download_s2.py, then data_2_check_s2.py")
            crop = open(best).read().strip()
        else:                                      # one date, e.g. "raichur_2026-09-27"
            crop = f"{name}_crop.npy"
            if not os.path.exists(os.path.join("data/darshita", crop)):
                dates = sorted(f[len("raichur_"):-len("_crop.npy")] for f in os.listdir("data/darshita")
                               if f.startswith("raichur_") and f.endswith("_crop.npy"))
                raise SystemExit(f"No data/darshita/{crop}. Dates available: {', '.join(dates)}")
        cube = np.load(os.path.join("data/darshita", crop)).astype(np.float32)
        info = json.load(open("data/darshita/raichur_bands.json"))
        print(f"Raichur data: {crop}")
        return cube, info["bands"]
    raise SystemExit(f"Unknown dataset '{name}'. Use one of: {', '.join(list(BENCHMARKS) + ['raichur'])}, "
                     f"or a path to a .mat / .npy file")


def default_k(n_channels):
    """How many channels to send by default: 10 for hyperspectral (5% of 200), and for few bands
    the K value closest to a third of them (4 of 12 for Sentinel-2)."""
    if n_channels > 40:
        return 10
    return min(k_values(n_channels), key=lambda k: abs(k - n_channels / 3))


def k_values(n_channels):
    """K values for the 'quality vs number of channels' graph (same as most papers: 5..30)."""
    if n_channels > 40:
        return [5, 10, 15, 20, 25, 30]
    return [k for k in [2, 3, 4, 5, 6, 8] if k < n_channels]


def to_table(cube):
    """(height, width, channels) -> table with one row per pixel, one column per channel."""
    return cube.reshape(-1, cube.shape[2])


def scale_01(pixels):
    """Scale every channel to 0..1 so every channel counts equally in the error."""
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    return (pixels - low) / np.maximum(high - low, 1e-12)       # a constant band stays 0, no division by 0


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
