"""
Loader for the Indian Pines AVIRIS hyperspectral benchmark (145 x 145 pixels).

Why this exists: the Raichur data has no ground-truth labels, so we can only
measure how well a method preserves neighbourhoods. Indian Pines ships with
16 land-cover classes, which lets us also ask the more direct question: do
the 3 reduced features still separate the classes?

Files expected (standard distribution of the dataset):
    Indian_pines_corrected.mat   key 'indian_pines_corrected'  (145,145,200)
    Indian_pines_gt.mat          key 'indian_pines_gt'         (145,145)
The 'corrected' cube already has the 20 water-absorption bands removed
(220 -> 200 bands). Label 0 means "unlabelled" and is excluded by default.

Returns the same kind of structure as src/data_loader.load_raichur_tile
(flat valid pixels + row/col + grid shape), so technique code is unchanged.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

CUBE_KEY = "indian_pines_corrected"
GT_KEY = "indian_pines_gt"

CLASS_NAMES = {
    1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill", 4: "Corn",
    5: "Grass-pasture", 6: "Grass-trees", 7: "Grass-pasture-mowed",
    8: "Hay-windrowed", 9: "Oats", 10: "Soybean-notill",
    11: "Soybean-mintill", 12: "Soybean-clean", 13: "Wheat",
    14: "Woods", 15: "Buildings-Grass-Trees-Drives", 16: "Stone-Steel-Towers",
}


def _load_key(path, key):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = loadmat(path)
    if key not in data:
        keys = [k for k in data if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not in {path.name}; found {keys}")
    return data[key]


def load_indian_pines(cube_path, gt_path, labelled_only=True):
    """Load Indian Pines as flat pixels plus labels and grid positions.

    labelled_only=True keeps only pixels with class 1..16 (10,249 pixels);
    False keeps all 21,025 (label 0 = unlabelled background/other).
    """
    cube = _load_key(cube_path, CUBE_KEY).astype("float64")
    gt = _load_key(gt_path, GT_KEY).astype(int)
    if cube.ndim != 3 or cube.shape[:2] != gt.shape:
        raise ValueError(f"Cube {cube.shape} and ground truth {gt.shape} do not align.")
    if not np.isfinite(cube).all():
        raise ValueError("Cube contains NaN/inf.")

    height, width, n_bands = cube.shape
    rows, cols = np.divmod(np.arange(height * width), width)  # row-major, as cube_to_pixels
    pixels = cube.reshape(height * width, n_bands)
    labels = gt.reshape(-1)

    if labelled_only:
        keep = labels > 0
        pixels, labels, rows, cols = pixels[keep], labels[keep], rows[keep], cols[keep]

    return {
        "pixels": pixels,
        "labels": labels,
        "row": rows,
        "col": cols,
        "grid_shape": (height, width),
        "n_bands": n_bands,
        "class_names": CLASS_NAMES,
        "labelled_only": labelled_only,
    }
