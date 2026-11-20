"""
Generic loader for public hyperspectral benchmark scenes stored as MATLAB .mat
files (Indian Pines, Salinas, Kennedy Space Center, Botswana, Pavia University).

Each scene is described by a small registry entry (file name + variable name for
the cube and for the ground truth). Handles two real-world quirks found in the
supplied files:
  * KSC_data.mat stores the cube as (bands, rows, cols) = (176, 512, 614), not
    (rows, cols, bands). Orientation is fixed by matching the ground-truth shape.
  * Botswana.mat has NO ground-truth file supplied, and PaviaU has only the
    ground truth (the image cube file is missing). Those cases are reported
    clearly rather than guessed at.
Label 0 always means "unlabelled" and is excluded when labelled_only=True.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# name -> {"cube": (file, key), "gt": (file, key) or None}
REGISTRY = {
    "indian_pines": {"cube": ("Indian_pines_corrected.mat", "indian_pines_corrected"),
                     "gt": ("Indian_pines_gt.mat", "indian_pines_gt")},
    "salinas": {"cube": ("Salinas_corrected.mat", "salinas_corrected"),
                "gt": ("Salinas_gt.mat", "salinas_gt")},
    "ksc": {"cube": ("KSC_data.mat", "data"), "gt": ("KSC_gt.mat", "gt")},
    "botswana": {"cube": ("Botswana.mat", "Botswana"), "gt": None},
    "pavia_university": {"cube": ("PaviaU.mat", "paviaU"),
                         "gt": ("PaviaU_gt.mat", "paviaU_gt")},
}


def _read(path, key):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    d = loadmat(path)
    if key not in d:
        raise KeyError(f"Key '{key}' not in {path.name}; found {[k for k in d if not k.startswith('__')]}")
    return d[key]


def available_datasets(data_dir):
    """Names from REGISTRY whose cube file exists in data_dir."""
    data_dir = Path(data_dir)
    return [n for n, e in REGISTRY.items() if (data_dir / e["cube"][0]).exists()]


def load_hsi(name, data_dir, labelled_only=True):
    """Return flat pixels (+ labels or None) and grid positions for one scene."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(REGISTRY)}")
    entry = REGISTRY[name]
    data_dir = Path(data_dir)
    cube = _read(data_dir / entry["cube"][0], entry["cube"][1])
    gt = None
    if entry["gt"] is not None:
        gt_path = data_dir / entry["gt"][0]
        gt = _read(gt_path, entry["gt"][1]).astype(int)

    if gt is not None:
        if cube.shape[:2] == gt.shape:
            pass
        elif cube.shape[1:] == gt.shape:
            cube = np.transpose(cube, (1, 2, 0))  # (bands, rows, cols) -> (rows, cols, bands)
        else:
            raise ValueError(f"{name}: cube {cube.shape} does not match ground truth {gt.shape}.")
    else:
        band_axis = int(np.argmin(cube.shape))  # no gt: assume the smallest axis is spectral
        cube = np.moveaxis(cube, band_axis, -1)

    cube = cube.astype("float64")
    if not np.isfinite(cube).all():
        raise ValueError(f"{name}: cube contains NaN/inf.")
    height, width, n_bands = cube.shape
    rows, cols = np.divmod(np.arange(height * width), width)
    pixels = cube.reshape(height * width, n_bands)
    labels = None if gt is None else gt.reshape(-1)

    keep = np.ones(height * width, dtype=bool)
    if labels is not None and labelled_only:
        keep = labels > 0
    return {
        "name": name,
        "pixels": pixels[keep],
        "labels": None if labels is None else labels[keep],
        "row": rows[keep],
        "col": cols[keep],
        "grid_shape": (height, width),
        "n_bands": n_bands,
        "n_pixels_total": height * width,
        "has_labels": labels is not None,
    }
