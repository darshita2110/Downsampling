"""
Loads the two datasets as plain numpy arrays.
No other project code needed: only numpy and scipy.
Run from the Downsampling folder.
"""
import numpy as np
from scipy.io import loadmat

# ---------- Raichur, Sentinel-2 (real satellite, multispectral, 12 bands) ----------
RAICHUR_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
                 "B07", "B08", "B8A", "B09", "B11", "B12"]

def load_raichur(date="2026-10-12"):
    """Returns array of shape (512, 512, 12), reflectance values roughly 0 to 1."""
    return np.load(f"data/darshita/raichur_{date}_crop.npy").astype(np.float32)

# ---------- Indian Pines (real airborne, hyperspectral, 200 bands) ----------
def load_indian_pines():
    """Returns array of shape (145, 145, 200)."""
    data = loadmat("data/darshita/Indian_pines_corrected.mat")
    return data["indian_pines_corrected"].astype(np.float32)


if __name__ == "__main__":
    r = load_raichur()
    print("Raichur     :", r.shape, "bands:", RAICHUR_BANDS)
    ip = load_indian_pines()
    print("Indian Pines:", ip.shape)