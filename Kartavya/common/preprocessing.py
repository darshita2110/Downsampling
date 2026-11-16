"""
Preprocessing shared by all Kartavya algorithms.

The pipeline is:
    image (H, W, C) in natural units
      -> normalise every channel to [0, 1]             (normalize_channels)
      -> keep only valid pixels, flatten to (N, C)     (image_to_matrix)
      -> optionally standardise with TRAINING mean/std (Standardizer)
    ... algorithm ...
      -> undo standardisation, put pixels back in place (matrix_to_image)
"""
import numpy as np


def normalize_channels(x, names, ranges):
    """
    Map each channel to [0, 1] using its KNOWN physical range, not the range of the current image.
    Using fixed ranges keeps every date on the same scale, so a model fitted on one date can be applied
    to another date.

    ranges: list of (prefix, low, high). The first prefix that a channel name starts with is used,
            e.g. ("RGB:", 0, 255) for display levels, ("NDVI", -1, 1) for the index.
    """
    out = np.empty_like(x, dtype=np.float32)
    used = []
    for c, name in enumerate(names):
        for prefix, low, high in ranges:
            if name.startswith(prefix):
                out[..., c] = (x[..., c] - low) / (high - low)
                used.append((name, low, high))
                break
        else:
            raise KeyError(f"No normalisation range configured for channel '{name}'")
    np.clip(out, 0.0, 1.0, out=out)
    return out, used


def image_to_matrix(img, valid):
    """
    (H, W, C) image -> (N, C) feature matrix: one ROW per valid pixel, one COLUMN per channel.
    This is the "pixels are samples, channels are features" view PCA needs.
    """
    return img[valid].astype(np.float64)


def matrix_to_image(mat, valid, fill=0.0):
    """(N, K) matrix of valid pixels -> (H, W, K) image; invalid pixels get `fill`."""
    h, w = valid.shape
    out = np.full((h, w, mat.shape[1]), fill, dtype=np.float32)
    out[valid] = mat
    return out


def sample_rows(mat, max_rows, rng):
    """Random subset of rows (without replacement) to keep fitting fast; all rows if max_rows is None."""
    if max_rows is None or mat.shape[0] <= max_rows:
        return mat
    return mat[rng.choice(mat.shape[0], size=max_rows, replace=False)]


class Standardizer:
    """
    z = (x - mean) / std, with mean and std learnt from TRAINING data only (no test leakage).
    With enabled=False it is the identity transform (PCA still centres the data itself).
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        if self.enabled:
            self.mean_ = x.mean(axis=0)
            self.std_ = x.std(axis=0)
            self.std_[self.std_ < 1e-12] = 1.0         # a constant channel is left unscaled
        else:
            self.mean_ = np.zeros(x.shape[1])
            self.std_ = np.ones(x.shape[1])
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, z):
        return z * self.std_ + self.mean_

    def state(self):
        return {"enabled": self.enabled, "mean": self.mean_, "std": self.std_}

    @classmethod
    def from_state(cls, d):
        s = cls(bool(d["enabled"]))
        s.mean_, s.std_ = np.asarray(d["mean"]), np.asarray(d["std"])
        return s


def spatial_split(valid, train_fraction=0.5):
    """
    Split ONE image into a training part (top rows) and a test part (bottom rows), used only when
    there is a single acquisition date. Returns two boolean masks.
    """
    h = valid.shape[0]
    cut = int(round(h * train_fraction))
    train, test = valid.copy(), valid.copy()
    train[cut:] = False
    test[:cut] = False
    return train, test, cut


def extract_patches(img, valid, size, stride, row_range=None):
    """
    Cut an (H, W, C) image into square patches for a CNN.
    Only patches whose pixels are ALL valid are kept (no-data pixels would teach the network nonsense).
    row_range: (start, stop) rows a patch must lie within entirely (e.g. the training half only).
    Returns (patches as float32 (N, C, size, size), top-left corners (N, 2)).
    """
    h, w, _ = img.shape
    r0, r1 = row_range if row_range else (0, h)
    out, corners = [], []
    for r in range(r0, r1 - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            if valid[r:r + size, c:c + size].all():
                out.append(img[r:r + size, c:c + size].transpose(2, 0, 1))
                corners.append((r, c))
    if not out:
        return np.zeros((0, img.shape[2], size, size), np.float32), np.zeros((0, 2), int)
    return np.stack(out).astype(np.float32), np.array(corners)
