"""
Preprocessing shared by all Kartavya algorithms.

The pipeline is:
    image (H, W, C) in native units and dtype
      -> normalise every channel to [0, 1] with TRAINING statistics   (Normalizer)
      -> keep only valid pixels, flatten to (N, C)                     (image_to_matrix)
      -> optionally standardise with TRAINING mean/std                 (Standardizer)
    ... algorithm ...
      -> undo standardisation, put pixels back in place                (matrix_to_image)
"""
import numpy as np


class Normalizer:
    """
    Map each channel to [0, 1] with FIXED per-channel ranges learnt once (from training pixels or from the
    data type), so every sample is on the same scale and a model fitted on one sample applies to another.
    This replaces the hard-coded per-dataset ranges of the original code (e.g. ("RGB:", 0, 255),
    ("NDVI", -1, 1)) with ranges detected from the data.

    method "auto"  : uint8 / int8 -> range of the type (0..255, identical to the original PNG experiment);
                     everything else -> training min/max per channel
           "dtype" : any integer type -> range of the type; float -> training min/max
           "minmax": training min/max per channel for every type
    """

    def __init__(self, method="auto"):
        self.method = method
        self.low = self.high = None
        self.method_used = None

    def fit(self, x, dtype):
        dtype = np.dtype(dtype)
        x = np.asarray(x)
        use_dtype = (dtype.kind in "iu" and (self.method == "dtype" or (self.method == "auto" and dtype.itemsize == 1)))
        if use_dtype:
            info = np.iinfo(dtype)
            self.low = np.full(x.shape[1], float(info.min))
            self.high = np.full(x.shape[1], float(info.max))
            self.method_used = f"dtype range {info.min}..{info.max}"
        else:
            self.low = np.nanmin(x, axis=0).astype(np.float64)
            self.high = np.nanmax(x, axis=0).astype(np.float64)
            self.method_used = "training min/max per channel"
        span = self.high - self.low
        self.constant_channels = np.flatnonzero(span <= 0).tolist()
        self.high = np.where(span <= 0, self.low + 1.0, self.high)     # a constant channel maps to 0
        return self

    def transform(self, data):
        """Native (H, W, C) or (N, C) -> float in [0, 1]. float64 input stays float64 (no precision loss);
        everything else becomes float32 (exact for 8/16-bit integers)."""
        out_dtype = np.float64 if np.asarray(data).dtype == np.float64 else np.float32
        out = (np.asarray(data, dtype=np.float64) - self.low) / (self.high - self.low)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(out_dtype)

    def inverse_transform(self, z):
        return np.asarray(z, dtype=np.float64) * (self.high - self.low) + self.low

    def state(self):
        return {"method": self.method, "method_used": self.method_used, "low": self.low, "high": self.high,
                "constant_channels": self.constant_channels}


def image_to_matrix(img, valid):
    """
    (H, W, C) image -> (N, C) feature matrix: one ROW per valid pixel, one COLUMN per channel.
    This is the "pixels are samples, channels are features" view the algorithms need.
    """
    return img[valid].astype(np.float64)


def matrix_to_image(mat, valid, fill=0.0, dtype=np.float32):
    """(N, K) matrix of valid pixels -> (H, W, K) image; invalid pixels get `fill`."""
    h, w = valid.shape
    out = np.full((h, w, mat.shape[1]), fill, dtype=dtype)
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
