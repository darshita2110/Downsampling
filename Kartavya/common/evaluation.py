"""
Bookkeeping for "how much smaller is the data?" questions, shared by all algorithms.

Storage is reported two ways:
  * theoretical: number of stored values x bytes per value (independent of file format)
  * on disk    : actual size of the .npy files written by the experiment
For a linear model the reduced data alone is not enough to rebuild the image - the receiver also
needs the model (for PCA: mean, scale and the C x K projection matrix), so that is counted too.
"""
import numpy as np


def dimensionality_reduction_pct(n_original, n_reduced):
    """100 * (1 - K / C): 8 channels -> 2 components is a 75 % reduction."""
    return 100.0 * (1.0 - n_reduced / n_original)


def storage_report(n_pixels, n_channels, n_components, model_values, bytes_per_value=4,
                   original_bytes_per_value=None):
    """
    n_pixels      : valid pixels stored
    model_values  : number of floats the receiver needs to invert the reduction
    bytes_per_value: 4 for float32 reduced data
    original_bytes_per_value: storage type of the original (defaults to bytes_per_value)
    """
    obpv = original_bytes_per_value or bytes_per_value
    original = n_pixels * n_channels * obpv
    reduced = n_pixels * n_components * bytes_per_value
    model = model_values * bytes_per_value
    return {"original_bytes": int(original),
            "reduced_bytes": int(reduced),
            "model_bytes": int(model),
            "reduced_plus_model_bytes": int(reduced + model),
            "compression_ratio": original / (reduced + model),
            "storage_saving_pct": 100.0 * (1 - (reduced + model) / original)}


def flatten_record(prefix_dict, sep="_"):
    """{"a": {"b": 1}} -> {"a_b": 1} for writing nested metrics into one CSV row."""
    flat = {}

    def walk(d, prefix):
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict):
                walk(v, key)
            elif isinstance(v, (list, tuple, np.ndarray)):
                flat[key] = str(list(v))
            else:
                flat[key] = v
    walk(prefix_dict, "")
    return flat
