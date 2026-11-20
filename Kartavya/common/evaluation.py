"""
Bookkeeping for "how much smaller is the data?" questions, plus the registry of every metric's
optimisation direction (used by the unified comparison).

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


# ------------------------------------------------------------------ metric registry
# direction: "max" = higher is better, "min" = lower is better. Only these metrics are compared across
# algorithms; every algorithm-specific metric stays in that algorithm's own files.
METRICS = {
    # reconstruction quality (test region, data in [0, 1])
    "PSNR_dB":                      {"direction": "max", "label": "PSNR (dB)", "family": "reconstruction"},
    "SSIM":                         {"direction": "max", "label": "SSIM", "family": "reconstruction"},
    "MSE":                          {"direction": "min", "label": "MSE", "family": "reconstruction"},
    "RMSE":                         {"direction": "min", "label": "RMSE", "family": "reconstruction"},
    "MAE":                          {"direction": "min", "label": "MAE", "family": "reconstruction"},
    # size
    "dimensionality_reduction_pct": {"direction": "max", "label": "Dimensionality reduction (%)", "family": "size"},
    "storage_saving_pct":           {"direction": "max", "label": "Storage saving incl. model (%)", "family": "size"},
    "compression_ratio":            {"direction": "max", "label": "Compression ratio", "family": "size"},
    # cost of applying the reduction (test region)
    "forward_seconds":              {"direction": "min", "label": "Forward (reduce) time (s)", "family": "cost"},
    "inverse_seconds":              {"direction": "min", "label": "Inverse (reconstruct) time (s)", "family": "cost"},
    "forward_peak_mb":              {"direction": "min", "label": "Forward peak memory (MB)", "family": "cost"},
    # supervised (LDA only; NaN = not applicable for unsupervised methods)
    "accuracy":                     {"direction": "max", "label": "Accuracy", "family": "classification"},
    "precision_macro":              {"direction": "max", "label": "Precision (macro)", "family": "classification"},
    "recall_macro":                 {"direction": "max", "label": "Recall (macro)", "family": "classification"},
    "f1_macro":                     {"direction": "max", "label": "F1 (macro)", "family": "classification"},
    "fisher_lda":                   {"direction": "max", "label": "Fisher separability", "family": "classification"},
}

RUN_METRICS = {   # one value per algorithm run
    "train_seconds": {"direction": "min", "label": "Training time (s)"},
    "test_seconds":  {"direction": "min", "label": "Evaluation time (s)"},
    "total_seconds": {"direction": "min", "label": "Total time (s)"},
    "train_peak_python_mb": {"direction": "min", "label": "Training peak memory (MB, tracemalloc)"},
}


def best_index(values, direction):
    """Index of the best finite value (ties: first), or None if nothing is finite."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) | (np.isinf(v) & (v > 0) if direction == "max" else np.isfinite(v))
    if not ok.any():
        return None
    w = np.where(ok, v, -np.inf if direction == "max" else np.inf)
    return int(np.argmax(w) if direction == "max" else np.argmin(w))
