"""
Small helpers shared by every algorithm: logging, folder creation, timing + memory measurement, JSON.
(Unchanged in behaviour from the original repository; dataset-independent helpers were added at the end.)
"""
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:                                    # psutil gives the process memory (RSS); optional
    import psutil
except ImportError:                     # pragma: no cover - the code still works without it
    psutil = None

KARTAVYA_ROOT = Path(__file__).resolve().parents[1]


def ensure_dirs(*dirs):
    """Create every folder given (and its parents) if it does not exist yet."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """Fix the random generators so every run picks the same pixels."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def get_logger(name, log_file=None):
    """A logger that prints to the terminal and, if log_file is given, also writes to that file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        h.close()
    logger.handlers.clear()                     # avoid duplicate lines when called twice
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        ensure_dirs(Path(log_file).parent)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def close_logger(logger):
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def rss_mb():
    """Memory currently held by this Python process, in MB (resident set size)."""
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


@contextmanager
def measure(stats, key):
    """
    Measure wall time and memory of the code inside a `with` block.

        stats = {}
        with measure(stats, "fit"):
            model.fit(X)
        stats["fit"] -> {"seconds": ..., "peak_python_mb": ..., "rss_after_mb": ...}

    peak_python_mb: the largest amount of memory allocated *inside* the block, tracked by tracemalloc
                    (NumPy arrays are included; PyTorch's C++ allocations are not).
    rss_after_mb  : total process memory after the block.
    Nested use is supported (an inner block does not stop the outer measurement).
    """
    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        if not already:
            tracemalloc.stop()
        stats[key] = {"seconds": seconds, "peak_python_mb": peak / 1e6, "rss_after_mb": rss_mb()}


def timed(fn, repeats):
    """Run fn `repeats` times under measure(); return the last result and the median statistics.
    (Millisecond timings are noisy, so each step is repeated and the median kept.)"""
    runs, out = [], None
    for _ in range(max(1, int(repeats))):
        st = {}
        with measure(st, "x"):
            out = fn()
        runs.append(st["x"])
    return out, {key: float(np.median([r[key] for r in runs])) for key in runs[0]}


def _to_jsonable(obj):
    """Turn NumPy numbers/arrays and Paths into plain Python objects so json can write them."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path):
    ensure_dirs(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_size_bytes(path):
    return Path(path).stat().st_size


def human_bytes(n):
    """1536 -> '1.50 KB' (powers of 1024)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024


def safe_name(text, max_len=60):
    """A string usable as a folder name on every OS."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return (s or "dataset")[:max_len]


def fingerprint(paths):
    """Short hash of the dataset files (absolute path, size, modification time): a different dataset - or
    the same folder with different files - gets a different output directory, so results never mix."""
    h = hashlib.sha1()
    for p in sorted(Path(x).resolve() for x in paths):
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:8]


def comb_safe(n, k):
    return math.comb(int(n), int(k))
