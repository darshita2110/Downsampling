"""
Small helpers shared by every algorithm in Kartavya/.

Nothing here is specific to PCA: logging, folder creation, timing, memory
measurement and JSON saving are needed by ICA, LDA, GA, PSO ... as well.
"""
import json
import logging
import os
import random
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

# Kartavya/common/utils.py -> parents[1] is Kartavya/, parents[2] is the repository root
KARTAVYA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KARTAVYA_ROOT.parent


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

    peak_python_mb: the largest amount of memory allocated *inside* the block,
                    tracked by tracemalloc (NumPy arrays are included).
    rss_after_mb  : total process memory after the block (includes everything loaded before).
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats[key] = {"seconds": seconds, "peak_python_mb": peak / 1e6, "rss_after_mb": rss_mb()}


def _to_jsonable(obj):
    """Turn NumPy numbers/arrays and Paths into plain Python objects so json can write them."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
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
