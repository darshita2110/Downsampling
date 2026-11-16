"""
The common interface every algorithm plugs into, and the code that runs one algorithm end to end.

An algorithm package (algorithms/<name>/pipeline.py) defines

    NAME, DISPLAY, FAMILY, REQUIRES          identification + optional third-party modules it needs
    output_name()                            folder name of its results (e.g. "pca" or "pca_nostd")
    train(ctx)                               fits / searches on TRAINING data only, saves its model files;
                                             may return {"status": "not_applicable", "reason": ...}
    test(ctx) -> pandas.DataFrame            loads the SAVED model files, evaluates the TEST region and
                                             returns one row per (test sample, K)

The same function `run_algorithm` is used by every algorithm's main.py and by run_all_algorithms.py, so the
two ways of running give identical results.

Output layout (a different dataset - or the same folder with different files - gets a different
<dataset_id> = <name>_<content hash>, so results can never be mixed):

    outputs/<dataset_id>/
        dataset/                 filename_index_mapping.json, dataset_inspection.json, split.json, normalization.json
        <algorithm>/
            models/ reduced_data/ reconstructed_data/ metrics/ graphs/ logs/
            metrics.csv  metrics.json  metrics_report.txt      (per-K summary over the test samples)
        comparison/              (run_all_algorithms.py only)
"""
import importlib
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common import config as gcfg
from common.data_loader import Dataset, DatasetError, make_split
from common.evaluation import METRICS
from common.utils import close_logger, ensure_dirs, get_logger, measure, rss_mb, save_json

CATEGORIES = ["models", "reduced_data", "reconstructed_data", "metrics", "graphs", "logs"]
BOOL_FLAGS = {"--self-check", "--no-standardize", "--skip-ubscnn", "--quick"}


# ============================================================================ K values
def default_k_values(n_channels, include_full=False):
    """K values evaluated by every algorithm (the same list for all, see config.py)."""
    c = int(n_channels)
    if c <= gcfg.MAX_C_FOR_ALL_K:
        ks = list(range(1, c))
    else:
        ks = [k for k in gcfg.K_GRID if k < c][:gcfg.MAX_K_VALUES]
    if include_full and c not in ks:
        ks.append(c)
    return ks


# ============================================================================ context
@dataclass
class RunContext:
    name: str                       # output folder name (e.g. "pca")
    display: str
    dataset: Dataset
    split: dict
    ks: list                        # common K list (1 .. C-1 or the grid)
    dataset_root: Path
    root: Path
    dirs: dict
    log: object = None
    options: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)          # free-form facts the algorithm wants in metrics.json

    @property
    def names(self):
        return self.dataset.channel_names

    @property
    def C(self):
        return self.dataset.n_channels

    @property
    def fraction(self):
        return self.split.get("spatial_train_fraction", gcfg.SPATIAL_TRAIN_FRACTION)

    def sibling(self, algorithm):
        """Result folder of another algorithm on the same dataset (for the original cross-method comparisons)."""
        return self.dataset_root / algorithm


# ============================================================================ dataset preparation
def resolve_dataset_path(configured=None, prompt="Dataset path (file or folder): "):
    """Command line (--dataset PATH or first argument) > configured value > interactive prompt."""
    args = sys.argv[1:]
    if "--dataset" in args:
        i = args.index("--dataset")
        if i + 1 < len(args):
            return args[i + 1]
    flagged = {i + 1 for i, a in enumerate(args[:-1]) if a.startswith("--") and a not in BOOL_FLAGS}
    positional = [a for i, a in enumerate(args) if not a.startswith("-") and i not in flagged]
    if positional:
        return positional[0]
    if configured:
        return configured
    if not sys.stdin or not sys.stdin.isatty():
        raise DatasetError("No dataset path given. Set DATASET_PATH in the file or pass --dataset PATH.")
    value = input(prompt).strip().strip('"').strip("'")
    if not value:
        raise DatasetError("No dataset path given.")
    return value


def cli_option(flag, default=None, cast=str):
    args = sys.argv[1:]
    if flag in args:
        i = args.index(flag)
        if cast is bool:
            return True
        if i + 1 < len(args):
            return cast(args[i + 1])
    return default


def prepare_dataset(dataset_path, label_path=None, output_root=None, logger=None):
    """Load + validate the dataset, decide the split, fit the normalisation (training pixels only) and write
    the dataset-level reports. Returns (dataset, split, dataset_root)."""
    output_root = Path(output_root) if output_root else gcfg.OUTPUT_ROOT
    log = logger or get_logger("kartavya.dataset")
    log.info(f"Dataset path: {Path(dataset_path).expanduser().resolve()}")
    ds = Dataset(dataset_path, label_path=label_path, logger=log)
    split = make_split(ds.n_samples)
    ds.fit_normalizer(split, log)
    root = output_root / ds.dataset_id
    droot = root / "dataset"
    ensure_dirs(droot)
    save_json({"dataset_dir": str(ds.path), "layout": ds.layout,
               "sort_rule": "ascending lexicographic order of the full (relative) filename",
               "samples": [{"index": s["index"], "name": s["name"], "files": [str(f) for f in s["files"]]}
                           for s in ds.samples],
               "labels": {int(k): str(v) for k, v in ds.label_map.items()}}, droot / "filename_index_mapping.json")
    save_json(ds.inspection, droot / "dataset_inspection.json")
    save_json(split, droot / "split.json")
    save_json(ds.normalizer.state() | {"channel_names": ds.channel_names}, droot / "normalization.json")
    log.info(f"Layout '{ds.layout}': {ds.n_samples} sample(s), {ds.n_channels} channels, dtype {ds.dtype}; "
             f"output folder {root}")
    for s in ds.inspection["samples"][:20]:
        log.info(f"  [{s['index']}] {s['name']}: {tuple(s['shape_hwc'])} {s['dtype']}, valid {s['valid_pct']:.2f}%")
    if ds.n_samples > 20:
        log.info(f"  ... {ds.n_samples - 20} more")
    log.info(f"Channels: {ds.channel_names[:16]}{' ...' if ds.n_channels > 16 else ''}")
    log.info(f"Split '{split['mode']}': train {split['train']} | test {split['test']}")
    if split["mode"] == "spatial":
        log.warning("Only ONE sample: the top part of the image trains and the bottom part tests.")
    if ds.label_map:
        log.info(f"Label rasters (used by LDA only): {[Path(v).name for v in ds.label_map.values()]}")
    return ds, split, root


# ============================================================================ one algorithm
def check_requirements(module):
    missing = []
    for req in getattr(module, "REQUIRES", []):
        try:
            importlib.import_module(req)
        except ImportError:
            missing.append(req)
    if missing:
        raise ImportError(f"{module.DISPLAY} needs the Python package(s) {missing}: pip install {' '.join(missing)}")


def run_algorithm(module, dataset, split, dataset_root, options=None, ks=None):
    """Train + test one algorithm and write its standard summary files. Raises on failure."""
    options = options or {}
    check_requirements(module)
    name = module.output_name()
    root = Path(dataset_root) / name
    if root.exists():
        shutil.rmtree(root)                 # never mix files of an earlier run into this one
    dirs = {c: root / c for c in CATEGORIES}
    ensure_dirs(*dirs.values())
    ctx = RunContext(name=name, display=module.DISPLAY, dataset=dataset, split=split,
                     ks=list(ks or default_k_values(dataset.n_channels)), dataset_root=Path(dataset_root),
                     root=root, dirs=dirs, options=options)
    stats, result = {}, {"algorithm": name, "display": module.DISPLAY, "family": module.FAMILY,
                         "output_dir": str(root), "status": "ok"}
    t0 = time.perf_counter()
    ctx.log = get_logger(f"{name}.train", dirs["logs"] / "train.log")
    ctx.log.info(f"===== {module.DISPLAY} | dataset {dataset.dataset_id} | {dataset.n_channels} channels | K = {ctx.ks}")
    try:
        with measure(stats, "train"):
            tr = module.train(ctx) or {}
    finally:
        close_logger(ctx.log)
    result.update({k: v for k, v in tr.items() if k in ("status", "reason")})
    df = None
    if result["status"] == "ok":
        ctx.log = get_logger(f"{name}.test", dirs["logs"] / "test.log")
        try:
            with measure(stats, "test"):
                df = module.test(ctx)
        finally:
            close_logger(ctx.log)
    result.update({"train_seconds": stats["train"]["seconds"], "train_peak_python_mb": stats["train"]["peak_python_mb"],
                   "test_seconds": stats.get("test", {}).get("seconds", float("nan")),
                   "test_peak_python_mb": stats.get("test", {}).get("peak_python_mb", float("nan")),
                   "total_seconds": time.perf_counter() - t0, "process_rss_mb_end": rss_mb()})
    write_summary(ctx, result, df)
    result["per_k"] = summarize_per_k(df) if df is not None and len(df) else pd.DataFrame()
    return result


def summarize_per_k(df):
    """Mean over test samples for every K (numeric columns only)."""
    num = df.select_dtypes(include=[np.number, bool]).copy()
    num["k"] = df["k"]
    out = num.groupby("k").mean(numeric_only=True).reset_index()
    out.insert(1, "n_test_samples", df.groupby("k").size().values)
    return out


def write_summary(ctx, result, df):
    """metrics.csv / metrics.json / metrics_report.txt in the algorithm's root folder."""
    per_k = summarize_per_k(df) if df is not None and len(df) else pd.DataFrame()
    per_k.to_csv(ctx.root / "metrics.csv", index=False)
    comparable = [m for m in METRICS if m in per_k.columns]
    payload = {"algorithm": ctx.name, "display": ctx.display, "dataset_id": ctx.dataset.dataset_id,
               "dataset_path": str(ctx.dataset.path), "n_channels": ctx.C, "channel_names": ctx.names,
               "split": ctx.split, "k_values": ctx.ks, "run": {k: v for k, v in result.items() if k != "per_k"},
               "comparable_metrics": comparable, "per_k": per_k.to_dict(orient="records"), "info": ctx.info}
    save_json(payload, ctx.root / "metrics.json")
    lines = [f"{ctx.display} on {ctx.dataset.dataset_id}", "=" * 60,
             f"status: {result['status']}" + (f" ({result.get('reason')})" if result.get("reason") else ""),
             f"channels: {ctx.C}   split: {ctx.split['mode']}   K values: {ctx.ks}",
             f"training {result['train_seconds']:.2f} s, evaluation {result['test_seconds']:.2f} s, "
             f"total {result['total_seconds']:.2f} s", ""]
    if len(per_k):
        cols = ["k"] + [m for m in ["PSNR_dB", "SSIM", "RMSE", "MAE", "MSE", "accuracy", "f1_macro",
                                    "storage_saving_pct", "compression_ratio", "forward_seconds"] if m in per_k]
        lines.append(per_k[cols].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        lines.append("")
        lines.append("Direction: " + ", ".join(f"{m} {'higher' if METRICS[m]['direction'] == 'max' else 'lower'} "
                                              "is better" for m in cols[1:]))
    (ctx.root / "metrics_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ main.py entry point
def run_single(module, dataset_path=None, label_path=None, output_root=None, options=None):
    """What every algorithm's main.py calls. Returns a process exit code."""
    log = get_logger("kartavya")
    try:
        path = resolve_dataset_path(dataset_path)
        labels = cli_option("--labels", label_path)
        ds, split, root = prepare_dataset(path, labels, output_root or cli_option("--output"), log)
        res = run_algorithm(module, ds, split, root, options)
    except (DatasetError, ImportError, FileNotFoundError) as e:
        log.error(str(e))
        return 2
    except Exception:                                    # pragma: no cover
        log.error("Run failed:\n" + traceback.format_exc())
        return 1
    if res["status"] != "ok":
        log.warning(f"{module.DISPLAY}: {res['status']} - {res.get('reason', '')}")
        log.info(f"Reports written to {res['output_dir']}")
        return 3
    log.info(f"{module.DISPLAY} finished in {res['total_seconds']:.1f} s. Results: {res['output_dir']}")
    pk = res["per_k"]
    if len(pk):
        cols = [c for c in ["k", "PSNR_dB", "SSIM", "RMSE", "accuracy", "storage_saving_pct"] if c in pk]
        log.info("\n" + pk[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0
