"""
Run EVERY channel-reduction algorithm of the repository on one dataset and compare them.

    python run_all_algorithms.py                              # asks for the dataset path
    python run_all_algorithms.py --dataset PATH               # file or folder of any supported format
    python run_all_algorithms.py --dataset PATH --labels GT   # labels for LDA (otherwise auto-detected)
    python run_all_algorithms.py --dataset PATH --only pca,ga # a subset of algorithms
    python run_all_algorithms.py --dataset PATH --epochs 20   # shorter UBS-CNN training

Or set DATASET_PATH below and run the file. What happens:
  1. the dataset is inspected and validated ONCE (format, size, channels, dtype, no-data), the train/test split
     is decided and the [0, 1] normalisation is fitted on training pixels - identical for every algorithm
  2. every algorithm in algorithms/ is discovered and run in turn through its own pipeline (the same code its
     main.py uses); a failing algorithm is logged and the others continue
  3. outputs/<dataset_id>/comparison/ receives comparison.csv, comparison.json, metric_comparison/ (per-metric
     tables and graphs), execution-time comparison, a quality-vs-storage graph and final_report.txt
"""
# ============================== CONFIGURATION ==============================
DATASET_PATH = ""      # file or folder; "" = ask when run
LABEL_PATH = ""        # optional, for LDA; "" = auto-detect *gt* / *label* files next to the data
OUTPUT_ROOT = ""       # "" = Kartavya/outputs
ALGORITHMS = None      # None = every algorithm found in algorithms/; or e.g. ["pca", "ica", "ga"]
UBSCNN_EPOCHS = None   # None = algorithms/ubscnn/config.py (100); fewer epochs = quicker run
# ===========================================================================
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms import discover  # noqa: E402
from common import comparison  # noqa: E402
from common.data_loader import DatasetError  # noqa: E402
from common.pipeline import (cli_option, default_k_values, prepare_dataset, resolve_dataset_path,  # noqa: E402
                             run_algorithm)
from common.utils import get_logger  # noqa: E402


def main():
    log = get_logger("run_all")
    try:
        path = resolve_dataset_path(DATASET_PATH or None)
        labels = cli_option("--labels", LABEL_PATH or None)
        out_root = cli_option("--output", OUTPUT_ROOT or None)
        only = cli_option("--only", None)
        only = [s.strip() for s in only.split(",")] if only else ALGORITHMS
        epochs = cli_option("--epochs", UBSCNN_EPOCHS, int)
        ds, split, root = prepare_dataset(path, labels, out_root, log)
    except (DatasetError, FileNotFoundError, ValueError) as e:
        log.error(str(e))
        return 2
    cmp_dir = root / "comparison"
    if cmp_dir.exists():
        shutil.rmtree(cmp_dir)                       # a comparison never mixes files of an earlier run
    log = get_logger("run_all", cmp_dir / "run_all.log")
    log.info(f"Dataset {ds.dataset_id}: {ds.n_samples} sample(s), {ds.n_channels} channels; split {split['mode']}")
    ks = default_k_values(ds.n_channels)
    log.info(f"Common K values for every algorithm: {ks}")
    algos = discover(only, log)
    log.info(f"Algorithms: {[n for n, _, _ in algos]}")

    results, t0 = [], time.perf_counter()
    for i, (name, module, load_error) in enumerate(algos, 1):
        log.info(f"[{i}/{len(algos)}] ===================== {name.upper()} =====================")
        if module is None:
            results.append({"algorithm": name, "display": name.upper(), "status": "failed",
                            "error_short": load_error, "per_k": None})
            continue
        try:
            res = run_algorithm(module, ds, split, root, options={"epochs": epochs}, ks=ks)
            log.info(f"{module.DISPLAY}: {res['status']} in {res['total_seconds']:.1f} s -> {res['output_dir']}")
            results.append(res)
        except Exception as e:                       # keep going with the remaining algorithms
            tb = traceback.format_exc()
            log.error(f"{module.DISPLAY} FAILED: {type(e).__name__}: {e}")
            log.debug(tb)
            (cmp_dir / f"error_{name}.txt").write_text(tb, encoding="utf-8")
            results.append({"algorithm": module.output_name(), "display": module.DISPLAY, "family": module.FAMILY,
                            "status": "failed", "error_short": f"{type(e).__name__}: {e}", "per_k": None,
                            "output_dir": str(root / module.output_name())})

    payload = comparison.build(results, ds, split, ks, cmp_dir, log)
    log.info(f"All algorithms finished in {time.perf_counter() - t0:.1f} s")
    for m in ["PSNR_dB", "SSIM", "RMSE", "accuracy"]:
        b = payload["best_per_metric"].get(m)
        if b and b["best_algorithm"]:
            log.info(f"  best {b['label']} ({'max' if b['direction'] == 'max' else 'min'}): {b['best_algorithm']} "
                     f"({b['best_value']:.4f} mean over K {b['common_k']})")
    log.info(f"Report: {cmp_dir / 'final_report.txt'}")
    failed = [r["display"] for r in results if r["status"] == "failed"]
    if failed:
        log.warning(f"Failed: {failed} (details in {cmp_dir}/error_*.txt)")
    return 0 if len(failed) < len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
