"""
Evaluate the SAVED LDA models on the test samples. Nothing is re-fitted.

Run from the repository root (same options as train.py):
    python Kartavya/algorithms/linear/lda/test.py

If train.py found no valid labels (metrics/label_report.json says usable = false) this script prints why
and exits with code 3 - there is nothing honest to evaluate.

With labels, for every test sample and K it
  1. projects ALL valid pixels to K discriminant components and saves them as a K-channel image
  2. classifies the LABELLED test pixels and reports accuracy, precision, recall, F1, confusion matrix
  3. reports Fisher separability before/after, reduction %, storage, runtime and memory
LDA has no inverse transform (it discards everything that does not separate classes), so no
reconstruction metrics are computed - that is a property of the method, not a missing feature.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd

from Kartavya.algorithms.linear.lda import config as cfg
from Kartavya.algorithms.linear.lda.lda import LDA
from Kartavya.algorithms.linear.lda.train import EXIT_NO_LABELS, load_entry
from Kartavya.common import visualization as viz
from Kartavya.common.evaluation import dimensionality_reduction_pct, storage_report
from Kartavya.common.metrics import classification_report, fisher_ratio
from Kartavya.common.preprocessing import Standardizer, matrix_to_image
from Kartavya.common.utils import ensure_dirs, file_size_bytes, get_logger, load_json, measure, rss_mb, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved LDA models")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def load_model(path):
    d = np.load(path, allow_pickle=False)
    m = LDA.from_state({k[4:]: d[k] for k in d.files if k.startswith("lda_")})
    sc = Standardizer.from_state({"enabled": bool(d["scaler_enabled"]), "mean": d["scaler_mean"],
                                  "std": d["scaler_std"]})
    return m, sc, [str(n) for n in d["channel_names"]], int(d["target_width"])


def timed(fn, repeats):
    runs, out = [], None
    for _ in range(repeats):
        st = {}
        with measure(st, "x"):
            out = fn()
        runs.append(st["x"])
    return out, {k: float(np.median([r[k] for r in runs])) for k in runs[0]}


def main():
    args = parse_args()
    exp = args.dataset + ("_nostd" if args.no_standardize else "") + (f"_{args.tag}" if args.tag else "")
    dirs = cfg.output_dirs(exp)
    ensure_dirs(dirs["logs"])
    log = get_logger(f"lda.test.{exp}", dirs["logs"] / "test.log")
    t_start = time.perf_counter()

    report_path = dirs["metrics"] / "label_report.json"
    if not report_path.exists():
        log.error("No label report - run train.py first.")
        sys.exit(2)
    report = load_json(report_path)
    if not report["usable"]:
        log.error("Supervised LDA cannot be evaluated: train.py found no valid ground-truth labels.")
        for r in report["reasons"]:
            log.error(f"  - {r}")
        log.error("Accuracy, precision, recall, F1 and the confusion matrix are NOT computed, because any "
                  "number would be based on invented labels.")
        sys.exit(EXIT_NO_LABELS)

    files = sorted(dirs["models"].glob("lda_K*.npz"), key=lambda p: int(p.stem.split("K")[1]))
    models = {}
    for p in files:
        m, scaler, names, width = load_model(p)
        models[m.scalings_.shape[1]] = m
    cfg.TARGET_WIDTH = width
    ks, C = sorted(models), len(names)
    split = load_json(dirs["models"] / "split.json")
    mapping = load_json(dirs["metrics"] / "filename_index_mapping.json")
    classes = models[ks[0]].classes_
    class_names = [cfg.CLASS_NAMES.get(int(c), str(c)) for c in classes]
    n_scaler = 2 * C if scaler.enabled else 0
    ensure_dirs(*dirs.values())
    log.info(f"Loaded saved models K = {ks}; classes {class_names}")

    rows = []
    for entry in split["test"]:
        s, data, rs, lab = load_entry(args.dataset, mapping, entry, cfg, report, split.get("spatial_train_fraction", 0.5))
        tag = f"{entry['index']:02d}_{s['date']}" + ("" if entry["rows"] == "all" else f"_{entry['rows']}")
        if lab is None:
            log.warning(f"{tag}: no label raster - skipped")
            continue
        img, valid = data["channels"][rs], data["valid"][rs]
        z_all = scaler.transform(img[valid].astype(np.float64))
        labelled = (lab != cfg.IGNORE_LABEL)[valid]
        y = lab[valid][labelled]
        known = np.isin(y, classes)
        if (~known).any():
            log.warning(f"{tag}: {int((~known).sum())} pixels have classes unseen in training; excluded")
        zl, y = z_all[labelled][known], y[known]
        log.info(f"{tag}: {valid.sum():,} valid pixels, {len(y):,} labelled test pixels")
        ensure_dirs(dirs["reduced_data"] / tag)
        orig_path = dirs["reduced_data"] / tag / "original_normalized.npy"
        np.save(orig_path, img.astype(np.float32))
        for k in ks:
            m = models[k]
            red, t_fwd = timed(lambda: m.transform(z_all), cfg.TIMING_REPEATS)
            pred, t_pred = timed(lambda: m.predict(zl), 1)
            red_img = matrix_to_image(red.astype(np.float32), valid)
            rp = dirs["reduced_data"] / tag / f"reduced_K{k}.npy"
            np.save(rp, red_img)
            rep = classification_report(y, pred, classes)
            sto = storage_report(int(valid.sum()), C, k, m.n_model_values() + n_scaler)
            rows.append({"date": s["date"], "sample": tag, "k": k, "n_labelled": len(y),
                         "dimensionality_reduction_pct": dimensionality_reduction_pct(C, k),
                         **{key: rep[key] for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                      "precision_weighted", "recall_weighted", "f1_weighted"]},
                         "fisher_original": fisher_ratio(zl, y), "fisher_lda": fisher_ratio(m.transform(zl), y),
                         **sto, "reduced_disk_bytes_npy": file_size_bytes(rp),
                         "original_disk_bytes_npy": file_size_bytes(orig_path),
                         "transform_seconds": t_fwd["seconds"], "transform_peak_mb": t_fwd["peak_python_mb"],
                         "predict_seconds": t_pred["seconds"], "process_rss_mb": rss_mb()})
            save_json(rep, dirs["metrics"] / f"classification_{tag}_K{k}.json")
            log.info(f"  K={k}: acc {rep['accuracy']:.4f} P {rep['precision_macro']:.4f} R {rep['recall_macro']:.4f} "
                     f"F1 {rep['f1_macro']:.4f} | reduction {rows[-1]['dimensionality_reduction_pct']:.1f}% | "
                     f"fwd {1000 * t_fwd['seconds']:.1f} ms")
            if k == ks[-1]:
                viz.plot_confusion_matrix(rep["confusion_matrix"], class_names, dirs["graphs"] / f"confusion_{tag}.png",
                                          f"LDA confusion matrix, {s['date']}, K={k}")
                viz.plot_component_images(red_img, valid, dirs["graphs"] / f"components_{tag}.png",
                                          f"LDA discriminant components, {s['date']}", n_show=min(3, k), prefix="LD")
    if not rows:
        log.error("No labelled test sample - nothing evaluated.")
        sys.exit(EXIT_NO_LABELS)
    df = pd.DataFrame(rows)
    df.to_csv(dirs["metrics"] / "test_metrics.csv", index=False)
    viz.plot_metric_panels(df, [("accuracy", "Accuracy"), ("f1_macro", "F1 (macro)"),
                                ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
                                ("fisher_lda", "Fisher separability after LDA"),
                                ("dimensionality_reduction_pct", "Dimensionality reduction (%)")],
                           dirs["graphs"] / "classification_vs_components.png",
                           f"LDA: class separation vs retained components ({exp})")
    save_json({"experiment": exp, "components_tested": ks, "rows": rows, "process_rss_mb_end": rss_mb(),
               "total_seconds": time.perf_counter() - t_start}, dirs["metrics"] / "test_summary.json")
    log.info(f"Testing finished. Metrics: {dirs['metrics']}")


if __name__ == "__main__":
    main()
