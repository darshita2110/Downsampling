"""
Train supervised LDA - only if real ground-truth labels exist.

Run from the repository root:
    python Kartavya/algorithms/linear/lda/train.py                 # Raichur PNG products
    python Kartavya/algorithms/linear/lda/train.py --dataset raw   # original bands, if present
    python Kartavya/algorithms/linear/lda/train.py --self-check    # verify the code on real labelled
                                                                   # benchmark data (Iris, Wine)

Steps for the Raichur data
  1. sort files, save the filename -> index mapping, decide the train/test split
  2. load + inspect the TRAINING images (sizes, channels, dtypes, ranges, alignment)
  3. look for label rasters; write metrics/label_report.json
       -> no valid labels: explain why, fit NOTHING, exit with code 3 (no labels are invented)
  4. with labels: keep labelled training pixels, standardise with training statistics,
     fit one LDA per K = 1 .. min(classes - 1, channels), save models, training metrics, log
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
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.metrics import classification_report, fisher_ratio
from Kartavya.common.preprocessing import Standardizer
from Kartavya.common.utils import ensure_dirs, get_logger, measure, rss_mb, save_json, set_seed

EXIT_NO_LABELS = 3


def parse_args():
    p = argparse.ArgumentParser(description="Train supervised LDA (needs real labels)")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--self-check", action="store_true",
                   help="verify the implementation on scikit-learn's bundled Iris and Wine datasets")
    return p.parse_args()


# ------------------------------------------------------------------------- self-check
def self_check():
    """
    Verify lda.py + the evaluation code on REAL labelled data that ships with scikit-learn
    (Iris: 150 flowers, 4 features, 3 species; Wine: 178 wines, 13 features, 3 cultivars).
    This checks the code; it says nothing about satellite images.
    """
    from sklearn.datasets import load_iris, load_wine
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    dirs = cfg.output_dirs("self_check")
    ensure_dirs(dirs["metrics"], dirs["graphs"], dirs["logs"])
    log = get_logger("lda.self_check", dirs["logs"] / "self_check.log")
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    rows = []
    for name, loader in [("iris", load_iris), ("wine", load_wine)]:
        bunch = loader()
        x, y = bunch.data, bunch.target
        # stratified 70/30 split: 70 % of every class trains
        tr = np.zeros(len(y), bool)
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == c))
            tr[idx[:int(round(0.7 * len(idx)))]] = True
        sc = Standardizer(True).fit(x[tr])
        xtr, xte = sc.transform(x[tr]), sc.transform(x[~tr])
        max_k = min(len(np.unique(y)) - 1, x.shape[1])
        for k in range(1, max_k + 1):
            m = LDA(k).fit_with_counts(xtr, y[tr])
            sk = SkLDA(solver="eigen", n_components=k).fit(xtr, y[tr])
            r = [abs(np.corrcoef(m.transform(xte)[:, i], sk.transform(xte)[:, i])[0, 1]) for i in range(k)]
            pred = m.predict(xte)
            rep = classification_report(y[~tr], pred, np.unique(y))
            rows.append({"dataset": name, "k": k, "max_components": max_k, "accuracy": rep["accuracy"],
                         "precision_macro": rep["precision_macro"], "recall_macro": rep["recall_macro"],
                         "f1_macro": rep["f1_macro"], "fisher_original": fisher_ratio(xte, y[~tr]),
                         "fisher_lda": fisher_ratio(m.transform(xte), y[~tr]),
                         "min_abs_corr_vs_sklearn": float(min(r)),
                         "sklearn_accuracy_same_k": float((sk.predict(xte) == y[~tr]).mean()),
                         "full_model_pred_agreement_vs_sklearn": float((pred == sk.predict(xte)).mean())})
            log.info(f"{name} K={k}: acc {rep['accuracy']:.3f} F1 {rep['f1_macro']:.3f} | |r| vs sklearn "
                     f"{min(r):.6f} | Fisher {rows[-1]['fisher_original']:.2f} -> {rows[-1]['fisher_lda']:.2f}")
            if k == max_k:
                viz.plot_confusion_matrix(rep["confusion_matrix"], list(bunch.target_names),
                                          dirs["graphs"] / f"confusion_{name}.png",
                                          f"Self-check: {name} test set, K={k}")
        try:
            LDA(max_k + 1).fit(xtr, y[tr])
            log.error("component limit NOT enforced")
        except ValueError as e:
            log.info(f"{name}: K={max_k + 1} correctly refused -> {e}")
    pd.DataFrame(rows).to_csv(dirs["metrics"] / "self_check.csv", index=False)
    log.info(f"Self-check written to {dirs['metrics'] / 'self_check.csv'}")


# ------------------------------------------------------------------------- main path
def load_entry(experiment, mapping, entry, cfg_, label_report, fraction):
    """Normalised image rows for one split entry, plus its labels on the same grid (or None)."""
    s = mapping["samples"][entry["index"]]
    data = dl.load_normalized_sample(experiment, s, cfg_)
    rs = dl.rows_slice(data["valid"].shape[0], entry["rows"], fraction)
    labels = None
    f = label_report["per_date"].get(s["date"], {}).get("file")
    if f:
        labels = dl.load_label_raster(Path(cfg_.LABEL_DIR) / f, data["grid_hw"])[rs]
    return s, data, rs, labels


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return
    standardize = cfg.STANDARDIZE and not args.no_standardize
    exp = args.dataset + ("_nostd" if args.no_standardize else "") + (f"_{args.tag}" if args.tag else "")
    dirs = cfg.output_dirs(exp)
    ensure_dirs(dirs["metrics"], dirs["logs"], dirs["models"], dirs["graphs"])
    log = get_logger(f"lda.train.{exp}", dirs["logs"] / "train.log")
    rng = set_seed(cfg.RANDOM_SEED)
    t_start = time.perf_counter()
    hp = {"standardize": standardize, "shrinkage": cfg.SHRINKAGE, "max_train_pixels": cfg.MAX_TRAIN_PIXELS,
          "seed": cfg.RANDOM_SEED, "ignore_label": cfg.IGNORE_LABEL, "min_pixels_per_class": cfg.MIN_PIXELS_PER_CLASS}
    log.info(f"Experiment '{exp}' | hyper-parameters {hp}")

    # 1. ordering + split
    mapping = dl.discover_samples(args.dataset, cfg, log)
    if not mapping["samples"]:
        log.error(f"No usable samples found for dataset '{args.dataset}' (nothing is fabricated).")
        sys.exit(2)
    save_json(mapping, dirs["metrics"] / "filename_index_mapping.json")
    for s in mapping["samples"]:
        log.info(f"  index {s['index']}: date {s['date']} -> {s['files']}")
    split = dl.make_split(len(mapping["samples"]), cfg)
    save_json(split, dirs["models"] / "split.json")
    log.info(f"Split mode: {split['mode']} | train {split['train']} | test {split['test']}")

    # 2. inspection of the training images (always done, labels or not)
    stats = {}
    with measure(stats, "load_and_inspect"):
        _, names, inspection = dl.load_split_pixels(args.dataset, mapping, split["train"], cfg,
                                                    cfg.SPATIAL_TRAIN_FRACTION, log)
    save_json(inspection, dirs["metrics"] / "dataset_inspection.json")
    log.info(f"Channels ({len(names)}): {names}")

    # 3. labels
    report = dl.label_availability(mapping, cfg.LABEL_DIR, cfg.LABEL_PATTERNS, cfg.IGNORE_LABEL,
                                   cfg.MIN_PIXELS_PER_CLASS)
    report["searched_also"] = ("Repository files, data/processed/sentinel2 (only RGB/False_Color/NDVI/NDWI "
                               "renderings), the Parquet/CSV schema in docs/DATA_ACCESS_GUIDE.md (x, y, lat, lon, "
                               "date, B02-B08, NDVI, NDWI, empty LST/population/infrastructure: no class column).")
    if report["usable"]:
        ys, xs = [], []
        for entry in split["train"]:
            s, data, rs, lab = load_entry(args.dataset, mapping, entry, cfg, report, cfg.SPATIAL_TRAIN_FRACTION)
            if lab is None:
                log.warning(f"{s['date']}: no label raster - its training pixels are not used")
                continue
            m = data["valid"][rs] & (lab != cfg.IGNORE_LABEL)
            xs.append(data["channels"][rs][m].astype(np.float64))
            ys.append(lab[m])
        if not xs:
            report["usable"] = False
            report["reasons"].append("None of the TRAINING dates has a label raster.")
        else:
            x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
            classes, counts = np.unique(y_tr, return_counts=True)
            report["training_class_counts"] = {int(c): int(n) for c, n in zip(classes, counts)}
            rare = [int(c) for c, n in zip(classes, counts) if n < cfg.MIN_PIXELS_PER_CLASS]
            if len(classes) < 2:
                report["usable"] = False
                report["reasons"].append(f"Training labels contain {len(classes)} class; LDA needs >= 2.")
            if rare:
                report["usable"] = False
                report["reasons"].append(f"Classes {rare} have < {cfg.MIN_PIXELS_PER_CLASS} training pixels. "
                                         "Merge or relabel them (they are not dropped silently).")
    save_json(report, dirs["metrics"] / "label_report.json")

    if not report["usable"]:
        log.error("SUPERVISED LDA CANNOT BE TRAINED OR EVALUATED: no valid ground-truth labels.")
        for r in report["reasons"]:
            log.error(f"  - {r}")
        log.error(f"  Requirement: {report['requirement']}")
        log.error(f"  Expected files: {cfg.LABEL_DIR}/" + " or ".join(cfg.LABEL_PATTERNS))
        log.error("  No labels were invented and no model was fitted. See documentation/LDA.md.")
        save_json({"experiment": exp, "status": "not_trained_no_labels", "label_report": report,
                   "timing_memory": stats, "total_seconds": time.perf_counter() - t_start},
                  dirs["metrics"] / "train_summary.json")
        sys.exit(EXIT_NO_LABELS)

    # 4. fit
    if cfg.MAX_TRAIN_PIXELS and len(y_tr) > cfg.MAX_TRAIN_PIXELS:
        pick = rng.choice(len(y_tr), cfg.MAX_TRAIN_PIXELS, replace=False)
        x_tr, y_tr = x_tr[pick], y_tr[pick]
    scaler = Standardizer(standardize).fit(x_tr)
    z_tr = scaler.transform(x_tr)
    max_k = min(len(np.unique(y_tr)) - 1, z_tr.shape[1])
    ks = [k for k in (cfg.COMPONENT_COUNTS or range(1, max_k + 1)) if k <= max_k]
    refused = [k for k in (cfg.COMPONENT_COUNTS or []) if k > max_k]
    if refused:
        log.warning(f"K {refused} refused: {len(np.unique(y_tr))} classes allow at most {max_k} LDA components")
    log.info(f"Fitting on {len(y_tr):,} labelled pixels, classes {report['training_class_counts']}, K = {ks}")
    rows = []
    for k in ks:
        st = {}
        with measure(st, "fit"):
            model = LDA(k, cfg.SHRINKAGE).fit_with_counts(z_tr, y_tr)
        rep = classification_report(y_tr, model.predict(z_tr), model.classes_)
        rows.append({"k": k, "max_components": max_k, "train_accuracy": rep["accuracy"],
                     "train_f1_macro": rep["f1_macro"], "fisher_original": fisher_ratio(z_tr, y_tr),
                     "fisher_lda": fisher_ratio(model.transform(z_tr), y_tr),
                     "fit_seconds": st["fit"]["seconds"], "fit_peak_mb": st["fit"]["peak_python_mb"]})
        np.savez(dirs["models"] / f"lda_K{k}.npz", **{f"lda_{a}": np.asarray(v) for a, v in model.state().items()},
                 scaler_enabled=standardize, scaler_mean=scaler.mean_, scaler_std=scaler.std_,
                 channel_names=np.array(names), target_width=cfg.TARGET_WIDTH)
        log.info(f"K={k}: train acc {rep['accuracy']:.4f} F1 {rep['f1_macro']:.4f}, fit {st['fit']['seconds']:.3f} s")
    pd.DataFrame(rows).to_csv(dirs["metrics"] / "train_metrics.csv", index=False)
    save_json({"experiment": exp, "status": "trained", "hyperparameters": hp, "channels": names, "split": split,
               "component_counts": ks, "max_components": max_k, "label_report": report, "train": rows,
               "timing_memory": stats, "process_rss_mb_end": rss_mb(),
               "total_seconds": time.perf_counter() - t_start}, dirs["metrics"] / "train_summary.json")
    log.info(f"Training finished. Models in {dirs['models']}")


if __name__ == "__main__":
    main()
