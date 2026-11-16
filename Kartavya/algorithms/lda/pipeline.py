"""
Supervised LDA in the common pipeline - only if real ground-truth labels exist.

train: label report (labels found? >= 2 classes? enough pixels per class?) -> without valid labels nothing is
       fitted and the algorithm is reported as "not applicable" (no labels are invented). With labels: labelled
       TRAINING pixels -> standardise -> one LDA per K = 1 .. min(classes - 1, channels) -> models, training metrics
test : loads the saved models; projects ALL valid test pixels (reduced K-channel image), classifies the LABELLED
       test pixels: accuracy, precision, recall, F1, confusion matrix, Fisher separability, reduction %, storage,
       time, memory. LDA has no inverse transform, so reconstruction metrics do not apply (reported as NaN).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.lda import config as cfg
from algorithms.lda.implementation import LDA
from common import visualization as viz
from common.data_loader import rows_slice, sample_tag
from common.evaluation import dimensionality_reduction_pct, storage_report
from common.metrics import classification_report, fisher_ratio
from common.preprocessing import Standardizer, matrix_to_image
from common.utils import ensure_dirs, file_size_bytes, measure, rss_mb, save_json, set_seed, timed

NAME, DISPLAY, FAMILY, REQUIRES = "lda", "LDA", "linear projection (supervised)", []
REQUIREMENT = ("Class ids per pixel from a source independent of the input images (field data, an external "
               "land-cover map, or manual annotation). Labels made by thresholding or clustering these same pixels "
               "are circular: LDA would just recover the rule used to create them.")


def output_name():
    return NAME if cfg.STANDARDIZE else f"{NAME}_nostd"


def _entry_data(ctx, entry):
    img, valid = ctx.dataset.load_normalized(entry["index"])
    rs = rows_slice(valid.shape[0], entry["rows"], ctx.fraction)
    lab = ctx.dataset.labels_for(entry["index"])
    return img[rs], valid[rs], (lab[rs] if lab is not None else None)


def label_report(ctx):
    ds = ctx.dataset
    rep = {"label_files": [str(p) for p in ds.label_files], "matched": {ds.samples[i]["name"]: str(p)
                                                                        for i, p in ds.label_map.items()},
           "ignore_label": cfg.IGNORE_LABEL, "min_pixels_per_class": cfg.MIN_PIXELS_PER_CLASS,
           "usable": False, "reasons": [], "requirement": REQUIREMENT}
    if not ds.label_files:
        rep["reasons"].append("No label raster found next to the data (file names containing "
                              "'gt'/'label'/'labels'/'groundtruth' ...) and no LABEL_PATH given.")
    elif not ds.label_map:
        rep["reasons"].append("Label files were found but none could be matched to an image sample by file name.")
    train_with = [e for e in ctx.split["train"] if e["index"] in ds.label_map]
    if ds.label_map and not train_with:
        rep["reasons"].append("None of the TRAINING samples has a label raster.")
    rep["usable"] = not rep["reasons"]
    return rep


def train(ctx):
    log, dirs = ctx.log, ctx.dirs
    rng = set_seed(cfg.RANDOM_SEED)
    hp = {"standardize": cfg.STANDARDIZE, "shrinkage": cfg.SHRINKAGE, "max_train_pixels": cfg.MAX_TRAIN_PIXELS,
          "seed": cfg.RANDOM_SEED, "ignore_label": cfg.IGNORE_LABEL, "min_pixels_per_class": cfg.MIN_PIXELS_PER_CLASS}
    log.info(f"hyper-parameters {hp}")
    rep = label_report(ctx)
    xs, ys = [], []
    if rep["usable"]:
        for entry in ctx.split["train"]:
            img, valid, lab = _entry_data(ctx, entry)
            if lab is None:
                log.warning(f"{ctx.dataset.samples[entry['index']]['name']}: no label raster - its training pixels are not used")
                continue
            m = valid & (lab != cfg.IGNORE_LABEL)
            xs.append(img[m].astype(np.float64))
            ys.append(lab[m])
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        classes, counts = np.unique(y_tr, return_counts=True)
        rep["training_class_counts"] = {int(c): int(n) for c, n in zip(classes, counts)}
        rare = [int(c) for c, n in zip(classes, counts) if n < cfg.MIN_PIXELS_PER_CLASS]
        if len(classes) < 2:
            rep["reasons"].append(f"Training labels contain {len(classes)} class; LDA needs >= 2.")
        if rare:
            rep["reasons"].append(f"Classes {rare} have < {cfg.MIN_PIXELS_PER_CLASS} training pixels. "
                                  "Merge or relabel them (they are not dropped silently).")
        rep["usable"] = not rep["reasons"]
    save_json(rep, dirs["metrics"] / "label_report.json")
    if not rep["usable"]:
        log.warning("SUPERVISED LDA CANNOT BE TRAINED OR EVALUATED: no valid ground-truth labels.")
        for r in rep["reasons"]:
            log.warning(f"  - {r}")
        log.warning(f"  Requirement: {REQUIREMENT}")
        save_json({"algorithm": ctx.name, "status": "not_applicable_no_labels", "label_report": rep},
                  dirs["metrics"] / "train_summary.json")
        return {"status": "not_applicable", "reason": "no usable ground-truth labels: " + " ".join(rep["reasons"])}

    if cfg.MAX_TRAIN_PIXELS and len(y_tr) > cfg.MAX_TRAIN_PIXELS:
        pick = rng.choice(len(y_tr), cfg.MAX_TRAIN_PIXELS, replace=False)
        x_tr, y_tr = x_tr[pick], y_tr[pick]
    scaler = Standardizer(cfg.STANDARDIZE).fit(x_tr)
    z_tr = scaler.transform(x_tr)
    max_k = min(len(np.unique(y_tr)) - 1, z_tr.shape[1])
    req = cfg.COMPONENT_COUNTS or sorted(set(ctx.ks) | {max_k})
    ks = [k for k in req if 1 <= k <= max_k]
    refused = [k for k in req if k > max_k]
    if refused:
        log.info(f"K {refused} not possible: {len(np.unique(y_tr))} classes allow at most {max_k} LDA components")
    log.info(f"Fitting on {len(y_tr):,} labelled pixels, classes {rep['training_class_counts']}, K = {ks}")
    rows = []
    for k in ks:
        st = {}
        with measure(st, "fit"):
            model = LDA(k, cfg.SHRINKAGE).fit_with_counts(z_tr, y_tr)
        r = classification_report(y_tr, model.predict(z_tr), model.classes_)
        rows.append({"k": k, "max_components": max_k, "train_accuracy": r["accuracy"], "train_f1_macro": r["f1_macro"],
                     "fisher_original": fisher_ratio(z_tr, y_tr), "fisher_lda": fisher_ratio(model.transform(z_tr), y_tr),
                     "fit_seconds": st["fit"]["seconds"], "fit_peak_mb": st["fit"]["peak_python_mb"]})
        np.savez(dirs["models"] / f"lda_K{k}.npz", **{f"lda_{a}": np.asarray(v) for a, v in model.state().items()},
                 scaler_enabled=cfg.STANDARDIZE, scaler_mean=scaler.mean_, scaler_std=scaler.std_,
                 channel_names=np.array(ctx.names), dataset_id=ctx.dataset.dataset_id)
        log.info(f"K={k}: train acc {r['accuracy']:.4f} F1 {r['f1_macro']:.4f}, fit {st['fit']['seconds']:.3f} s")
    pd.DataFrame(rows).to_csv(dirs["metrics"] / "train_metrics.csv", index=False)
    save_json({"algorithm": ctx.name, "status": "trained", "hyperparameters": hp, "channels": ctx.names,
               "split": ctx.split, "component_counts": ks, "max_components": max_k, "label_report": rep,
               "train": rows, "process_rss_mb_end": rss_mb()}, dirs["metrics"] / "train_summary.json")
    ctx.info.update({"max_components": max_k, "classes": rep["training_class_counts"]})


def load_models(ctx):
    files = sorted(ctx.dirs["models"].glob("lda_K*.npz"), key=lambda p: int(p.stem.split("K")[1]))
    models, scaler = {}, None
    for p in files:
        d = np.load(p, allow_pickle=False)
        if str(d["dataset_id"]) != ctx.dataset.dataset_id:
            raise ValueError(f"{p.name} was trained on a different dataset")
        m = LDA.from_state({k[4:]: d[k] for k in d.files if k.startswith("lda_")})
        scaler = Standardizer.from_state({"enabled": bool(d["scaler_enabled"]), "mean": d["scaler_mean"],
                                          "std": d["scaler_std"]})
        models[m.scalings_.shape[1]] = m
    if not models:
        raise FileNotFoundError("No LDA models were saved")
    return models, scaler


def test(ctx):
    t0 = time.perf_counter()
    log, dirs = ctx.log, ctx.dirs
    models, scaler = load_models(ctx)
    ks, C = sorted(models), ctx.C
    classes = models[ks[0]].classes_
    class_names = [cfg.CLASS_NAMES.get(int(c), str(c)) for c in classes]
    n_scaler = 2 * C if scaler.enabled else 0
    log.info(f"Loaded saved models K = {ks}; classes {class_names}")
    rows = []
    for entry in ctx.split["test"]:
        s = ctx.dataset.samples[entry["index"]]
        tag = sample_tag(ctx.dataset, entry)
        img, valid, lab = _entry_data(ctx, entry)
        if lab is None:
            log.warning(f"{tag}: no label raster - skipped")
            continue
        z_all = scaler.transform(img[valid].astype(np.float64))
        labelled = (lab != cfg.IGNORE_LABEL)[valid]
        y = lab[valid][labelled]
        known = np.isin(y, classes)
        if (~known).any():
            log.warning(f"{tag}: {int((~known).sum())} pixels have classes unseen in training; excluded")
        zl, y = z_all[labelled][known], y[known]
        if len(y) == 0:
            log.warning(f"{tag}: no labelled test pixels - skipped")
            continue
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
            r = classification_report(y, pred, classes)
            sto = storage_report(int(valid.sum()), C, k, m.n_model_values() + n_scaler)
            rows.append({"sample_name": s["name"], "sample": tag, "k": k, "n_channels": C, "n_labelled": len(y),
                         "dimensionality_reduction_pct": dimensionality_reduction_pct(C, k),
                         **{key: r[key] for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                    "precision_weighted", "recall_weighted", "f1_weighted"]},
                         "fisher_original": fisher_ratio(zl, y), "fisher_lda": fisher_ratio(m.transform(zl), y),
                         **sto, "reduced_disk_bytes_npy": file_size_bytes(rp),
                         "original_disk_bytes_npy": file_size_bytes(orig_path),
                         "forward_seconds": t_fwd["seconds"], "forward_peak_mb": t_fwd["peak_python_mb"],
                         "transform_seconds": t_fwd["seconds"], "transform_peak_mb": t_fwd["peak_python_mb"],
                         "predict_seconds": t_pred["seconds"], "process_rss_mb": rss_mb(),
                         # reconstruction metrics do not exist for LDA (no inverse transform)
                         "PSNR_dB": np.nan, "SSIM": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAE": np.nan})
            save_json(r, dirs["metrics"] / f"classification_{tag}_K{k}.json")
            log.info(f"  K={k}: acc {r['accuracy']:.4f} P {r['precision_macro']:.4f} R {r['recall_macro']:.4f} "
                     f"F1 {r['f1_macro']:.4f} | reduction {rows[-1]['dimensionality_reduction_pct']:.1f}% | "
                     f"fwd {1000 * t_fwd['seconds']:.1f} ms")
            if k == ks[-1]:
                viz.plot_confusion_matrix(r["confusion_matrix"], class_names, dirs["graphs"] / f"confusion_{tag}.png",
                                          f"LDA confusion matrix, {s['name']}, K={k}")
                viz.plot_component_images(red_img, valid, dirs["graphs"] / f"components_{tag}.png",
                                          f"LDA discriminant components, {s['name']}", n_show=min(3, k), prefix="LD")
    if not rows:
        raise RuntimeError("No labelled test sample - LDA could not be evaluated (label the test region too).")
    df = pd.DataFrame(rows)
    df.to_csv(dirs["metrics"] / "test_metrics.csv", index=False)
    viz.plot_metric_panels(df, [("accuracy", "Accuracy"), ("f1_macro", "F1 (macro)"),
                                ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
                                ("fisher_lda", "Fisher separability after LDA"),
                                ("dimensionality_reduction_pct", "Dimensionality reduction (%)")],
                           dirs["graphs"] / "classification_vs_components.png",
                           f"LDA: class separation vs retained components ({ctx.dataset.name})", series_col="sample_name")
    save_json({"algorithm": ctx.name, "components_tested": ks, "rows": rows, "process_rss_mb_end": rss_mb(),
               "total_seconds": time.perf_counter() - t0}, dirs["metrics"] / "test_summary.json")
    return df


# ------------------------------------------------------------------------- self-check
def self_check(output_root):
    """
    Verify implementation.py + the evaluation code on REAL labelled data that ships with scikit-learn
    (Iris and Wine) against scikit-learn's LDA. This checks the code; it says nothing about any image dataset.
    """
    from sklearn.datasets import load_iris, load_wine
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    from common.utils import get_logger
    root = Path(output_root) / "lda_self_check"
    ensure_dirs(root / "metrics", root / "graphs", root / "logs")
    log = get_logger("lda.self_check", root / "logs" / "self_check.log")
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    rows = []
    for name, loader in [("iris", load_iris), ("wine", load_wine)]:
        bunch = loader()
        x, y = bunch.data, bunch.target
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
                         "fisher_lda": fisher_ratio(m.transform(xte), y[~tr]), "min_abs_corr_vs_sklearn": float(min(r)),
                         "sklearn_accuracy_same_k": float((sk.predict(xte) == y[~tr]).mean()),
                         "full_model_pred_agreement_vs_sklearn": float((pred == sk.predict(xte)).mean())})
            log.info(f"{name} K={k}: acc {rep['accuracy']:.3f} F1 {rep['f1_macro']:.3f} | |r| vs sklearn {min(r):.6f}")
            if k == max_k:
                viz.plot_confusion_matrix(rep["confusion_matrix"], list(bunch.target_names),
                                          root / "graphs" / f"confusion_{name}.png", f"Self-check: {name} test set, K={k}")
        try:
            LDA(max_k + 1).fit(xtr, y[tr])
            log.error("component limit NOT enforced")
        except ValueError as e:
            log.info(f"{name}: K={max_k + 1} correctly refused -> {e}")
    pd.DataFrame(rows).to_csv(root / "metrics" / "self_check.csv", index=False)
    log.info(f"Self-check written to {root / 'metrics' / 'self_check.csv'}")
    return root
