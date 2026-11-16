"""
Fit ICA on the TRAINING pixels only and save one model per component count K.

Run from the repository root:
    python Kartavya/algorithms/linear/ica/train.py                  # PNG products (8 channels)
    python Kartavya/algorithms/linear/ica/train.py --no-standardize # ICA on the [0,1] values directly
    python Kartavya/algorithms/linear/ica/train.py --dataset raw    # original B02/B03/B04/B08, if present

What it does
  1. lists images in sorted-filename order and saves the filename -> index mapping
  2. inspects every image (size, mode, dtype, range, map panel, colour-map decoding, alignment)
  3. builds the training feature matrix (valid TRAINING pixels x channels) normalised to [0, 1]
  4. fits the standardiser on training pixels, then FastICA for every K (unlike PCA, ICA models
     for different K are NOT nested, so each K is a separate fit)
  5. records convergence (iterations, final change, converged flag) and all hyper-parameters
  6. cross-checks every K against scikit-learn's FastICA
  7. saves models, tables, graphs and a log. Test data is never touched here.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd

from Kartavya.algorithms.linear.ica import config as cfg
from Kartavya.algorithms.linear.ica.ica import FastICA
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.metrics import mae, mse, psnr, rmse
from Kartavya.common.preprocessing import Standardizer, sample_rows
from Kartavya.common.utils import ensure_dirs, get_logger, measure, rss_mb, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train ICA for channel reduction")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--no-standardize", action="store_true", help="skip z-scoring before ICA")
    p.add_argument("--tag", default=None, help="experiment name suffix")
    return p.parse_args()


def experiment_name(args):
    return args.dataset + ("_nostd" if args.no_standardize else "") + (f"_{args.tag}" if args.tag else "")


def hyperparameters(standardize):
    return {"algorithm": cfg.ICA_ALGORITHM, "fun": cfg.ICA_FUN, "fun_alpha": cfg.ICA_FUN_ALPHA,
            "max_iter": cfg.ICA_MAX_ITER, "tol": cfg.ICA_TOL, "whitening": "PCA, unit variance",
            "random_seed": cfg.RANDOM_SEED, "max_train_pixels": cfg.MAX_TRAIN_PIXELS,
            "standardize": standardize, "rank_tol": cfg.RANK_TOL, "target_width": cfg.TARGET_WIDTH}


def sklearn_check(z_fit, ours, k, log):
    """Fit scikit-learn FastICA with the same settings and compare. ICA is only unique up to the
    order and sign of components, so components are matched by their largest |correlation|."""
    try:
        from sklearn.decomposition import FastICA as SkFastICA
    except ImportError:
        log.warning("scikit-learn not installed; skipping the cross-check")
        return {}
    sk = SkFastICA(n_components=k, algorithm="parallel", whiten="unit-variance", fun="logcosh",
                   fun_args={"alpha": cfg.ICA_FUN_ALPHA}, max_iter=cfg.ICA_MAX_ITER, tol=cfg.ICA_TOL,
                   random_state=cfg.RANDOM_SEED)
    s_sk = sk.fit_transform(z_fit)
    s_ours = ours.transform(z_fit)
    corr = np.abs(np.corrcoef(s_ours.T, s_sk.T)[:k, k:]) if k > 1 else np.array([[abs(np.corrcoef(s_ours[:, 0], s_sk[:, 0])[0, 1])]])
    rec_sk = sk.inverse_transform(s_sk)
    rec_ours = ours.inverse_transform(s_ours)
    return {"sklearn_n_iter": int(sk.n_iter_),
            "min_best_abs_corr_with_sklearn_components": float(corr.max(axis=1).min()),
            "one_to_one_match": bool(len(set(corr.argmax(axis=1))) == k),
            "max_abs_diff_reconstruction": float(np.abs(rec_sk - rec_ours).max())}


def main():
    args = parse_args()
    standardize = cfg.STANDARDIZE and not args.no_standardize
    exp = experiment_name(args)
    dirs = cfg.output_dirs(exp)
    ensure_dirs(*dirs.values())
    log = get_logger(f"ica.train.{exp}", dirs["logs"] / "train.log")
    rng = set_seed(cfg.RANDOM_SEED)
    t_start = time.perf_counter()
    hp = hyperparameters(standardize)
    log.info(f"Experiment '{exp}' | hyper-parameters {hp}")

    # ---------------------------------------------------------- 1. ordering + mapping + split
    mapping = dl.discover_samples(args.dataset, cfg, log)
    if not mapping["samples"]:
        where = cfg.RAW_DATASET_DIR if args.dataset == "raw" else cfg.PNG_DATASET_DIR
        log.error(f"No usable samples found in {where}."
                  + (" Original Sentinel-2 bands are not available, so the raw experiment is skipped "
                     "(no bands are fabricated)." if args.dataset == "raw" else ""))
        sys.exit(2)
    save_json(mapping, dirs["metrics"] / "filename_index_mapping.json")
    for s in mapping["samples"]:
        log.info(f"  index {s['index']}: date {s['date']} -> {s['files']}")
    split = dl.make_split(len(mapping["samples"]), cfg)
    save_json(split, dirs["models"] / "split.json")
    log.info(f"Split mode: {split['mode']} | train {split['train']} | test {split['test']}")
    if split["mode"] == "spatial":
        log.warning("Only ONE date is available: fitting on the top half of the image, testing on the bottom "
                    "half. With all five dates, earlier dates train and later dates test.")

    # ---------------------------------------------------------- 2. TRAINING pixels only
    stats = {}
    with measure(stats, "load_and_preprocess"):
        x_train, names, inspection = dl.load_split_pixels(args.dataset, mapping, split["train"], cfg,
                                                          cfg.SPATIAL_TRAIN_FRACTION, log)
    save_json(inspection, dirs["metrics"] / "dataset_inspection.json")
    x_fit = sample_rows(x_train, cfg.MAX_TRAIN_PIXELS, rng)
    log.info(f"Training matrix: {x_train.shape[0]:,} pixels x {x_train.shape[1]} channels; "
             f"fitting on {x_fit.shape[0]:,} sampled pixels. Channels: {names}")

    # ---------------------------------------------------------- 3. standardise (train statistics)
    scaler = Standardizer(standardize).fit(x_fit)
    z_fit = scaler.transform(x_fit)
    cov_eig = np.sort(np.linalg.eigvalsh(np.cov(z_fit.T)))[::-1]
    rank = FastICA.numerical_rank(np.clip(cov_eig, 0, None), cfg.RANK_TOL)
    ks = cfg.COMPONENT_COUNTS or list(range(1, rank + 1))
    skipped = [k for k in ks if k > rank]
    ks = [k for k in ks if k <= rank]
    log.info(f"Covariance eigenvalues: {np.round(cov_eig, 6).tolist()} -> numerical rank {rank}")
    if rank < len(names):
        log.info(f"K > {rank} is not fitted: {len(names) - rank} direction(s) have zero variance "
                 "(duplicated channels), so whitening them would divide by ~0.")
    if skipped:
        log.warning(f"Requested K {skipped} exceed the rank and are skipped.")

    # ---------------------------------------------------------- 4. one ICA model per K
    conv_rows, recon_rows, histories = [], [], {}
    for k in ks:
        st = {}
        with measure(st, "fit"):
            model = FastICA(k, cfg.ICA_FUN_ALPHA, cfg.ICA_MAX_ITER, cfg.ICA_TOL, cfg.RANDOM_SEED,
                            cfg.RANK_TOL).fit(z_fit)
        check = sklearn_check(z_fit, model, k, log)
        histories[f"K={k}"] = model.history_
        conv = {"k": k, "n_iter": model.n_iter_, "converged": model.converged_,
                "final_change": model.history_[-1], "tol": cfg.ICA_TOL, "max_iter": cfg.ICA_MAX_ITER,
                "fit_seconds": st["fit"]["seconds"], "fit_peak_mb": st["fit"]["peak_python_mb"], **check}
        conv_rows.append(conv)
        log.info(f"K={k}: {'converged' if model.converged_ else 'NOT converged'} in {model.n_iter_} iterations "
                 f"(final change {model.history_[-1]:.2e}, tol {cfg.ICA_TOL:g}), fit {st['fit']['seconds']:.2f} s, "
                 f"peak {st['fit']['peak_python_mb']:.0f} MB | sklearn: {check}")
        if not model.converged_:
            log.warning(f"K={k} hit max_iter={cfg.ICA_MAX_ITER}; consider raising ICA_MAX_ITER")

        rec = scaler.inverse_transform(model.inverse_transform(model.transform(z_fit)))
        recon_rows.append({"k": k, "MSE": mse(x_fit, rec), "RMSE": rmse(x_fit, rec), "MAE": mae(x_fit, rec),
                           "PSNR_dB": psnr(x_fit, rec)})

        path = dirs["models"] / f"ica_K{k}.npz"
        np.savez(path, **{f"ica_{key}": np.asarray(v) for key, v in model.state().items()},
                 scaler_enabled=standardize, scaler_mean=scaler.mean_, scaler_std=scaler.std_,
                 channel_names=np.array(names), experiment=exp, dataset=args.dataset,
                 target_width=cfg.TARGET_WIDTH)
        if k == ks[-1]:
            # mixing columns, scaled to unit length, show which channels each source drives
            mix = model.mixing_ / np.linalg.norm(model.mixing_, axis=0)
            viz.plot_loadings(mix.T, names, dirs["graphs"] / f"ica_mixing_K{k}.png", n_show=min(k, 6),
                              title=f"ICA mixing weights, K={k} (which channels each source drives)", prefix="IC")

    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv(dirs["metrics"] / "convergence.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(dirs["metrics"] / "train_reconstruction.csv", index=False)
    viz.plot_convergence(histories, cfg.ICA_TOL, dirs["graphs"] / "convergence.png",
                         f"FastICA convergence per component count ({exp})")
    viz.plot_correlation(np.corrcoef(x_fit.T), names, dirs["graphs"] / "channel_correlation.png")

    summary = {"experiment": exp, "dataset": args.dataset, "hyperparameters": hp, "channels": names,
               "split": split, "n_training_pixels_total": int(x_train.shape[0]),
               "n_pixels_used_for_fit": int(x_fit.shape[0]), "covariance_eigenvalues": cov_eig,
               "numerical_rank": rank, "component_counts": ks, "convergence": conv_rows,
               "timing_memory": stats, "process_rss_mb_end": rss_mb(),
               "total_seconds": time.perf_counter() - t_start}
    save_json(summary, dirs["metrics"] / "train_summary.json")
    log.info(f"Training finished in {summary['total_seconds']:.1f} s. Models: {dirs['models']}")


if __name__ == "__main__":
    main()
