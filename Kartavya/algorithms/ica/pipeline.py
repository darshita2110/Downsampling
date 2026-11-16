"""
ICA (FastICA) in the common pipeline.

train: TRAINING pixels -> standardise -> numerical rank -> one FastICA per K (models are NOT nested) ->
       convergence record + scikit-learn check -> model files, tables, graphs
test : loads the saved models; evaluates reconstruction, independence (ICA vs PCA axes of the same subspace)
       and verifies that ICA's reconstruction equals the K-dimensional PCA projection
"""
import time

import numpy as np
import pandas as pd

from algorithms.ica import config as cfg
from algorithms.ica.implementation import FastICA
from common import visualization as viz
from common.data_loader import load_split_pixels
from common.metrics import component_statistics, mae, mse, psnr, rmse
from common.preprocessing import Standardizer, sample_rows
from common.reduction_eval import EvalSpec, evaluate, write_test_summary
from common.utils import measure, rss_mb, save_json, set_seed

NAME, DISPLAY, FAMILY, REQUIRES = "ica", "ICA", "linear transform (unsupervised)", []


def output_name():
    return NAME if cfg.STANDARDIZE else f"{NAME}_nostd"


def hyperparameters():
    return {"algorithm": cfg.ICA_ALGORITHM, "fun": cfg.ICA_FUN, "fun_alpha": cfg.ICA_FUN_ALPHA,
            "max_iter": cfg.ICA_MAX_ITER, "tol": cfg.ICA_TOL, "whitening": "PCA, unit variance",
            "random_seed": cfg.RANDOM_SEED, "max_train_pixels": cfg.MAX_TRAIN_PIXELS,
            "standardize": cfg.STANDARDIZE, "rank_tol": cfg.RANK_TOL}


def sklearn_check(z_fit, ours, k, log):
    """scikit-learn FastICA with the same settings; components matched by their largest |correlation|."""
    if not cfg.SKLEARN_CHECK:
        return {}
    try:
        from sklearn.decomposition import FastICA as SkFastICA
    except ImportError:
        log.warning("scikit-learn not installed; skipping the cross-check")
        return {}
    import warnings
    sk = SkFastICA(n_components=k, algorithm="parallel", whiten="unit-variance", fun="logcosh",
                   fun_args={"alpha": cfg.ICA_FUN_ALPHA}, max_iter=cfg.ICA_MAX_ITER, tol=cfg.ICA_TOL,
                   random_state=cfg.RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_sk = sk.fit_transform(z_fit)
    s_ours = ours.transform(z_fit)
    corr = np.abs(np.corrcoef(s_ours.T, s_sk.T)[:k, k:]) if k > 1 else \
        np.array([[abs(np.corrcoef(s_ours[:, 0], s_sk[:, 0])[0, 1])]])
    return {"sklearn_n_iter": int(sk.n_iter_),
            "min_best_abs_corr_with_sklearn_components": float(corr.max(axis=1).min()),
            "one_to_one_match": bool(len(set(corr.argmax(axis=1))) == k),
            "max_abs_diff_reconstruction": float(np.abs(sk.inverse_transform(s_sk) - ours.inverse_transform(s_ours)).max())}


def train(ctx):
    log, dirs = ctx.log, ctx.dirs
    rng = set_seed(cfg.RANDOM_SEED)
    hp = hyperparameters()
    log.info(f"hyper-parameters {hp}")
    stats = {}
    with measure(stats, "load_and_preprocess"):
        x_train, names, inspection = load_split_pixels(ctx.dataset, ctx.split["train"], ctx.fraction, log, rng)
    save_json(inspection, dirs["metrics"] / "dataset_inspection.json")
    x_fit = sample_rows(x_train, cfg.MAX_TRAIN_PIXELS, rng)
    if len(x_fit) < 2:
        raise ValueError("ICA needs at least 2 valid training pixels")
    log.info(f"Training matrix: {x_train.shape[0]:,} pixels x {x_train.shape[1]} channels; fitting on {x_fit.shape[0]:,}")

    scaler = Standardizer(cfg.STANDARDIZE).fit(x_fit)
    z_fit = scaler.transform(x_fit)
    cov_eig = np.sort(np.linalg.eigvalsh(np.cov(z_fit.T)))[::-1]
    rank = FastICA.numerical_rank(np.clip(cov_eig, 0, None), cfg.RANK_TOL)
    C = len(names)
    req = cfg.COMPONENT_COUNTS or sorted(set(ctx.ks) | {C})
    ks = [k for k in req if 1 <= k <= rank]
    skipped = [k for k in req if k > rank]
    log.info(f"Covariance eigenvalues: {np.round(cov_eig, 6).tolist()[:20]} -> numerical rank {rank}")
    if skipped:
        log.warning(f"K {skipped} not fitted: {C - rank} direction(s) have zero variance (e.g. duplicated "
                    "channels), so whitening them would divide by ~0.")
    if not ks:
        raise ValueError("The training data have zero variance in every direction; ICA cannot be fitted")

    conv_rows, recon_rows, histories = [], [], {}
    for k in ks:
        st = {}
        with measure(st, "fit"):
            model = FastICA(k, cfg.ICA_FUN_ALPHA, cfg.ICA_MAX_ITER, cfg.ICA_TOL, cfg.RANDOM_SEED, cfg.RANK_TOL).fit(z_fit)
        check = sklearn_check(z_fit, model, k, log)
        histories[f"K={k}"] = model.history_
        conv_rows.append({"k": k, "n_iter": model.n_iter_, "converged": model.converged_,
                          "final_change": model.history_[-1], "tol": cfg.ICA_TOL, "max_iter": cfg.ICA_MAX_ITER,
                          "fit_seconds": st["fit"]["seconds"], "fit_peak_mb": st["fit"]["peak_python_mb"], **check})
        log.info(f"K={k}: {'converged' if model.converged_ else 'NOT converged'} in {model.n_iter_} iterations "
                 f"(final change {model.history_[-1]:.2e}), fit {st['fit']['seconds']:.2f} s | sklearn: {check}")
        if not model.converged_:
            log.warning(f"K={k} hit max_iter={cfg.ICA_MAX_ITER}; consider raising ICA_MAX_ITER")
        rec = scaler.inverse_transform(model.inverse_transform(model.transform(z_fit)))
        recon_rows.append({"k": k, "MSE": mse(x_fit, rec), "RMSE": rmse(x_fit, rec), "MAE": mae(x_fit, rec),
                           "PSNR_dB": psnr(x_fit, rec)})
        np.savez(dirs["models"] / f"ica_K{k}.npz", **{f"ica_{key}": np.asarray(v) for key, v in model.state().items()},
                 scaler_enabled=cfg.STANDARDIZE, scaler_mean=scaler.mean_, scaler_std=scaler.std_,
                 channel_names=np.array(names), dataset_id=ctx.dataset.dataset_id)
        if k == ks[-1]:
            mix = model.mixing_ / np.linalg.norm(model.mixing_, axis=0)
            viz.plot_loadings(mix.T, names, dirs["graphs"] / f"ica_mixing_K{k}.png", n_show=min(k, 6),
                              title=f"ICA mixing weights, K={k} (which channels each source drives)", prefix="IC")
    pd.DataFrame(conv_rows).to_csv(dirs["metrics"] / "convergence.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(dirs["metrics"] / "train_reconstruction.csv", index=False)
    viz.plot_convergence(histories, cfg.ICA_TOL, dirs["graphs"] / "convergence.png",
                         f"FastICA convergence per component count ({ctx.dataset.name})")
    with np.errstate(invalid="ignore", divide="ignore"):
        viz.plot_correlation(np.nan_to_num(np.corrcoef(x_fit.T)), names, dirs["graphs"] / "channel_correlation.png")
    save_json({"algorithm": ctx.name, "hyperparameters": hp, "channels": names, "split": ctx.split,
               "n_training_pixels_total": int(x_train.shape[0]), "n_pixels_used_for_fit": int(x_fit.shape[0]),
               "covariance_eigenvalues": cov_eig, "numerical_rank": rank, "component_counts": ks,
               "skipped_component_counts": skipped, "convergence": conv_rows, "timing_memory": stats,
               "process_rss_mb_end": rss_mb()}, dirs["metrics"] / "train_summary.json")
    ctx.info.update({"numerical_rank": rank, "skipped_k": skipped})


def load_models(ctx):
    files = sorted(ctx.dirs["models"].glob("ica_K*.npz"), key=lambda p: int(p.stem.split("K")[1]))
    if not files:
        raise FileNotFoundError(f"No ICA models in {ctx.dirs['models']}")
    models, scaler = {}, None
    for p in files:
        d = np.load(p, allow_pickle=False)
        if [str(n) for n in d["channel_names"]] != ctx.names or str(d["dataset_id"]) != ctx.dataset.dataset_id:
            raise ValueError(f"{p.name} was trained on a different dataset")
        m = FastICA.from_state({k[4:]: d[k] for k in d.files if k.startswith("ica_")})
        scaler = Standardizer.from_state({"enabled": bool(d["scaler_enabled"]), "mean": d["scaler_mean"],
                                          "std": d["scaler_std"]})
        models[m.n_components] = m
    return models, scaler


def test(ctx):
    t0 = time.perf_counter()
    models, scaler = load_models(ctx)
    ks = sorted(models)
    C = ctx.C
    n_scaler = 2 * C if scaler.enabled else 0
    for k in ks:
        m = models[k]
        ctx.log.info(f"  K={k}: trained with {m.n_iter_} iterations, converged={m.converged_}, tol={m.tol:g}")
    cache = {}

    def zstd(x):
        if cache.get("x") is not x:
            cache["x"], cache["z"] = x, scaler.transform(x)
        return cache["z"]

    def row_extra(k, x, s, rec):
        m = models[k]
        pca_rec = scaler.inverse_transform(m.project_onto_whitened_subspace(zstd(x)))
        return {"max_abs_diff_vs_pca_projection": float(np.abs(pca_rec - rec).max()),
                "train_n_iter": m.n_iter_, "train_converged": m.converged_}

    def side(k, x, s):
        m = models[k]
        z = zstd(x)
        pcs = (z - m.mean_) @ m.whitening_.T
        ica_stats, pca_stats = component_statistics(s), component_statistics(pcs)
        return {"ica_mean_abs_kurtosis": ica_stats["mean_abs_excess_kurtosis"],
                "pca_mean_abs_kurtosis": pca_stats["mean_abs_excess_kurtosis"],
                "ica_linear_corr": ica_stats["mean_abs_linear_corr"], "pca_linear_corr": pca_stats["mean_abs_linear_corr"],
                "ica_square_corr": ica_stats["mean_abs_square_corr"], "pca_square_corr": pca_stats["mean_abs_square_corr"],
                "ica_kurtosis_per_component": str(np.round(ica_stats["excess_kurtosis"], 3).tolist())}

    spec = EvalSpec(ks=ks, kind="components", method_label=f"ICA ({ctx.name})",
                    forward=lambda k, x: models[k].transform(zstd(x)),
                    inverse=lambda k, s: scaler.inverse_transform(models[k].inverse_transform(s)),
                    n_model_values=lambda k: models[k].n_model_values() + n_scaler,
                    gallery_k=cfg.GALLERY_K, timing_repeats=cfg.TIMING_REPEATS,
                    save_reconstructed=cfg.SAVE_RECONSTRUCTED_K, row_extra=row_extra, side=side,
                    quality_extra=("train_n_iter", "FastICA iterations to converge (training)"))
    df, _, _ = evaluate(ctx, spec)
    idf = pd.DataFrame(spec.side_records)
    idf.to_csv(ctx.dirs["metrics"] / "test_independence.csv", index=False)
    ifirst = idf[(idf["sample"] == idf["sample"].iloc[0]) & (idf.k > 1)]
    if len(ifirst):
        viz.plot_paired_bars(ifirst, "k", ["ICA sources", "PCA axes (same subspace)"],
                             [("Non-Gaussianity: mean |excess kurtosis|", ["ica_mean_abs_kurtosis", "pca_mean_abs_kurtosis"]),
                              ("Non-linear dependence: mean |corr(s_i^2, s_j^2)|", ["ica_square_corr", "pca_square_corr"])],
                             ctx.dirs["graphs"] / "independence_ica_vs_pca.png",
                             f"How independent are the components? ({ifirst['sample_name'].iloc[0]} test region)")
    indep = idf.groupby("k")[["ica_mean_abs_kurtosis", "pca_mean_abs_kurtosis", "ica_square_corr",
                              "pca_square_corr"]].mean().reset_index().to_dict(orient="records")
    write_test_summary(ctx, df, {"independence": indep,
                                 "max_abs_diff_vs_pca_projection": float(df["max_abs_diff_vs_pca_projection"].max())}, t0)
    return df
