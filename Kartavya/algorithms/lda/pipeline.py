"""
PCA in the common pipeline.

train: TRAINING pixels -> standardise (training mean/std) -> fit PCA -> scikit-learn cross-check ->
       explained-variance table, training reconstruction error, model file, graphs
test : loads the saved model; for every K projects the TEST pixels, reconstructs, and evaluates
"""
import numpy as np
import pandas as pd

from algorithms.pca import config as cfg
from algorithms.pca.implementation import PCA
from common import visualization as viz
from common.data_loader import load_split_pixels
from common.metrics import mae, mse, psnr, rmse
from common.preprocessing import Standardizer, sample_rows
from common.reduction_eval import EvalSpec, evaluate, write_test_summary
from common.utils import measure, rss_mb, save_json, set_seed

NAME, DISPLAY, FAMILY, REQUIRES = "pca", "PCA", "linear transform (unsupervised)", []


def output_name():
    return NAME if cfg.STANDARDIZE else f"{NAME}_nostd"


def train(ctx):
    log, dirs = ctx.log, ctx.dirs
    rng = set_seed(cfg.RANDOM_SEED)
    log.info(f"standardize={cfg.STANDARDIZE} | max_train_pixels={cfg.MAX_TRAIN_PIXELS} | seed={cfg.RANDOM_SEED}")
    stats = {}
    with measure(stats, "load_and_preprocess"):
        x_all, names, inspection = load_split_pixels(ctx.dataset, ctx.split["train"], ctx.fraction, log, rng)
    save_json(inspection, dirs["metrics"] / "dataset_inspection.json")
    x_fit = sample_rows(x_all, cfg.MAX_TRAIN_PIXELS, rng)
    if len(x_fit) < 2:
        raise ValueError("PCA needs at least 2 valid training pixels")
    log.info(f"Feature matrix: {x_all.shape[0]:,} pixels x {x_all.shape[1]} channels; fitting on {x_fit.shape[0]:,}")

    scaler = Standardizer(cfg.STANDARDIZE).fit(x_fit)
    z_fit = scaler.transform(x_fit)
    with measure(stats, "pca_fit"):
        model = PCA().fit(z_fit)
    log.info(f"PCA fit in {stats['pca_fit']['seconds']:.3f} s, peak {stats['pca_fit']['peak_python_mb']:.1f} MB")

    evr = model.all_explained_variance_ratio_
    cum = np.cumsum(evr)
    ev_table = pd.DataFrame({"component": np.arange(1, len(evr) + 1), "eigenvalue": model.explained_variance_,
                             "explained_variance_ratio": evr, "cumulative_explained_variance": cum})
    ev_table.to_csv(dirs["metrics"] / "explained_variance.csv", index=False)
    for _, r in ev_table.head(20).iterrows():
        log.info(f"  PC{int(r.component)}: {100 * r.explained_variance_ratio:6.2f}%  cumulative "
                 f"{100 * r.cumulative_explained_variance:6.2f}%")
    k_for = {f"{int(t * 100)}%": model.components_for_variance(t) for t in cfg.VARIANCE_THRESHOLDS}
    log.info(f"Components needed: {k_for}")

    check = {}
    if cfg.SKLEARN_CHECK:
        try:
            from sklearn.decomposition import PCA as SkPCA
            sk = SkPCA(svd_solver="full").fit(z_fit)
            check["max_abs_diff_explained_variance_ratio"] = float(np.abs(sk.explained_variance_ratio_ - evr).max())
            # same axes up to sign; (near-)zero-variance axes are not unique and are skipped
            cos = np.abs(np.sum(sk.components_ * model.components_, axis=1))
            nondeg = evr > 1e-9
            check["n_zero_variance_components"] = int((~nondeg).sum())
            check["min_abs_cosine_nonzero_components"] = float(cos[nondeg].min()) if nondeg.any() else float("nan")
            k2 = min(2, model.n_features_)
            rec_ours = model.inverse_transform(model.transform(z_fit[:20000], k2))
            sk2 = SkPCA(n_components=k2, svd_solver="full").fit(z_fit)
            rec_sk = sk2.inverse_transform(sk2.transform(z_fit[:20000]))
            check[f"max_abs_diff_reconstruction_k{k2}"] = float(np.abs(rec_ours - rec_sk).max())
            log.info(f"scikit-learn cross-check: {check}")
        except ImportError:
            log.warning("scikit-learn not installed; skipping the cross-check")

    rows = []
    for k in range(1, model.n_features_ + 1):
        rec = scaler.inverse_transform(model.inverse_transform(model.transform(z_fit, k)))
        rows.append({"k": k, "MSE": mse(x_fit, rec), "RMSE": rmse(x_fit, rec), "MAE": mae(x_fit, rec),
                     "PSNR_dB": psnr(x_fit, rec), "cumulative_explained_variance": cum[k - 1]})
    pd.DataFrame(rows).to_csv(dirs["metrics"] / "train_reconstruction.csv", index=False)

    model_path = dirs["models"] / "pca_model.npz"
    np.savez(model_path, **{f"pca_{k}": np.asarray(v) for k, v in model.state().items()},
             scaler_enabled=cfg.STANDARDIZE, scaler_mean=scaler.mean_, scaler_std=scaler.std_,
             channel_names=np.array(names), dataset_id=ctx.dataset.dataset_id)
    log.info(f"Saved model -> {model_path}")

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(x_fit.T))
    viz.plot_explained_variance(evr, dirs["graphs"] / "explained_variance.png",
                                title=f"Explained variance per component ({ctx.dataset.name})",
                                thresholds=cfg.VARIANCE_THRESHOLDS)
    viz.plot_correlation(corr, names, dirs["graphs"] / "channel_correlation.png")
    viz.plot_loadings(model.components_, names, dirs["graphs"] / "pca_loadings.png")
    save_json({"algorithm": ctx.name, "standardize": cfg.STANDARDIZE, "channels": names, "n_channels": len(names),
               "split": ctx.split, "n_training_pixels_total": int(x_all.shape[0]),
               "n_pixels_used_for_fit": int(x_fit.shape[0]), "explained_variance_ratio": evr,
               "cumulative_explained_variance": cum, "components_for_variance": k_for, "sklearn_check": check,
               "timing_memory": stats, "process_rss_mb_end": rss_mb(), "model_file": model_path},
              dirs["metrics"] / "train_summary.json")
    ctx.info.update({"components_for_variance": k_for, "explained_variance_ratio": evr.tolist()})


def load_model(ctx):
    d = np.load(ctx.dirs["models"] / "pca_model.npz", allow_pickle=False)
    if [str(n) for n in d["channel_names"]] != ctx.names or str(d["dataset_id"]) != ctx.dataset.dataset_id:
        raise ValueError("pca_model.npz was trained on a different dataset")
    model = PCA.from_state({k[4:]: d[k] for k in d.files if k.startswith("pca_")})
    scaler = Standardizer.from_state({"enabled": bool(d["scaler_enabled"]), "mean": d["scaler_mean"],
                                      "std": d["scaler_std"]})
    return model, scaler


def test(ctx):
    import time
    t0 = time.perf_counter()
    model, scaler = load_model(ctx)
    C = model.n_features_
    ks = cfg.COMPONENT_COUNTS or sorted(set(ctx.ks) | {C})
    ks = [k for k in ks if 1 <= k <= C]
    n_extra = 2 * C if scaler.enabled else 0          # standardiser mean + std also travel with the model
    cum = np.cumsum(model.all_explained_variance_ratio_)
    cache = {}

    def forward(k, x):
        if cache.get("x") is not x:                   # standardise each test sample once
            cache["x"], cache["z"] = x, scaler.transform(x)
        return model.transform(cache["z"], k)

    spec = EvalSpec(ks=ks, kind="components", method_label=f"PCA ({ctx.name})",
                    forward=forward, inverse=lambda k, z: scaler.inverse_transform(model.inverse_transform(z)),
                    n_model_values=lambda k: model.n_model_values(k) + n_extra,
                    gallery_k=cfg.GALLERY_K, timing_repeats=cfg.TIMING_REPEATS,
                    save_reconstructed=cfg.SAVE_RECONSTRUCTED_K,
                    row_extra=lambda k, x, z, rec: {"cumulative_explained_variance_train": float(cum[k - 1])},
                    quality_extra=("cumulative_explained_variance_train", "Cumulative explained variance"))
    df, _, _ = evaluate(ctx, spec)
    write_test_summary(ctx, df, t_start=t0)
    return df
