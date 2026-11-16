"""
Training and evaluation code shared by the band-SELECTION algorithms (GA, PSO, ACO, SA and UBS-CNN).

In the original repository the four search methods had four copies of the same train.py / test.py that
differed only in the search object and a few names. The common parts live here; each algorithm's
pipeline.py supplies only what is specific to it (its search class, hyper-parameters, extra columns/graphs).
Nothing about the methodology changed:
  * the training matrix is a seeded sample of MAX_TRAIN_PIXELS valid TRAINING pixels, split into a "fit"
    part (reconstruction models) and a "validation" part (scoring) - test pixels are never used
  * every K is searched with several independent seeds; the best run is kept
  * optionally every C-choose-K subset is scored to check the search found the optimum (now only when the
    search space is small enough, see config.EXHAUSTIVE_MAX_SUBSETS)
  * the final LinearReconstructor of the chosen subset is fitted on all sampled training pixels and saved
  * test loads the saved subsets + weights; nothing is re-selected or re-fitted
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from common import config as gcfg
from common import visualization as viz
from common.data_loader import load_split_pixels
from common.preprocessing import sample_rows
from common.reduction_eval import EvalSpec, compare_with_pca, compare_with_searches, evaluate, write_test_summary
from common.utils import comb_safe, measure, rss_mb, save_json, set_seed


def training_matrices(ctx, max_train_pixels, validation_fraction, seed):
    """TRAINING pixels -> (x_train, x_s sample, x_fit, x_val, corr, near-duplicate pairs). Writes inspection."""
    rng = set_seed(seed)
    stats = {}
    with measure(stats, "load_and_preprocess"):
        x_train, names, inspection = load_split_pixels(ctx.dataset, ctx.split["train"], ctx.fraction, ctx.log, rng)
    save_json(inspection, ctx.dirs["metrics"] / "dataset_inspection.json")
    x_s = sample_rows(x_train, max_train_pixels, rng).astype(np.float32)
    n_val = int(round(len(x_s) * validation_fraction))
    if len(x_s) - n_val < 2 or n_val < 2:
        raise ValueError(f"Only {len(x_s)} valid training pixels: too few to fit and validate a reconstruction model")
    perm = rng.permutation(len(x_s))
    x_val, x_fit = x_s[perm[:n_val]], x_s[perm[n_val:]]
    C = len(names)
    ctx.log.info(f"Training matrix: {x_train.shape[0]:,} pixels x {C} channels; sample {len(x_s):,} -> "
                 f"fit {len(x_fit):,} / validation {len(x_val):,}")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(x_fit.T))
    np.fill_diagonal(corr, 1.0)
    viz.plot_correlation(corr, names, ctx.dirs["graphs"] / "channel_correlation.png")
    dup = [(names[i], names[j], round(float(corr[i, j]), 6)) for i in range(C) for j in range(i + 1, C)
           if abs(corr[i, j]) > 0.999]
    if dup:
        ctx.log.info(f"Near-duplicate channel pairs (|r| > 0.999): {dup[:20]}{' ...' if len(dup) > 20 else ''}. "
                     "Keeping both members of a pair adds no information; the redundancy term penalises it.")
    return x_train, x_s, x_fit, x_val, corr, dup, stats, rng


def exhaustive_allowed(C, k, enabled):
    return bool(enabled) and comb_safe(C, k) <= gcfg.EXHAUSTIVE_MAX_SUBSETS


@dataclass
class SearchSpec:
    label: str                                  # "ga", "pso", "aco", "sa"
    title: str                                  # "GA"
    seeds: list
    build: Callable                             # (C, k, fitness, corr, seed) -> searcher with .run()
    steps_attr: str                             # "generations_" / "iterations_"
    steps_col: str                              # "generations" / "iterations"
    conv_row: Callable                          # (history_entry, names) -> dict
    hyperparameters: dict
    result_extra: Callable = None               # searcher -> dict of extra columns for <label>_runs_K*.csv
    log_extra: Callable = None                  # searcher -> str
    convergence: dict = field(default_factory=dict)   # kwargs for viz.plot_ga_convergence
    extra_graphs: Callable = None               # (ctx, runs, ks) -> None


def run_search(ctx, cfg, spec):
    """Training for GA / PSO / ACO / SA (identical procedure, different search)."""
    from algorithms.ga.implementation import LinearReconstructor, SubsetFitness, exhaustive_search
    log, dirs, names = ctx.log, ctx.dirs, ctx.names
    log.info(f"{spec.title} hyper-parameters {spec.hyperparameters}")
    x_train, x_s, x_fit, x_val, corr, dup, stats, _ = training_matrices(
        ctx, cfg.MAX_TRAIN_PIXELS, cfg.VALIDATION_FRACTION, cfg.RANDOM_SEED)
    C = len(names)
    fitness = SubsetFitness(x_fit, x_val, names, cfg.W_MIN_R2, cfg.W_REDUNDANCY)
    ks = [k for k in (cfg.FEATURE_COUNTS or ctx.ks) if 1 <= k < C]
    sel_rows, conv_rows, check_rows, runs, optimum, selected = [], [], [], {}, {}, {}
    lab = spec.label
    for k in ks:
        runs[k], results = {}, []
        for seed in spec.seeds:
            st = {}
            # each seed gets a FRESH fitness cache so evaluation counts and timings are honest
            f_run = SubsetFitness(x_fit, x_val, names, cfg.W_MIN_R2, cfg.W_REDUNDANCY)
            with measure(st, "search"):
                s = spec.build(C, k, f_run, corr, seed).run()
            runs[k][seed] = s.history_
            for h in s.history_:
                conv_rows.append({"k": k, "seed": seed, **spec.conv_row(h, names)})
            d = f_run.details(s.best_)
            res = {"k": k, "seed": seed, "fitness": d["fitness"], "selected": "+".join(d["selected"]),
                   "selected_idx": d["selected_idx"], spec.steps_col: getattr(s, spec.steps_attr),
                   "stop_reason": s.stop_reason_, **(spec.result_extra(s) if spec.result_extra else {}),
                   "evaluations": s.n_evaluations_, "search_space": s.search_space,
                   "seconds": st["search"]["seconds"], "peak_python_mb": st["search"]["peak_python_mb"]}
            results.append(res)
            log.info(f"K={k} seed {seed}: best {d['selected'][:8]}{'...' if len(d['selected']) > 8 else ''} "
                     f"fitness {d['fitness']:.5f} after {getattr(s, spec.steps_attr)} {spec.steps_col}"
                     f"{spec.log_extra(s) if spec.log_extra else ''} ({s.stop_reason_}); "
                     f"{s.n_evaluations_}/{s.search_space} subsets evaluated, {st['search']['seconds']:.2f} s, "
                     f"peak {st['search']['peak_python_mb']:.1f} MB")
        best = max(results, key=lambda r: r["fitness"])
        selected[k] = best["selected_idx"]
        d = fitness.details(np.isin(np.arange(C), selected[k]).astype(np.int8))
        # duplicated channels create subsets with IDENTICAL fitness, so success is judged by fitness, not names
        agree = sum(abs(r["fitness"] - best["fitness"]) < 1e-9 for r in results)

        check = {}
        if exhaustive_allowed(C, k, cfg.RUN_EXHAUSTIVE_CHECK):
            st = {}
            with measure(st, "exh"):
                ranking = exhaustive_search(fitness, C, k)
            optimum[k] = ranking[0]["fitness"]
            ties = [r for r in ranking if abs(r["fitness"] - optimum[k]) < 1e-9]
            worse = [r for r in ranking if r["fitness"] < optimum[k] - 1e-9]
            rank = 1 + sum(r["fitness"] > d["fitness"] + 1e-9 for r in ranking)   # ties share a rank
            seeds_opt = sum(abs(r["fitness"] - optimum[k]) < 1e-9 for r in results)
            check = {"exhaustive_best": " | ".join("+".join(r["selected"]) for r in ties[:20]),
                     "n_tied_optimal_subsets": len(ties), "exhaustive_best_fitness": optimum[k],
                     f"{lab}_rank_among_all_subsets": rank, f"{lab}_found_optimum": rank == 1,
                     "seeds_reaching_optimum": f"{seeds_opt}/{len(results)}",
                     "best_non_optimal": "+".join(worse[0]["selected"]) if worse else "",
                     "best_non_optimal_fitness": worse[0]["fitness"] if worse else np.nan,
                     "exhaustive_seconds": st["exh"]["seconds"]}
            pd.DataFrame([{**{c: v for c, v in r.items() if c not in ("per_channel_R2", "selected_idx")},
                           "selected": "+".join(r["selected"])} for r in ranking]) \
                .to_csv(dirs["metrics"] / f"exhaustive_ranking_K{k}.csv", index=False)
            log.info(f"K={k} exhaustive check: optimum fitness {optimum[k]:.5f} shared by {len(ties)} subset(s); "
                     f"{spec.title} subset ranks #{rank} of {len(ranking)}; seeds reaching the optimum "
                     f"{check['seeds_reaching_optimum']}")
        else:
            check = {"exhaustive_best": f"not run ({comb_safe(C, k):,} subsets > EXHAUSTIVE_MAX_SUBSETS="
                                        f"{gcfg.EXHAUSTIVE_MAX_SUBSETS})" if cfg.RUN_EXHAUSTIVE_CHECK else "disabled"}
        check_rows.append({"k": k, f"{lab}_best": best["selected"], f"{lab}_best_fitness": best["fitness"],
                           "seeds_agreeing": f"{agree}/{len(results)}",
                           "mean_evaluations": float(np.mean([r["evaluations"] for r in results])),
                           "search_space": best["search_space"],
                           f"mean_{lab}_seconds": float(np.mean([r["seconds"] for r in results])), **check})
        for r in results:
            r["selected_idx"] = " ".join(map(str, r["selected_idx"]))

        rec = LinearReconstructor(selected[k]).fit(x_s)
        save_selection_model(ctx, f"{lab}_K{k}.npz", rec, d)
        sel_rows.append(selection_row(k, C, names, d))
        pd.DataFrame(results).to_csv(dirs["metrics"] / f"{lab}_runs_K{k}.csv", index=False)
        log.info(f"K={k} SELECTED {d['selected'][:10]}{'...' if k > 10 else ''} (seeds agreeing {agree}/{len(results)}) "
                 f"| mean R2 {d['mean_R2']:.4f}, min R2 {d['min_R2']:.4f} ({d['worst_channel']}), "
                 f"redundancy {d['redundancy']:.3f}")

    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(dirs["metrics"] / "selected_features.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(dirs["metrics"] / "convergence.csv", index=False)
    pd.DataFrame(check_rows).to_csv(dirs["metrics"] / f"{lab}_vs_exhaustive.csv", index=False)

    viz.plot_ga_convergence(runs, optimum, dirs["graphs"] / f"{lab}_convergence.png", **spec.convergence)
    viz.plot_selection_matrix(selected, names, ks, dirs["graphs"] / "selected_channels.png",
                              f"Channels kept by {spec.title} at each K")
    fitness_graph(ctx, sel_df, "fitness")
    if spec.extra_graphs:
        spec.extra_graphs(ctx, runs, ks)
    summary = {"algorithm": ctx.name, "hyperparameters": spec.hyperparameters, "channels": names,
               "split": ctx.split, "n_training_pixels_total": int(x_train.shape[0]),
               "n_fit_pixels": int(len(x_fit)), "n_validation_pixels": int(len(x_val)),
               "near_duplicate_channels": dup, "feature_counts": ks,
               "selected": {k: [names[i] for i in v] for k, v in selected.items()},
               f"{lab}_vs_exhaustive": check_rows, "timing_memory": stats, "process_rss_mb_end": rss_mb()}
    save_json(summary, dirs["metrics"] / "train_summary.json")
    ctx.info["selected_channels"] = summary["selected"]
    return summary


def selection_row(k, C, names, d, objective_col="fitness"):
    return {"k": k, "n_channels": C, "selected": "+".join(d["selected"]),
            "selected_idx": " ".join(map(str, d["selected_idx"])),
            "dropped": "+".join(n for n in names if n not in d["selected"]),
            objective_col: d["fitness"], "mean_R2_val": d["mean_R2"], "min_R2_val": d["min_R2"],
            "worst_channel": d["worst_channel"], "redundancy": d["redundancy"], "reduction_pct": 100 * (1 - k / C),
            **{f"R2_{n}": v for n, v in d["per_channel_R2"].items()}}


def fitness_graph(ctx, sel_df, col):
    t = sel_df.assign(sample_name="train (validation pixels)")
    viz.plot_metric_panels(t, [(col, "Fitness" if col == "fitness" else "Objective"),
                               ("mean_R2_val", "Mean R² (information kept)"),
                               ("min_R2_val", "Worst-channel R²"), ("redundancy", "Redundancy (mean |r|)")],
                           ctx.dirs["graphs"] / "fitness_terms_vs_k.png", "Fitness and its terms vs K (training)",
                           series_col="sample_name", x_label="Selected channels (K)")


def save_selection_model(ctx, filename, rec, d, **extra):
    np.savez(ctx.dirs["models"] / filename, selected=rec.selected, coef=rec.coef_,
             channel_names=np.array(ctx.names), selected_names=np.array(d["selected"]), fitness=d["fitness"],
             dataset_id=ctx.dataset.dataset_id, **extra)


def load_selection_models(ctx, prefix):
    from algorithms.ga.implementation import LinearReconstructor
    files = sorted(ctx.dirs["models"].glob(f"{prefix}_K*.npz"), key=lambda p: int(p.stem.split("K")[-1]))
    if not files:
        raise FileNotFoundError(f"No saved models {prefix}_K*.npz in {ctx.dirs['models']} - training did not finish")
    models, fitness = {}, {}
    for p in files:
        d = np.load(p, allow_pickle=False)
        if [str(n) for n in d["channel_names"]] != ctx.names or str(d["dataset_id"]) != ctx.dataset.dataset_id:
            raise ValueError(f"{p.name} was trained on a different dataset/channel set")
        m = LinearReconstructor.from_state({"selected": d["selected"], "coef": d["coef"]})
        models[len(m.selected)], fitness[len(m.selected)] = m, float(d["fitness"])
    return models, fitness


def evaluate_selection(ctx, cfg, prefix, title, compare_others=(), per_sample=None, objective_label=None):
    """Test step of every band-selection method (loads the saved subsets; nothing is re-fitted)."""
    import time
    t0 = time.perf_counter()
    models, fitness = load_selection_models(ctx, prefix)
    ks = sorted(models)
    names = ctx.names
    ctx.log.info(f"Loaded saved subsets for K = {ks}")
    for k in ks:
        sel = [names[i] for i in models[k].selected]
        ctx.log.info(f"  K={k}: keep {sel[:12]}{' ...' if k > 12 else ''} (training fitness {fitness[k]:.5f})")

    def row_extra(k, x, z, rec):
        m = models[k]
        kept_err = float(np.abs(rec[:, m.selected] - x[:, m.selected]).max())  # kept channels come back exactly
        return {"selected": "+".join(names[i] for i in m.selected), "train_fitness": fitness[k],
                "max_abs_error_on_kept_channels": kept_err}

    spec = EvalSpec(ks=ks, kind="features", method_label=title, reduced_dtype=cfg.REDUCED_DTYPE,
                    gallery_k=cfg.GALLERY_K, timing_repeats=cfg.TIMING_REPEATS,
                    forward=lambda k, x: models[k].reduce(x.astype(np.float32)),
                    inverse=lambda k, z: models[k].reconstruct_from_reduced(z),
                    n_model_values=lambda k: models[k].n_model_values(), row_extra=row_extra,
                    per_sample=per_sample, lossless_tol=1e-12,
                    native_selection=lambda k: models[k].selected.tolist(),
                    reduced_image_labels=lambda k: [names[i] for i in models[k].selected],
                    quality_extra=("train_fitness", objective_label or f"{title} fitness (training)"))
    df, gdf, cdf = evaluate(ctx, spec)
    comparison = compare_with_pca(ctx, df, prefix)
    others = compare_with_searches(ctx, df, prefix, compare_others) if compare_others else None
    write_test_summary(ctx, df, {"selected": {int(k): [names[i] for i in models[k].selected] for k in ks},
                                 f"{prefix}_vs_pca": comparison, "search_comparison_file": others}, t0)
    return df
