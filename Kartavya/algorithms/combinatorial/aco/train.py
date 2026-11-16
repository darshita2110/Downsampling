"""
Run Ant Colony Optimization on TRAINING pixels only and save the best channel subset for every K.

Run from the repository root:
    python Kartavya/algorithms/combinatorial/aco/train.py                 # PNG products (8 channels)
    python Kartavya/algorithms/combinatorial/aco/train.py --dataset raw   # original bands, if present

What it does
  1. lists images in sorted-filename order and saves the filename -> index mapping
  2. inspects every image (size, mode, dtype, range, map panel, colour-map decoding, alignment)
  3. builds the training matrix (valid TRAINING pixels x channels, normalised to [0, 1]), samples it
     and splits the sample into a "fit" part (reconstruction models) and a "validation" part (scoring)
  4. for every K: runs ACO with several colony seeds, keeps the best subset, and (optionally) scores every
     possible subset to check that ACO really found the optimum
  5. fits the final reconstruction model of the chosen subset on all sampled training pixels
  6. saves models, selected features, convergence tables, graphs and a log. Test data is never touched.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd

from Kartavya.algorithms.combinatorial.aco import config as cfg
# fitness + reconstruction are shared with the GA and PSO so the two searches are compared on identical terms
from Kartavya.algorithms.combinatorial.ga.ga import LinearReconstructor, SubsetFitness, exhaustive_search
from Kartavya.algorithms.combinatorial.aco.aco import ChannelSubsetACO
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.preprocessing import sample_rows
from Kartavya.common.utils import ensure_dirs, get_logger, measure, rss_mb, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train ACO channel selection")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--tag", default=None, help="experiment name suffix")
    return p.parse_args()


def hyperparameters():
    keys = ["N_ANTS", "MAX_ITERATIONS", "PATIENCE", "ALPHA", "BETA", "RHO", "Q", "TAU_INIT", "TAU_MIN", "TAU_MAX",
            "HEURISTIC_EPS", "ACO_SEEDS", "W_MIN_R2", "W_REDUNDANCY", "MAX_TRAIN_PIXELS",
            "VALIDATION_FRACTION", "RANDOM_SEED", "TARGET_WIDTH"]
    return {k.lower(): getattr(cfg, k) for k in keys}


def main():
    args = parse_args()
    exp = args.dataset + (f"_{args.tag}" if args.tag else "")
    dirs = cfg.output_dirs(exp)
    ensure_dirs(*dirs.values())
    log = get_logger(f"aco.train.{exp}", dirs["logs"] / "train.log")
    rng = set_seed(cfg.RANDOM_SEED)
    t_start = time.perf_counter()
    hp = hyperparameters()
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
        log.warning("Only ONE date is available: selecting on the top half of the image, testing on the bottom "
                    "half. With all five dates, earlier dates train and later dates test.")

    # ---------------------------------------------------------- 2. TRAINING pixels only
    stats = {}
    with measure(stats, "load_and_preprocess"):
        x_train, names, inspection = dl.load_split_pixels(args.dataset, mapping, split["train"], cfg,
                                                          cfg.SPATIAL_TRAIN_FRACTION, log)
    save_json(inspection, dirs["metrics"] / "dataset_inspection.json")
    x_s = sample_rows(x_train, cfg.MAX_TRAIN_PIXELS, rng).astype(np.float32)
    n_val = int(round(len(x_s) * cfg.VALIDATION_FRACTION))
    perm = rng.permutation(len(x_s))
    x_val, x_fit = x_s[perm[:n_val]], x_s[perm[n_val:]]
    C = len(names)
    log.info(f"Training matrix: {x_train.shape[0]:,} pixels x {C} channels; sample {len(x_s):,} -> "
             f"fit {len(x_fit):,} / validation {len(x_val):,}. Genes (channels): {names}")
    corr = np.corrcoef(x_fit.T)
    viz.plot_correlation(corr, names, dirs["graphs"] / "channel_correlation.png")
    dup = [(names[i], names[j], round(float(corr[i, j]), 6)) for i in range(C) for j in range(i + 1, C)
           if abs(corr[i, j]) > 0.999]
    if dup:
        log.info(f"Near-duplicate channel pairs (|r| > 0.999): {dup}. Keeping both members of a pair adds "
                 "no information; the redundancy term penalises it.")

    fitness = SubsetFitness(x_fit, x_val, names, cfg.W_MIN_R2, cfg.W_REDUNDANCY)
    ks = cfg.FEATURE_COUNTS or list(range(1, C))

    # ---------------------------------------------------------- 3. ACO per K (several seeds)
    sel_rows, conv_rows, check_rows, runs, optimum, selected = [], [], [], {}, {}, {}
    for k in ks:
        runs[k], results = {}, []
        for seed in cfg.ACO_SEEDS:
            st = {}
            # each seed gets a FRESH fitness cache so evaluation counts and timings are honest
            f_run = SubsetFitness(x_fit, x_val, names, cfg.W_MIN_R2, cfg.W_REDUNDANCY)
            with measure(st, "aco"):
                colony = ChannelSubsetACO(C, k, f_run, corr, cfg.N_ANTS, cfg.MAX_ITERATIONS, cfg.PATIENCE,
                                         cfg.ALPHA, cfg.BETA, cfg.RHO, cfg.Q, cfg.TAU_INIT, cfg.TAU_MIN,
                                         cfg.TAU_MAX, cfg.HEURISTIC_EPS, seed).run()
            runs[k][seed] = colony.history_
            for h in colony.history_:
                conv_rows.append({"k": k, "seed": seed, **{c: v for c, v in h.items() if c != ("gbest_selected_idx", "pheromone")},
                                  **{f"tau_{names[j]}": tau for j, tau in enumerate(h["pheromone"])},
                                  "gbest_selected": "+".join(names[i] for i in h["gbest_selected_idx"])})
            d = f_run.details(colony.best_)
            results.append({"k": k, "seed": seed, "fitness": d["fitness"], "selected": "+".join(d["selected"]),
                            "selected_idx": d["selected_idx"], "iterations": colony.iterations_,
                            "stop_reason": colony.stop_reason_, "evaluations": colony.n_evaluations_,
                            "search_space": colony.search_space, "seconds": st["aco"]["seconds"],
                            "peak_python_mb": st["aco"]["peak_python_mb"]})
            log.info(f"K={k} seed {seed}: best {d['selected']} fitness {d['fitness']:.5f} after "
                     f"{colony.iterations_} iterations ({colony.stop_reason_}); {colony.n_evaluations_}/{colony.search_space} "
                     f"subsets evaluated, {st['aco']['seconds']:.2f} s, peak {st['aco']['peak_python_mb']:.1f} MB")
        best = max(results, key=lambda r: r["fitness"])
        selected[k] = best["selected_idx"]
        d = fitness.details(np.isin(np.arange(C), selected[k]).astype(np.int8))
        # duplicated channels (B04 and B03 appear in both RGB and False_Color) create subsets with IDENTICAL
        # fitness, so success is judged by fitness, not by channel names
        agree = sum(abs(r["fitness"] - best["fitness"]) < 1e-9 for r in results)

        # ---------------------------------------------- 4. verification against exhaustive search
        check = {}
        if cfg.RUN_EXHAUSTIVE_CHECK:
            st = {}
            with measure(st, "exh"):
                ranking = exhaustive_search(fitness, C, k)
            optimum[k] = ranking[0]["fitness"]
            ties = [r for r in ranking if abs(r["fitness"] - optimum[k]) < 1e-9]
            worse = [r for r in ranking if r["fitness"] < optimum[k] - 1e-9]
            rank_of_ga = 1 + sum(r["fitness"] > d["fitness"] + 1e-9 for r in ranking)   # ties share a rank
            seeds_opt = sum(abs(r["fitness"] - optimum[k]) < 1e-9 for r in results)
            check = {"exhaustive_best": " | ".join("+".join(r["selected"]) for r in ties),
                     "n_tied_optimal_subsets": len(ties), "exhaustive_best_fitness": optimum[k],
                     "aco_rank_among_all_subsets": rank_of_ga, "aco_found_optimum": rank_of_ga == 1,
                     "seeds_reaching_optimum": f"{seeds_opt}/{len(results)}",
                     "best_non_optimal": "+".join(worse[0]["selected"]) if worse else "",
                     "best_non_optimal_fitness": worse[0]["fitness"] if worse else np.nan,
                     "exhaustive_seconds": st["exh"]["seconds"]}
            pd.DataFrame([{**{c: v for c, v in r.items() if c not in ("per_channel_R2", "selected_idx")},
                           "selected": "+".join(r["selected"])} for r in ranking]) \
                .to_csv(dirs["metrics"] / f"exhaustive_ranking_K{k}.csv", index=False)
            log.info(f"K={k} exhaustive check: optimum fitness {optimum[k]:.5f} shared by {len(ties)} subset(s) "
                     f"[{check['exhaustive_best']}]; ACO subset ranks #{rank_of_ga} of {len(ranking)}; "
                     f"seeds reaching the optimum {check['seeds_reaching_optimum']}")
        check_rows.append({"k": k, "aco_best": best["selected"], "aco_best_fitness": best["fitness"],
                           "seeds_agreeing": f"{agree}/{len(results)}",
                           "mean_evaluations": float(np.mean([r["evaluations"] for r in results])),
                           "search_space": best["search_space"],
                           "mean_aco_seconds": float(np.mean([r["seconds"] for r in results])), **check})
        for r in results:
            r["selected_idx"] = " ".join(map(str, r["selected_idx"]))

        # ---------------------------------------------- 5. final reconstruction model for this K
        rec = LinearReconstructor(selected[k]).fit(x_s)
        np.savez(dirs["models"] / f"aco_K{k}.npz", selected=rec.selected, coef=rec.coef_,
                 channel_names=np.array(names), selected_names=np.array(d["selected"]),
                 fitness=d["fitness"], experiment=exp, dataset=args.dataset, target_width=cfg.TARGET_WIDTH)
        sel_rows.append({"k": k, "n_channels": C, "selected": "+".join(d["selected"]),
                         "selected_idx": " ".join(map(str, d["selected_idx"])),
                         "dropped": "+".join(n for n in names if n not in d["selected"]),
                         "fitness": d["fitness"], "mean_R2_val": d["mean_R2"], "min_R2_val": d["min_R2"],
                         "worst_channel": d["worst_channel"], "redundancy": d["redundancy"],
                         "reduction_pct": 100 * (1 - k / C),
                         **{f"R2_{n}": v for n, v in d["per_channel_R2"].items()}})
        pd.DataFrame(results).to_csv(dirs["metrics"] / f"aco_runs_K{k}.csv", index=False)
        log.info(f"K={k} SELECTED {d['selected']} (seeds agreeing {agree}/{len(results)}) | mean R2 {d['mean_R2']:.4f}, "
                 f"min R2 {d['min_R2']:.4f} ({d['worst_channel']}), redundancy {d['redundancy']:.3f}")

    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(dirs["metrics"] / "selected_features.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(dirs["metrics"] / "convergence.csv", index=False)
    pd.DataFrame(check_rows).to_csv(dirs["metrics"] / "aco_vs_exhaustive.csv", index=False)

    # ---------------------------------------------------------- 6. graphs
    viz.plot_ga_convergence(runs, optimum, dirs["graphs"] / "aco_convergence.png",
                            f"ACO convergence: best-so-far fitness per iteration ({exp})", x_key="iteration", best_key="gbest", mean_key="iter_mean", unit="Iteration",
                            mean_label="colony mean (first seed)")
    # final pheromone per K (what the colony learned) and its evolution for the hardest K
    viz.plot_pheromone({k: runs[k][cfg.ACO_SEEDS[0]] for k in ks}, names, dirs["graphs"] / "aco_pheromone.png",
                       f"Pheromone per channel over iterations (seed {cfg.ACO_SEEDS[0]})")
    viz.plot_selection_matrix(selected, names, ks, dirs["graphs"] / "selected_channels.png",
                              "Channels kept by ACO at each K")
    t = sel_df.assign(date="train (validation pixels)")
    viz.plot_metric_panels(t, [("fitness", "Fitness"), ("mean_R2_val", "Mean R² (information kept)"),
                               ("min_R2_val", "Worst-channel R²"), ("redundancy", "Redundancy (mean |r|)")],
                           dirs["graphs"] / "fitness_terms_vs_k.png", "Fitness and its terms vs K (training)",
                           x_label="Selected channels (K)")

    summary = {"experiment": exp, "dataset": args.dataset, "hyperparameters": hp, "channels": names,
               "split": split, "n_training_pixels_total": int(x_train.shape[0]),
               "n_fit_pixels": int(len(x_fit)), "n_validation_pixels": int(len(x_val)),
               "near_duplicate_channels": dup, "feature_counts": ks,
               "selected": {k: [names[i] for i in v] for k, v in selected.items()},
               "aco_vs_exhaustive": check_rows, "timing_memory": stats, "process_rss_mb_end": rss_mb(),
               "total_seconds": time.perf_counter() - t_start}
    save_json(summary, dirs["metrics"] / "train_summary.json")
    log.info(f"Training finished in {summary['total_seconds']:.1f} s. Models: {dirs['models']}")


if __name__ == "__main__":
    main()
