"""ACO channel selection plugged into the common pipeline (same fitness as GA/PSO, different search)."""
from algorithms.aco import config as cfg
from algorithms.aco.implementation import ChannelSubsetACO
from common import visualization as viz
from common.band_selection import SearchSpec, evaluate_selection, run_search

NAME, DISPLAY, FAMILY, REQUIRES = "aco", "ACO", "band selection (ant colony search)", []


def output_name():
    return NAME


def hyperparameters():
    keys = ["N_ANTS", "MAX_ITERATIONS", "PATIENCE", "ALPHA", "BETA", "RHO", "Q", "TAU_INIT", "TAU_MIN", "TAU_MAX",
            "HEURISTIC_EPS", "ACO_SEEDS", "W_MIN_R2", "W_REDUNDANCY", "MAX_TRAIN_PIXELS", "VALIDATION_FRACTION",
            "RANDOM_SEED"]
    return {k.lower(): getattr(cfg, k) for k in keys}


def _conv_row(h, names):
    row = {c: v for c, v in h.items() if c not in ("gbest_selected_idx", "pheromone")}
    if len(names) <= 64:                                    # one column per channel (as in the original)
        row.update({f"tau_{names[j]}": tau for j, tau in enumerate(h["pheromone"])})
    else:
        row["pheromone"] = " ".join(f"{t:.4g}" for t in h["pheromone"])
    row["gbest_selected"] = "+".join(names[i] for i in h["gbest_selected_idx"])
    return row


def _pheromone_graph(ctx, runs, ks):
    seed = cfg.ACO_SEEDS[0]
    viz.plot_pheromone({k: runs[k][seed] for k in ks}, ctx.names, ctx.dirs["graphs"] / "aco_pheromone.png",
                       f"Pheromone per channel over iterations (seed {seed})")


def train(ctx):
    spec = SearchSpec(
        label="aco", title="ACO", seeds=cfg.ACO_SEEDS, hyperparameters=hyperparameters(),
        build=lambda C, k, f, corr, seed: ChannelSubsetACO(C, k, f, corr, cfg.N_ANTS, cfg.MAX_ITERATIONS, cfg.PATIENCE,
                                                           cfg.ALPHA, cfg.BETA, cfg.RHO, cfg.Q, cfg.TAU_INIT,
                                                           cfg.TAU_MIN, cfg.TAU_MAX, cfg.HEURISTIC_EPS, seed),
        steps_attr="iterations_", steps_col="iterations", conv_row=_conv_row, extra_graphs=_pheromone_graph,
        convergence={"title": f"ACO convergence: best-so-far fitness per iteration ({ctx.dataset.name})",
                     "x_key": "iteration", "best_key": "gbest", "mean_key": "iter_mean", "unit": "Iteration",
                     "mean_label": "colony mean (first seed)"})
    run_search(ctx, cfg, spec)


def test(ctx):
    return evaluate_selection(ctx, cfg, "aco", "ACO channel selection", compare_others=("ga", "pso"))
