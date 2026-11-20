"""PSO channel selection plugged into the common pipeline (same fitness as the GA, different search)."""
from algorithms.pso import config as cfg
from algorithms.pso.implementation import ChannelSubsetPSO
from common.band_selection import SearchSpec, evaluate_selection, run_search

NAME, DISPLAY, FAMILY, REQUIRES = "pso", "PSO", "band selection (swarm search)", []


def output_name():
    return NAME


def hyperparameters():
    keys = ["N_PARTICLES", "MAX_ITERATIONS", "PATIENCE", "W_START", "W_END", "C1", "C2", "V_MAX", "PSO_SEEDS",
            "W_MIN_R2", "W_REDUNDANCY", "MAX_TRAIN_PIXELS", "VALIDATION_FRACTION", "RANDOM_SEED"]
    return {k.lower(): getattr(cfg, k) for k in keys}


def train(ctx):
    spec = SearchSpec(
        label="pso", title="PSO", seeds=cfg.PSO_SEEDS, hyperparameters=hyperparameters(),
        build=lambda C, k, f, corr, seed: ChannelSubsetPSO(C, k, f, cfg.N_PARTICLES, cfg.MAX_ITERATIONS,
                                                           cfg.PATIENCE, cfg.W_START, cfg.W_END, cfg.C1, cfg.C2,
                                                           cfg.V_MAX, seed),
        steps_attr="iterations_", steps_col="iterations",
        conv_row=lambda h, names: {**{c: v for c, v in h.items() if c != "gbest_selected_idx"},
                                   "gbest_selected": "+".join(names[i] for i in h["gbest_selected_idx"])},
        convergence={"title": f"PSO convergence: global-best fitness per iteration ({ctx.dataset.name})",
                     "x_key": "iteration", "best_key": "gbest", "mean_key": "iter_mean", "unit": "Iteration",
                     "mean_label": "swarm mean (first seed)"})
    run_search(ctx, cfg, spec)


def test(ctx):
    return evaluate_selection(ctx, cfg, "pso", "PSO channel selection", compare_others=("ga",))
