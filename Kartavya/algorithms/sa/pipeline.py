"""SA channel selection plugged into the common pipeline (same objective as GA/PSO/ACO, different search)."""
from algorithms.sa import config as cfg
from algorithms.sa.implementation import ChannelSubsetSA
from common import visualization as viz
from common.band_selection import SearchSpec, evaluate_selection, run_search

NAME, DISPLAY, FAMILY, REQUIRES = "sa", "SA", "band selection (simulated annealing)", []


def output_name():
    return NAME


def hyperparameters():
    keys = ["MAX_ITERATIONS", "COOLING_RATE", "INITIAL_ACCEPT_PROB", "T0_SAMPLES", "T_MIN_RATIO", "PATIENCE",
            "SA_SEEDS", "W_MIN_R2", "W_REDUNDANCY", "MAX_TRAIN_PIXELS", "VALIDATION_FRACTION", "RANDOM_SEED"]
    return {k.lower(): getattr(cfg, k) for k in keys}


def _conv_row(h, names):
    return {**{c: v for c, v in h.items() if not c.endswith("_selected_idx")},
            "current_selected": "+".join(names[i] for i in h["current_selected_idx"]),
            "best_selected": "+".join(names[i] for i in h["best_selected_idx"])}


def _temperature_graph(ctx, runs, ks):
    seed = cfg.SA_SEEDS[0]
    viz.plot_sa_temperature({k: runs[k][seed] for k in ks}, ctx.dirs["graphs"] / "sa_temperature.png",
                            f"Temperature and acceptance during annealing (seed {seed})")


def train(ctx):
    spec = SearchSpec(
        label="sa", title="SA", seeds=cfg.SA_SEEDS, hyperparameters=hyperparameters(),
        build=lambda C, k, f, corr, seed: ChannelSubsetSA(C, k, f, cfg.MAX_ITERATIONS, cfg.COOLING_RATE,
                                                          cfg.INITIAL_ACCEPT_PROB, cfg.T0_SAMPLES, cfg.T_MIN_RATIO,
                                                          cfg.PATIENCE, seed),
        steps_attr="iterations_", steps_col="iterations", conv_row=_conv_row, extra_graphs=_temperature_graph,
        result_extra=lambda s: {"T0": s.t0_, "mean_worsening_delta_for_T0": s.mean_worse_delta_,
                                "final_temperature": s.final_temperature_, "acceptance_rate": s.acceptance_rate_},
        log_extra=lambda s: f" (T0 {s.t0_:.2e}, final T {s.final_temperature_:.2e}, acceptance {s.acceptance_rate_:.2f})",
        convergence={"title": f"SA convergence: best objective per move ({ctx.dataset.name})", "x_key": "iteration",
                     "best_key": "best", "mean_key": "current", "unit": "Move",
                     "mean_label": "current solution (first seed)"})
    run_search(ctx, cfg, spec)


def test(ctx):
    return evaluate_selection(ctx, cfg, "sa", "SA channel selection", compare_others=("ga", "pso", "aco"))
