"""GA channel selection plugged into the common pipeline (train on training pixels, test with saved subsets)."""
from algorithms.ga import config as cfg
from algorithms.ga.implementation import ChannelSubsetGA
from common.band_selection import SearchSpec, evaluate_selection, run_search

NAME, DISPLAY, FAMILY, REQUIRES = "ga", "GA", "band selection (evolutionary search)", []


def output_name():
    return NAME


def hyperparameters():
    keys = ["POPULATION_SIZE", "MAX_GENERATIONS", "PATIENCE", "TOURNAMENT_SIZE", "CROSSOVER_RATE",
            "MUTATION_RATE", "ELITISM", "GA_SEEDS", "W_MIN_R2", "W_REDUNDANCY", "MAX_TRAIN_PIXELS",
            "VALIDATION_FRACTION", "RANDOM_SEED"]
    return {k.lower(): getattr(cfg, k) for k in keys}


def train(ctx):
    spec = SearchSpec(
        label="ga", title="GA", seeds=cfg.GA_SEEDS, hyperparameters=hyperparameters(),
        build=lambda C, k, f, corr, seed: ChannelSubsetGA(C, k, f, cfg.POPULATION_SIZE, cfg.MAX_GENERATIONS,
                                                          cfg.PATIENCE, cfg.TOURNAMENT_SIZE, cfg.CROSSOVER_RATE,
                                                          cfg.MUTATION_RATE, cfg.ELITISM, seed),
        steps_attr="generations_", steps_col="generations",
        conv_row=lambda h, names: {**{c: v for c, v in h.items() if c != "best_selected_idx"},
                                   "best_selected": "+".join(names[i] for i in h["best_selected_idx"])},
        convergence={"title": f"GA convergence: best fitness per generation ({ctx.dataset.name})"})
    run_search(ctx, cfg, spec)


def test(ctx):
    return evaluate_selection(ctx, cfg, "ga", "GA channel selection")
