"""
Train BS-Net-Conv (used as the UBS-CNN method) on TRAINING patches only, rank the channels by their
average attention weight, and save the top-K selection for every K.

Run from the repository root:
    python Kartavya/algorithms/combinatorial/ubscnn/train.py
    python Kartavya/algorithms/combinatorial/ubscnn/train.py --epochs 20     # quicker run
    python Kartavya/algorithms/combinatorial/ubscnn/train.py --dataset raw   # original bands, if present

What it does
  1. lists images in sorted-filename order, saves the filename -> index mapping, inspects every image
  2. cuts the TRAINING region into 16 x 16 x C patches (values in [0, 1]); 10 % are held out to monitor
     over-fitting (they are still training-region patches; the test region is never touched)
  3. trains BS-Net-Conv for every seed: Adam, MSE + 0.01 * L1(band weights), records loss per epoch and the
     average band weights after every epoch; saves the weights (.pt)
  4. averages the band weights over training patches and seeds -> channel ranking; top-K = selection
  5. for every K fits the linear model that rebuilds all channels from the K selected ones (training pixels),
     and scores the subset with the objective used by GA/PSO/ACO/SA (evaluation only) against the
     exhaustive optimum
  6. saves models, selected features, histories, graphs and a log
"""
import argparse
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import pandas as pd
import torch

from Kartavya.algorithms.combinatorial.ga.ga import LinearReconstructor, SubsetFitness, exhaustive_search
from Kartavya.algorithms.combinatorial.ubscnn import config as cfg
from Kartavya.algorithms.combinatorial.ubscnn.model import (BSNetConv, band_weights, bsnet_loss, count_parameters,
                                                            reconstruction_mse)
from Kartavya.common import data_loader as dl
from Kartavya.common import visualization as viz
from Kartavya.common.preprocessing import extract_patches, sample_rows
from Kartavya.common.utils import ensure_dirs, file_size_bytes, get_logger, measure, rss_mb, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train BS-Net-Conv (UBS-CNN) channel selection")
    p.add_argument("--dataset", choices=["png", "raw"], default="png")
    p.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS")
    p.add_argument("--tag", default=None, help="experiment name suffix")
    return p.parse_args()


def training_patches(experiment, mapping, split, log):
    """Patches from the TRAINING region of every training sample (all pixels valid)."""
    blocks = []
    for entry in split["train"]:
        data = dl.load_normalized_sample(experiment, mapping["samples"][entry["index"]], cfg)
        rs = dl.rows_slice(data["valid"].shape[0], entry["rows"], cfg.SPATIAL_TRAIN_FRACTION)
        p, _ = extract_patches(data["channels"], data["valid"], cfg.PATCH_SIZE, cfg.PATCH_STRIDE,
                               (rs.start, rs.stop))
        log.info(f"  {mapping['samples'][entry['index']]['date']}: rows {rs.start}-{rs.stop} -> {len(p):,} patches "
                 f"of {cfg.PATCH_SIZE}x{cfg.PATCH_SIZE}x{p.shape[1]} (stride {cfg.PATCH_STRIDE})")
        blocks.append(p)
    return np.concatenate(blocks)


def train_one(seed, x_tr, x_val, epochs, names, log):
    """Train one BS-Net-Conv. Returns (model, per-epoch history, per-epoch mean band weights)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BSNetConv(x_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    xt = torch.from_numpy(x_tr)
    xv = torch.from_numpy(x_val)
    history, weight_hist = [], []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        order = rng.permutation(len(xt))
        sums = np.zeros(3)
        n_batches = len(xt) // cfg.BATCH_SIZE                  # as in the authors' code (last partial batch dropped)
        for b in range(n_batches):
            xb = xt[order[b * cfg.BATCH_SIZE:(b + 1) * cfg.BATCH_SIZE]]
            opt.zero_grad()
            x_hat, w = model(xb)
            loss, mse, l1 = bsnet_loss(xb, x_hat, w, cfg.L1_LAMBDA)
            loss.backward()
            opt.step()
            sums += [loss.item(), mse.item(), l1.item()]
        w_all = band_weights(model, xt)                          # evaluation-mode weights on all training patches
        mean_w = w_all.mean(0)
        weight_hist.append(mean_w)
        rec = {"seed": seed, "epoch": ep, "train_loss": sums[0] / n_batches, "train_mse": sums[1] / n_batches,
               "train_l1_term": sums[2] / n_batches, "val_mse": reconstruction_mse(model, xv),
               "epoch_seconds": time.perf_counter() - t0,
               "ranking": " > ".join(names[i] for i in np.argsort(-mean_w)),
               **{f"w_{n}": float(v) for n, v in zip(names, mean_w)}}
        history.append(rec)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            log.info(f"  seed {seed} epoch {ep:3d}: loss {rec['train_loss']:.4f} (MSE {rec['train_mse']:.5f} + "
                     f"L1 {rec['train_l1_term']:.4f}), val MSE {rec['val_mse']:.5f}, {rec['epoch_seconds']:.1f} s | "
                     f"weights {np.round(mean_w, 3).tolist()}")
    return model, history, np.array(weight_hist)


def main():
    args = parse_args()
    epochs = args.epochs or cfg.EPOCHS
    exp = args.dataset + (f"_{args.tag}" if args.tag else "")
    dirs = cfg.output_dirs(exp)
    ensure_dirs(*dirs.values())
    log = get_logger(f"ubscnn.train.{exp}", dirs["logs"] / "train.log")
    rng = set_seed(cfg.RANDOM_SEED)
    if cfg.TORCH_THREADS:
        torch.set_num_threads(cfg.TORCH_THREADS)
    t_start = time.perf_counter()
    hp = {"method": "BS-Net-Conv (Cai et al., IEEE TGRS 2020, arXiv:1904.08269) used as UBS-CNN",
          "patch_size": cfg.PATCH_SIZE, "patch_stride": cfg.PATCH_STRIDE, "l1_lambda": cfg.L1_LAMBDA,
          "learning_rate": cfg.LEARNING_RATE, "batch_size": cfg.BATCH_SIZE, "epochs": epochs, "seeds": cfg.SEEDS,
          "optimizer": "Adam", "val_patch_fraction": cfg.VAL_PATCH_FRACTION, "torch": torch.__version__,
          "threads": torch.get_num_threads()}
    log.info(f"Experiment '{exp}' | {hp}")

    # ---------------------------------------------------------- 1. ordering + mapping + split + inspection
    mapping = dl.discover_samples(args.dataset, cfg, log)
    if not mapping["samples"]:
        where = cfg.RAW_DATASET_DIR if args.dataset == "raw" else cfg.PNG_DATASET_DIR
        log.error(f"No usable samples found in {where}."
                  + (" Original Sentinel-2 bands are not available, so the raw experiment is skipped "
                     "(no bands are fabricated)." if args.dataset == "raw" else ""))
        sys.exit(2)
    save_json(mapping, dirs["metrics"] / "filename_index_mapping.json")
    split = dl.make_split(len(mapping["samples"]), cfg)
    save_json(split, dirs["models"] / "split.json")
    log.info(f"Split mode: {split['mode']} | train {split['train']} | test {split['test']}")
    if split["mode"] == "spatial":
        log.warning("Only ONE date is available: training on the top half of the image, testing on the bottom half.")
    stats = {}
    with measure(stats, "load_pixels"):
        x_pix, names, inspection = dl.load_split_pixels(args.dataset, mapping, split["train"], cfg,
                                                        cfg.SPATIAL_TRAIN_FRACTION, log)
    save_json(inspection, dirs["metrics"] / "dataset_inspection.json")
    C = len(names)

    # ---------------------------------------------------------- 2. training patches
    with measure(stats, "make_patches"):
        patches = training_patches(args.dataset, mapping, split, log)
    perm = rng.permutation(len(patches))
    n_val = int(round(len(patches) * cfg.VAL_PATCH_FRACTION))
    x_val, x_tr = patches[perm[:n_val]], patches[perm[n_val:]]
    log.info(f"Patches: {len(patches):,} -> train {len(x_tr):,} / held-out {len(x_val):,}; "
             f"{BSNetConv(C).__class__.__name__} has {count_parameters(BSNetConv(C)):,} trainable parameters")

    # ---------------------------------------------------------- 3. train one network per seed
    histories, weight_hists, final_w, seed_rows = [], {}, {}, []
    for seed in cfg.SEEDS:
        st = {}
        with measure(st, "train"):
            model, hist, w_hist = train_one(seed, x_tr, x_val, epochs, names, log)
        path = dirs["models"] / f"bsnet_conv_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "n_bands": C, "channel_names": names,
                    "hyperparameters": hp}, path)
        histories += hist
        weight_hists[seed] = w_hist
        final_w[seed] = w_hist[-1]
        rank = np.argsort(-w_hist[-1])
        seed_rows.append({"seed": seed, "train_seconds": st["train"]["seconds"],
                          # tracemalloc does not see PyTorch's C++ allocations, so the process peak RSS is reported too
                          "train_peak_python_mb": st["train"]["peak_python_mb"], "rss_after_mb": st["train"]["rss_after_mb"],
                          "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          "final_train_mse": hist[-1]["train_mse"], "final_val_mse": hist[-1]["val_mse"],
                          "weights_file_bytes": file_size_bytes(path), "ranking": " > ".join(names[i] for i in rank),
                          **{f"w_{n}": float(v) for n, v in zip(names, w_hist[-1])}})
        log.info(f"seed {seed}: trained in {st['train']['seconds']:.0f} s (process peak RSS {seed_rows[-1]['process_peak_rss_mb']:.0f} MB); "
                 f"ranking {seed_rows[-1]['ranking']}")
    hist_df = pd.DataFrame(histories)
    hist_df.to_csv(dirs["metrics"] / "training_history.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(dirs["metrics"] / "seed_rankings.csv", index=False)

    # ---------------------------------------------------------- 4. consensus ranking (paper Eq. 11, averaged over seeds)
    mean_w = np.mean([final_w[s] for s in cfg.SEEDS], axis=0)
    ranking = np.argsort(-mean_w, kind="stable")
    top_sets = {s: set(np.argsort(-final_w[s])[:k]) for s in cfg.SEEDS for k in [3]}
    log.info(f"Consensus channel ranking (mean weight over {len(cfg.SEEDS)} seeds): "
             + ", ".join(f"{names[i]} {mean_w[i]:.3f}" for i in ranking))
    rank_df = pd.DataFrame({"rank": np.arange(1, C + 1), "channel": [names[i] for i in ranking],
                            "mean_weight": mean_w[ranking],
                            **{f"weight_seed{s}": final_w[s][ranking] for s in cfg.SEEDS}})
    rank_df.to_csv(dirs["metrics"] / "channel_ranking.csv", index=False)

    # ---------------------------------------------------------- 5. selections for every K + evaluation-only scoring
    x_s = sample_rows(x_pix, cfg.MAX_TRAIN_PIXELS, rng).astype(np.float32)
    n_v = int(round(len(x_s) * cfg.VALIDATION_FRACTION))
    p2 = rng.permutation(len(x_s))
    objective = SubsetFitness(x_s[p2[n_v:]], x_s[p2[:n_v]], names, cfg.W_MIN_R2, cfg.W_REDUNDANCY)
    ks = cfg.FEATURE_COUNTS or list(range(1, C))
    sel_rows, selected = [], {}
    for k in ks:
        sel = np.sort(ranking[:k])
        selected[k] = sel.tolist()
        mask = np.isin(np.arange(C), sel).astype(np.int8)
        d = objective.details(mask)
        row = {"k": k, "n_channels": C, "selected": "+".join(d["selected"]), "selected_idx": " ".join(map(str, sel)),
               "dropped": "+".join(n for n in names if n not in d["selected"]), "objective": d["fitness"],
               "mean_R2_val": d["mean_R2"], "min_R2_val": d["min_R2"], "worst_channel": d["worst_channel"],
               "redundancy": d["redundancy"], "reduction_pct": 100 * (1 - k / C),
               "seeds_with_same_topk": sum(set(np.argsort(-final_w[s])[:k]) == set(sel) for s in cfg.SEEDS)}
        if cfg.RUN_EXHAUSTIVE_CHECK:
            rk = exhaustive_search(objective, C, k)
            opt = rk[0]["fitness"]
            row.update({"exhaustive_optimum": opt, "optimum_subsets": " | ".join("+".join(r["selected"]) for r in rk
                                                                               if abs(r["fitness"] - opt) < 1e-9),
                        "rank_among_all_subsets": 1 + sum(r["fitness"] > d["fitness"] + 1e-9 for r in rk),
                        "n_subsets": len(rk), "gap_to_optimum": opt - d["fitness"]})
        rec = LinearReconstructor(sel).fit(x_s)
        np.savez(dirs["models"] / f"ubscnn_K{k}.npz", selected=rec.selected, coef=rec.coef_,
                 channel_names=np.array(names), selected_names=np.array(d["selected"]), fitness=d["fitness"],
                 mean_weight=mean_w, experiment=exp, dataset=args.dataset, target_width=cfg.TARGET_WIDTH)
        sel_rows.append(row)
        log.info(f"K={k}: select {d['selected']} | objective {d['fitness']:.5f}"
                 + (f" (optimum {row['exhaustive_optimum']:.5f}, rank {row['rank_among_all_subsets']}/{row['n_subsets']})"
                    if cfg.RUN_EXHAUSTIVE_CHECK else "")
                 + f" | same top-{k} in {row['seeds_with_same_topk']}/{len(cfg.SEEDS)} seeds")
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(dirs["metrics"] / "selected_features.csv", index=False)

    # ---------------------------------------------------------- 6. graphs
    viz.plot_training_curves(hist_df, dirs["graphs"] / "training_loss.png",
                             f"BS-Net-Conv training ({exp}): loss per epoch, one line per seed")
    viz.plot_weight_evolution(weight_hists[cfg.SEEDS[0]], names, dirs["graphs"] / "band_weight_evolution.png",
                              f"Average band-attention weight per epoch (seed {cfg.SEEDS[0]})")
    viz.plot_seed_weights(final_w, names, dirs["graphs"] / "final_band_weights.png",
                          "Final average band weight per channel (higher = selected first)")
    viz.plot_selection_matrix(selected, names, ks, dirs["graphs"] / "selected_channels.png",
                              "Channels kept by UBS-CNN (BS-Net-Conv) at each K")
    if cfg.RUN_EXHAUSTIVE_CHECK:
        viz.plot_paired_bars(sel_df, "k", ["UBS-CNN top-K", "Exhaustive optimum"],
                             [("Objective (GA/PSO/ACO/SA criterion)", ["objective", "exhaustive_optimum"])],
                             dirs["graphs"] / "objective_vs_optimum.png",
                             "How good is the CNN's ranking? Objective of its top-K vs the best possible subset",
                             x_label="Selected channels (K)")
    viz.plot_correlation(np.corrcoef(x_s.T), names, dirs["graphs"] / "channel_correlation.png")

    summary = {"experiment": exp, "hyperparameters": hp, "channels": names, "split": split,
               "n_patches": int(len(patches)), "n_train_patches": int(len(x_tr)), "n_heldout_patches": int(len(x_val)),
               "n_parameters": count_parameters(BSNetConv(C)), "seeds": seed_rows,
               "consensus_mean_weight": dict(zip(names, mean_w.tolist())),
               "consensus_ranking": [names[i] for i in ranking], "top3_per_seed": {s: sorted(names[i] for i in v)
                                                                                    for s, v in top_sets.items()},
               "selected": {k: [names[i] for i in v] for k, v in selected.items()},
               "timing_memory": stats, "process_rss_mb_end": rss_mb(), "total_seconds": time.perf_counter() - t_start}
    save_json(summary, dirs["metrics"] / "train_summary.json")
    log.info(f"Training finished in {summary['total_seconds']:.0f} s. Models: {dirs['models']}")


if __name__ == "__main__":
    main()
