"""
UBS-CNN (BS-Net-Conv) in the common pipeline.

train: TRAINING region -> 16 x 16 x C patches (10 % held out) -> one network per seed (Adam, MSE + 0.01 L1 on the
       band weights) -> average band weights over patches and seeds = channel ranking -> top-K for every K ->
       LinearReconstructor per K (as GA/PSO/ACO/SA) + evaluation-only objective vs the exhaustive optimum
test : saved selections + weights (nothing re-trained); networks run forward on TEST patches only to report
       their attention weights and own reconstruction error
"""
import numpy as np
import pandas as pd

from algorithms.ubscnn import config as cfg
from common import visualization as viz
from common.band_selection import (evaluate_selection, exhaustive_allowed, fitness_graph, save_selection_model,
                                   selection_row, training_matrices)
from common.data_loader import rows_slice
from common.preprocessing import extract_patches
from common.utils import comb_safe, file_size_bytes, measure, rss_mb, save_json

NAME, DISPLAY, FAMILY, REQUIRES = "ubscnn", "UBS-CNN", "band selection (CNN attention)", ["torch"]


def output_name():
    return NAME


def _peak_rss_mb():
    try:
        import resource
        import sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1024
    except ImportError:                        # Windows
        return float("nan")


def training_patches(ctx, stride):
    blocks = []
    for entry in ctx.split["train"]:
        img, valid = ctx.dataset.load_normalized(entry["index"])
        rs = rows_slice(valid.shape[0], entry["rows"], ctx.fraction)
        p, _ = extract_patches(img, valid, cfg.PATCH_SIZE, stride, (rs.start, rs.stop))
        blocks.append(p)
    return np.concatenate(blocks) if blocks else np.zeros((0, ctx.C, cfg.PATCH_SIZE, cfg.PATCH_SIZE), np.float32)


def choose_stride(ctx):
    """Configured stride; lowered for small images until MIN_TRAIN_PATCHES patches exist (1 = authors' setting)."""
    stride = cfg.PATCH_STRIDE
    patches = training_patches(ctx, stride)
    while len(patches) < cfg.MIN_TRAIN_PATCHES and stride > 1:
        stride = max(1, stride // 2)
        patches = training_patches(ctx, stride)
    if len(patches) < 2:
        raise ValueError(f"The training region yields {len(patches)} fully valid {cfg.PATCH_SIZE}x{cfg.PATCH_SIZE} "
                         "patches; UBS-CNN needs images (training part) at least "
                         f"{cfg.PATCH_SIZE} x {cfg.PATCH_SIZE} pixels without no-data")
    return stride, patches


def train_one(seed, x_tr, x_val, epochs, names, log):
    import torch
    from algorithms.ubscnn.implementation import BSNetConv, band_weights, bsnet_loss, reconstruction_mse
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BSNetConv(x_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    xt, xv = torch.from_numpy(x_tr), torch.from_numpy(x_val)
    bs = min(cfg.BATCH_SIZE, len(xt))
    history, weight_hist = [], []
    import time
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        order = rng.permutation(len(xt))
        sums = np.zeros(3)
        n_batches = max(1, len(xt) // bs)                       # as in the authors' code (last partial batch dropped)
        for b in range(n_batches):
            xb = xt[order[b * bs:(b + 1) * bs]]
            if len(xb) < 2:                                     # BatchNorm needs > 1 sample
                continue
            opt.zero_grad()
            x_hat, w = model(xb)
            loss, mse, l1 = bsnet_loss(xb, x_hat, w, cfg.L1_LAMBDA)
            loss.backward()
            opt.step()
            sums += [loss.item(), mse.item(), l1.item()]
        mean_w = band_weights(model, xt).mean(0)
        weight_hist.append(mean_w)
        rec = {"seed": seed, "epoch": ep, "train_loss": sums[0] / n_batches, "train_mse": sums[1] / n_batches,
               "train_l1_term": sums[2] / n_batches, "val_mse": reconstruction_mse(model, xv) if len(xv) else np.nan,
               "epoch_seconds": time.perf_counter() - t0,
               "ranking": " > ".join(names[i] for i in np.argsort(-mean_w)[:20])}
        if len(names) <= 64:
            rec.update({f"w_{n}": float(v) for n, v in zip(names, mean_w)})
        history.append(rec)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            log.info(f"  seed {seed} epoch {ep:3d}: loss {rec['train_loss']:.4f} (MSE {rec['train_mse']:.5f} + "
                     f"L1 {rec['train_l1_term']:.4f}), val MSE {rec['val_mse']:.5f}, {rec['epoch_seconds']:.1f} s")
    return model, history, np.array(weight_hist)


def train(ctx):
    import torch
    from algorithms.ga.implementation import LinearReconstructor, SubsetFitness, exhaustive_search
    from algorithms.ubscnn.implementation import BSNetConv, count_parameters
    log, dirs, names, C = ctx.log, ctx.dirs, ctx.names, ctx.C
    epochs = int(ctx.options.get("epochs") or cfg.EPOCHS)
    if cfg.TORCH_THREADS:
        torch.set_num_threads(cfg.TORCH_THREADS)
    x_train, x_s, x_fit, x_val_pix, corr, dup, stats, rng = training_matrices(
        ctx, cfg.MAX_TRAIN_PIXELS, cfg.VALIDATION_FRACTION, cfg.RANDOM_SEED)
    with measure(stats, "make_patches"):
        stride, patches = choose_stride(ctx)
    hp = {"method": "BS-Net-Conv (Cai et al., IEEE TGRS 2020, arXiv:1904.08269) used as UBS-CNN",
          "patch_size": cfg.PATCH_SIZE, "patch_stride_configured": cfg.PATCH_STRIDE, "patch_stride_used": stride,
          "l1_lambda": cfg.L1_LAMBDA, "learning_rate": cfg.LEARNING_RATE, "batch_size": cfg.BATCH_SIZE,
          "epochs": epochs, "seeds": cfg.SEEDS, "optimizer": "Adam", "val_patch_fraction": cfg.VAL_PATCH_FRACTION,
          "torch": torch.__version__, "threads": torch.get_num_threads()}
    log.info(f"hyper-parameters {hp}")
    if stride != cfg.PATCH_STRIDE:
        log.info(f"Small training region: stride lowered {cfg.PATCH_STRIDE} -> {stride} to get {len(patches):,} patches")
    perm = rng.permutation(len(patches))
    n_val = int(round(len(patches) * cfg.VAL_PATCH_FRACTION))
    x_val, x_tr = patches[perm[:n_val]], patches[perm[n_val:]]
    log.info(f"Patches: {len(patches):,} -> train {len(x_tr):,} / held-out {len(x_val):,}; BS-Net-Conv has "
             f"{count_parameters(BSNetConv(C)):,} trainable parameters")

    histories, weight_hists, final_w, seed_rows = [], {}, {}, []
    for seed in cfg.SEEDS:
        st = {}
        with measure(st, "train"):
            model, hist, w_hist = train_one(seed, x_tr, x_val, epochs, names, log)
        path = dirs["models"] / f"bsnet_conv_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "n_bands": C, "channel_names": names, "hyperparameters": hp,
                    "dataset_id": ctx.dataset.dataset_id}, path)
        histories += hist
        weight_hists[seed], final_w[seed] = w_hist, w_hist[-1]
        rank = np.argsort(-w_hist[-1])
        seed_rows.append({"seed": seed, "train_seconds": st["train"]["seconds"],
                          # tracemalloc does not see PyTorch's C++ allocations, so the process peak RSS is reported too
                          "train_peak_python_mb": st["train"]["peak_python_mb"], "rss_after_mb": st["train"]["rss_after_mb"],
                          "process_peak_rss_mb": _peak_rss_mb(), "final_train_mse": hist[-1]["train_mse"],
                          "final_val_mse": hist[-1]["val_mse"], "weights_file_bytes": file_size_bytes(path),
                          "ranking": " > ".join(names[i] for i in rank),
                          **({f"w_{n}": float(v) for n, v in zip(names, w_hist[-1])} if C <= 64 else {})})
        log.info(f"seed {seed}: trained in {st['train']['seconds']:.0f} s; ranking {seed_rows[-1]['ranking'][:200]}")
    hist_df = pd.DataFrame(histories)
    hist_df.to_csv(dirs["metrics"] / "training_history.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(dirs["metrics"] / "seed_rankings.csv", index=False)

    mean_w = np.mean([final_w[s] for s in cfg.SEEDS], axis=0)
    ranking = np.argsort(-mean_w, kind="stable")
    log.info("Consensus channel ranking: " + ", ".join(f"{names[i]} {mean_w[i]:.3f}" for i in ranking[:20]))
    pd.DataFrame({"rank": np.arange(1, C + 1), "channel": [names[i] for i in ranking], "mean_weight": mean_w[ranking],
                  **{f"weight_seed{s}": final_w[s][ranking] for s in cfg.SEEDS}}).to_csv(
        dirs["metrics"] / "channel_ranking.csv", index=False)

    objective = SubsetFitness(x_fit, x_val_pix, names, cfg.W_MIN_R2, cfg.W_REDUNDANCY)
    ks = [k for k in (cfg.FEATURE_COUNTS or ctx.ks) if 1 <= k < C]
    sel_rows, selected = [], {}
    for k in ks:
        sel = np.sort(ranking[:k])
        selected[k] = sel.tolist()
        d = objective.details(np.isin(np.arange(C), sel).astype(np.int8))
        row = selection_row(k, C, names, d, objective_col="objective")
        row["seeds_with_same_topk"] = sum(set(np.argsort(-final_w[s])[:k]) == set(sel) for s in cfg.SEEDS)
        if exhaustive_allowed(C, k, cfg.RUN_EXHAUSTIVE_CHECK):
            rk = exhaustive_search(objective, C, k)
            opt = rk[0]["fitness"]
            row.update({"exhaustive_optimum": opt,
                        "optimum_subsets": " | ".join("+".join(r["selected"]) for r in rk if abs(r["fitness"] - opt) < 1e-9),
                        "rank_among_all_subsets": 1 + sum(r["fitness"] > d["fitness"] + 1e-9 for r in rk),
                        "n_subsets": len(rk), "gap_to_optimum": opt - d["fitness"]})
        else:
            row["exhaustive_optimum"] = np.nan
            row["optimum_subsets"] = f"not run ({comb_safe(C, k):,} subsets)"
        rec = LinearReconstructor(sel).fit(x_s)
        save_selection_model(ctx, f"ubscnn_K{k}.npz", rec, d, mean_weight=mean_w)
        sel_rows.append(row)
        log.info(f"K={k}: select {d['selected'][:10]} | objective {d['fitness']:.5f}"
                 + (f" (optimum {row['exhaustive_optimum']:.5f}, rank {row['rank_among_all_subsets']}/{row['n_subsets']})"
                    if "n_subsets" in row else "") + f" | same top-{k} in {row['seeds_with_same_topk']}/{len(cfg.SEEDS)} seeds")
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(dirs["metrics"] / "selected_features.csv", index=False)

    viz.plot_training_curves(hist_df, dirs["graphs"] / "training_loss.png",
                             f"BS-Net-Conv training ({ctx.dataset.name}): loss per epoch, one line per seed")
    viz.plot_weight_evolution(weight_hists[cfg.SEEDS[0]], names, dirs["graphs"] / "band_weight_evolution.png",
                              f"Average band-attention weight per epoch (seed {cfg.SEEDS[0]})")
    viz.plot_seed_weights(final_w, names, dirs["graphs"] / "final_band_weights.png",
                          "Final average band weight per channel (higher = selected first)")
    viz.plot_selection_matrix(selected, names, ks, dirs["graphs"] / "selected_channels.png",
                              "Channels kept by UBS-CNN (BS-Net-Conv) at each K")
    fitness_graph(ctx, sel_df, "objective")
    ok = sel_df["exhaustive_optimum"].notna()
    if ok.any():
        viz.plot_paired_bars(sel_df[ok], "k", ["UBS-CNN top-K", "Exhaustive optimum"],
                             [("Objective (GA/PSO/ACO/SA criterion)", ["objective", "exhaustive_optimum"])],
                             dirs["graphs"] / "objective_vs_optimum.png",
                             "How good is the CNN's ranking? Objective of its top-K vs the best possible subset",
                             x_label="Selected channels (K)")
    save_json({"algorithm": ctx.name, "hyperparameters": hp, "channels": names, "split": ctx.split,
               "n_patches": int(len(patches)), "n_train_patches": int(len(x_tr)), "n_heldout_patches": int(len(x_val)),
               "n_parameters": count_parameters(BSNetConv(C)), "seeds": seed_rows,
               "consensus_mean_weight": dict(zip(names, mean_w.tolist())),
               "consensus_ranking": [names[i] for i in ranking],
               "selected": {k: [names[i] for i in v] for k, v in selected.items()},
               "timing_memory": stats, "process_rss_mb_end": rss_mb()}, dirs["metrics"] / "train_summary.json")
    ctx.info.update({"patch_stride_used": stride, "epochs": epochs,
                     "selected_channels": {k: [names[i] for i in v] for k, v in selected.items()}})


def test(ctx):
    import torch
    from algorithms.ubscnn.implementation import BSNetConv, band_weights, reconstruction_mse
    names = ctx.names
    stride = int(ctx.info.get("patch_stride_used", cfg.PATCH_STRIDE))
    nets = {}
    for p in sorted(ctx.dirs["models"].glob("bsnet_conv_seed*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck.get("dataset_id") != ctx.dataset.dataset_id:
            raise ValueError(f"{p.name} was trained on a different dataset")
        net = BSNetConv(ck["n_bands"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        nets[int(p.stem.split("seed")[1])] = net
    ctx.log.info(f"Loaded {len(nets)} trained BS-Net-Conv networks (seeds {sorted(nets)}); forward pass only")
    train_rank = pd.read_csv(ctx.dirs["metrics"] / "channel_ranking.csv")
    cnn_records = []

    def per_sample(tag, sample_name, img, valid):
        tp, _ = extract_patches(img, valid, cfg.PATCH_SIZE, stride)
        if not len(tp):
            ctx.log.warning(f"  {tag}: no fully valid test patch for the network forward pass")
            return
        tp_t = torch.from_numpy(tp)
        test_w = {}
        for seed, net in nets.items():
            st = {}
            with measure(st, "fwd"):
                w = band_weights(net, tp_t)
            test_w[seed] = w.mean(0)
            cnn_records.append({"sample_name": sample_name, "sample": tag, "seed": seed, "n_test_patches": len(tp),
                                "network_reconstruction_mse": reconstruction_mse(net, tp_t),
                                "forward_seconds": st["fwd"]["seconds"],
                                **({f"w_{n}": float(v) for n, v in zip(names, test_w[seed])} if len(names) <= 64 else {})})
        mw = np.mean(list(test_w.values()), axis=0)
        test_rank = [names[i] for i in np.argsort(-mw)]
        cnn_records[-1]["test_ranking_mean_of_seeds"] = " > ".join(test_rank[:30])
        ctx.log.info(f"  CNN on {len(tp):,} test patches: test-region ranking {test_rank[:10]} "
                     f"(training ranking {train_rank.channel.tolist()[:10]})")

    df = evaluate_selection(ctx, cfg, "ubscnn", "UBS-CNN (BS-Net-Conv) selection",
                            compare_others=("ga", "pso", "aco", "sa"), per_sample=per_sample,
                            objective_label="Objective (training, evaluation only)")
    pd.DataFrame(cnn_records).to_csv(ctx.dirs["metrics"] / "test_network_outputs.csv", index=False)
    return df
