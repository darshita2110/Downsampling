"""
Plots shared by Kartavya experiments (matplotlib, saved as PNG; no window is opened).

Style rules used everywhere:
  * one y-axis per chart (two measures with different units -> two panels, never a twin axis)
  * categorical colours in a fixed order, so "date 1" is the same colour in every graph
  * sequential (one hue, light -> dark) colour maps for error magnitude
  * light grid and axes so the data stands out
"""
import matplotlib

matplotlib.use("Agg")                     # draw to files only (works on servers without a screen)
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np

# Fixed categorical order (validated colour-blind-safe palette for adjacent series)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
    "lines.linewidth": 2.0, "lines.markersize": 5.5,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_explained_variance(ratio, path, title="Explained variance per principal component",
                            thresholds=(0.90, 0.95, 0.99)):
    """Bars: variance share of each component. Line: running total. Both in % on the same axis."""
    k = np.arange(1, len(ratio) + 1)
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(k, 100 * ratio, color=SERIES[0], width=0.6, label="Individual", zorder=2)
    ax.plot(k, 100 * cum, color=SERIES[1], marker="o", label="Cumulative", zorder=3)
    for t in thresholds:
        ax.axhline(100 * t, color=TEXT2, lw=0.8, ls=":", zorder=1)
        ax.text(k[-1] + 0.45, 100 * t, f"{int(t * 100)}%", va="center", fontsize=8, color=TEXT2)
    for ki, c in zip(k, cum):
        ax.annotate(f"{100 * c:.1f}", (ki, 100 * c), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=TEXT)
    ax.set_xticks(k)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title, loc="left")
    ax.legend(loc="center right")
    return _save(fig, path)


def plot_metric_panels(table, metrics, path, title, series_col="date", x_col="k",
                       x_label="Retained components (K)"):
    """
    Small multiples: one panel per metric, one line per series (e.g. per test date).
    table: pandas DataFrame with columns x_col, series_col and every metric.
    """
    n = len(metrics)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.3 * rows), squeeze=False)
    series = list(dict.fromkeys(table[series_col]))
    for ax, (col, label) in zip(axes.flat, metrics):
        for i, s in enumerate(series):
            t = table[table[series_col] == s].sort_values(x_col)
            ax.plot(t[x_col], t[col], color=SERIES[i % len(SERIES)], marker="o", label=str(s))
        ax.set_title(label, loc="left")
        ax.set_xlabel(x_label)
        ax.set_xticks(sorted(table[x_col].unique()))
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    if len(series) > 1:
        axes.flat[0].legend(title=series_col)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def plot_group_lines(table, value_col, groups, path, title, y_label, x_col="k",
                     x_label="Retained components (K)"):
    """One line per product group (RGB, False_Color, NDVI, NDWI). Lines converge at high K, so the
    legend (not end-of-line labels, which would collide) identifies them."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, g in enumerate(groups):
        t = table[table["group"] == g].sort_values(x_col)
        ax.plot(t[x_col], t[value_col], color=SERIES[i], marker="o", label=g)
    ax.set_xticks(sorted(table[x_col].unique()))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    return _save(fig, path)


def plot_tradeoff(table, x_col, y_col, path, title, x_label, y_label, label_col="k", series_col="date"):
    """Quality vs size: one point per K, labelled with K."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, s in enumerate(dict.fromkeys(table[series_col])):
        t = table[table[series_col] == s].sort_values(x_col)
        ax.plot(t[x_col], t[y_col], color=SERIES[i % len(SERIES)], marker="o", label=str(s))
        if i == 0:
            for _, r in t.iterrows():
                ax.annotate(f"K={int(r[label_col])}", (r[x_col], r[y_col]), textcoords="offset points",
                            xytext=(5, 6), fontsize=7.5, color=TEXT2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left")
    if table[series_col].nunique() > 1:
        ax.legend(title=series_col)
    return _save(fig, path)


def plot_correlation(corr, names, path, title="Channel correlation (training pixels)"):
    """Correlation matrix with a diverging map centred on 0."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_loadings(components, names, path, n_show=4, title="PCA loadings (what each component is made of)",
                  prefix="PC"):
    """Bar chart of the weight of every input channel in the first components (one panel each)."""
    n_show = min(n_show, components.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(3.3 * n_show, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(names))
    for i, ax in enumerate(axes):
        w = components[i]
        ax.barh(y, w, color=[SERIES[0] if v >= 0 else SERIES[1] for v in w], height=0.6)
        ax.axvline(0, color=TEXT2, lw=0.8)
        ax.set_title(f"{prefix}{i + 1}", loc="left")
        ax.set_xlim(-1, 1)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def _stretch(img, valid, lo=2, hi=98):
    """Percentile stretch for display only."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        v = img[..., c][valid]
        a, b = np.percentile(v, [lo, hi]) if v.size else (0, 1)
        out[..., c] = np.clip((img[..., c] - a) / max(b - a, 1e-9), 0, 1)
    out[~valid] = 1.0
    return out


def plot_component_images(reduced, valid, path, title, n_show=3, prefix="PC"):
    """Each principal-component image in grey, plus PC1-PC3 as a colour composite."""
    n_show = min(n_show, reduced.shape[-1])
    panels = n_show + (1 if reduced.shape[-1] >= 3 else 0)
    fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3.9))
    axes = np.atleast_1d(axes)
    for i in range(n_show):
        axes[i].imshow(_stretch(reduced[..., i:i + 1], valid)[..., 0], cmap="gray")
        axes[i].set_title(f"{prefix}{i + 1}", loc="left")
    if reduced.shape[-1] >= 3:
        axes[-1].imshow(_stretch(reduced[..., :3], valid))
        axes[-1].set_title(f"{prefix}1-{prefix}3 as RGB", loc="left")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def plot_reconstruction_gallery(original, recons, valid, names, path, title):
    """
    Rows: RGB composite, False-colour composite, NDVI, NDWI, absolute error (all channels).
    Columns: original, then the reconstruction for every K in `recons` ({K: image}).
    Data are in [0, 1]; composites are shown exactly as stored (no extra stretch) so that what you see
    is what was reconstructed.
    """
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    if all(n in idx for n in ["RGB:R(B04)", "RGB:G(B03)", "RGB:B(B02)"]):
        rows.append(("RGB", [idx["RGB:R(B04)"], idx["RGB:G(B03)"], idx["RGB:B(B02)"]], None))
    if all(n in idx for n in ["False_Color:R(B08)", "False_Color:G(B04)", "False_Color:B(B03)"]):
        rows.append(("False colour", [idx["False_Color:R(B08)"], idx["False_Color:G(B04)"],
                                      idx["False_Color:B(B03)"]], None))
    if "NDVI" in idx:
        rows.append(("NDVI", [idx["NDVI"]], "RdYlGn"))
    if "NDWI" in idx:
        rows.append(("NDWI", [idx["NDWI"]], "RdBu"))
    if not rows:                                    # raw bands: show the first three as a composite
        rows.append(("Bands " + ",".join(names[:3]), list(range(min(3, len(names)))), None))
    rows.append(("|error|", None, "Reds"))

    cols = ["Original"] + [f"K = {k}" for k in recons]
    imgs = [original] + list(recons.values())
    aspect = original.shape[0] / original.shape[1]              # height / width of the test image
    cell_w = 2.7
    fig, axes = plt.subplots(len(rows), len(cols), squeeze=False, layout="constrained",
                             figsize=(cell_w * len(cols) + 0.9, (cell_w * aspect + 0.35) * len(rows) + 0.5))
    err_max = max(float(np.abs(r - original)[valid].mean(axis=-1).max()) if valid.any() else 1 for r in recons.values())
    err_max = min(err_max, 0.25)
    for r, (label, ch, cmap) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.axis("off")
            if ch is None:                              # error row
                if c == 0:
                    ax.text(0.5, 0.5, "mean |error|\nover all channels", ha="center", va="center",
                            fontsize=9, color=TEXT2, transform=ax.transAxes)
                    continue
                e = np.abs(img - original).mean(axis=-1)
                e[~valid] = 0
                im = ax.imshow(e, cmap=cmap, vmin=0, vmax=err_max)
            elif len(ch) == 3:
                show = np.clip(img[..., ch], 0, 1).copy()
                show[~valid] = 1
                ax.imshow(show)
            else:
                show = np.where(valid, img[..., ch[0]], np.nan)
                ax.imshow(show, cmap=cmap, vmin=0, vmax=1)
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.text(-0.06, 0.5, label, rotation=90, ha="right", va="center", fontsize=10,
                        transform=ax.transAxes, color=TEXT)
    fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.05, pad=0.02, shrink=0.35, anchor=(0, 0),
                 label="|error| (0-1 scale)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


def plot_convergence(histories, tol, path, title="FastICA convergence"):
    """One line per model: rotation change after each iteration (log scale), with the tolerance."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (label, h) in enumerate(histories.items()):
        h = np.maximum(np.asarray(h, dtype=float), 1e-8)        # exact zeros (e.g. K=1) drawn at 1e-8 on the log axis
        ax.plot(np.arange(1, len(h) + 1), h, color=SERIES[i % len(SERIES)], marker="o", label=label)
    ax.axhline(tol, color=TEXT2, lw=0.9, ls=":")
    ax.text(ax.get_xlim()[1], tol, f" tol = {tol:g}", va="bottom", ha="right", fontsize=8, color=TEXT2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max rotation change  |1 - |<w_new, w_old>||")
    ax.set_title(title, loc="left")
    ax.legend(title="Components", ncol=2)
    return _save(fig, path)


def plot_paired_bars(table, x_col, labels, panels, path, title, x_label="Retained components (K)"):
    """
    Small multiples of grouped bars, e.g. ICA vs PCA for several statistics.
    panels: list of (panel title, [column for series 1, column for series 2, ...]).
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.6), squeeze=False)
    xs = np.arange(len(table))
    width = 0.8 / len(labels)
    for ax, (ptitle, pcols) in zip(axes.flat, panels):
        for i, (c, lab) in enumerate(zip(pcols, labels)):
            ax.bar(xs + (i - (len(labels) - 1) / 2) * width, table[c], width=width * 0.92,
                   color=SERIES[i], label=lab, zorder=2)
        ax.set_xticks(xs, table[x_col].astype(int))
        ax.set_xlabel(x_label)
        ax.set_title(ptitle, loc="left")
    axes.flat[0].legend()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrix(cm, classes, path, title):
    """Row-normalised confusion matrix (sequential single hue) with raw counts printed in each cell."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 3, 0.9 * len(classes) + 2.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of true class")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_ga_convergence(runs, optimum, path, title, x_key="generation", best_key="best_so_far",
                        mean_key="gen_mean", unit="Generation", mean_label="population mean (first seed)"):
    """
    GA / PSO convergence, one panel per K (small multiples, shared metric).
    runs   : {k: {seed: history list of dicts with generation, best_so_far, gen_mean}}
    optimum: {k: best fitness found by exhaustive search} or {} (drawn as a dotted reference line)
    Solid lines = best fitness found so far for each seed; thin dashed line = population mean of the
    first seed (shows the population closing in on the best).
    """
    ks = sorted(runs)
    cols = min(len(ks), 4)
    rows = int(np.ceil(len(ks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False)
    for ax, k in zip(axes.flat, ks):
        for i, (seed, h) in enumerate(runs[k].items()):
            g = [r[x_key] for r in h]
            ax.plot(g, [r[best_key] for r in h], color=SERIES[i % len(SERIES)], marker="o",
                    markersize=3, lw=1.6, label=f"seed {seed}")
            if i == 0:
                ax.plot(g, [r[mean_key] for r in h], color=SERIES[0], lw=1.0, ls="--", alpha=0.7,
                        label=mean_label)
        if k in optimum:
            ax.axhline(optimum[k], color=TEXT2, lw=0.9, ls=":")
        last = max(r[x_key] for h in runs[k].values() for r in h)
        ax.set_xlim(-0.5, max(last, 1) + 0.5)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if last == 0:
            ax.text(0.5, 0.35, f"every subset fits in the\ninitial {'swarm' if unit == 'Iteration' else 'population'}: done at {unit.lower()} 0",
                    transform=ax.transAxes, ha="center", fontsize=8, color=TEXT2)
        ax.set_title(f"K = {k}", loc="left")
        ax.set_xlabel(unit)
    for ax in list(axes.flat)[len(ks):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Fitness")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title + "   (dotted = exhaustive-search optimum)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path)


def plot_selection_matrix(selected, names, ks, path, title):
    """Rows = K, columns = channels; a filled cell means the channel is kept at that K."""
    m = np.zeros((len(ks), len(names)))
    for r, k in enumerate(ks):
        m[r, selected[k]] = 1
    fig, ax = plt.subplots(figsize=(0.8 * len(names) + 2.2, 0.45 * len(ks) + 1.9))
    ax.imshow(m, cmap=matplotlib.colors.ListedColormap([SURFACE, SERIES[0]]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(ks)), [f"K = {k}" for k in ks])
    ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ks)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=GRID, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_pheromone(histories, names, path, title):
    """
    ACO pheromone per channel across iterations, one panel per K (small multiples).
    histories: {k: history list with "iteration" and "pheromone" (one value per channel)}.
    One line per channel in fixed palette order (8 channels = 8 categorical colours), legend below.
    """
    ks = sorted(histories)
    cols = min(len(ks), 4)
    rows = int(np.ceil(len(ks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False, sharey=True)
    for ax, k in zip(axes.flat, ks):
        h = histories[k]
        it = [r["iteration"] for r in h]
        tau = np.array([r["pheromone"] for r in h])
        for j, n in enumerate(names):
            ax.plot(it, tau[:, j], color=SERIES[j % len(SERIES)], lw=1.6, marker="o", markersize=2.5, label=n)
        ax.set_title(f"K = {k}", loc="left")
        ax.set_xlabel("Iteration")
        ax.set_xlim(-0.5, max(max(it), 1) + 0.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in list(axes.flat)[len(ks):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Pheromone τ (after update)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _save(fig, path)


def plot_sa_temperature(histories, path, title, window=20):
    """
    Two panels (one y-axis each): temperature per move (log scale) and the rolling share of proposed
    moves that were accepted. One line per K, fixed palette order.
    histories: {k: history list with "iteration", "temperature", "accepted"}.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for i, (k, h) in enumerate(sorted(histories.items())):
        it = np.array([r["iteration"] for r in h])
        T = np.array([r["temperature"] for r in h])
        acc = np.array([float(r["accepted"]) for r in h])
        roll = np.convolve(acc, np.ones(window) / window, mode="valid") if len(acc) >= window else acc
        color = SERIES[i % len(SERIES)]
        axes[0].plot(it, T, color=color, lw=1.6, label=f"K = {k}")
        axes[1].plot(it[len(it) - len(roll):], roll, color=color, lw=1.6, label=f"K = {k}")
    axes[0].set_yscale("log")
    axes[0].set_title("Temperature (geometric cooling)", loc="left")
    axes[0].set_xlabel("Move")
    axes[0].set_ylabel("T")
    axes[1].set_title(f"Accepted moves (rolling mean over {window})", loc="left")
    axes[1].set_xlabel("Move")
    axes[1].set_ylabel("Share accepted")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_curves(hist, path, title):
    """
    Neural-network training history, small multiples (one metric per panel, one line per seed).
    hist: DataFrame with seed, epoch, train_loss, train_mse, train_l1_term, val_mse.
    """
    panels = [("train_loss", "Total loss (MSE + L1)", False), ("train_mse", "Training MSE (log)", True),
              ("val_mse", "Held-out patch MSE (log)", True), ("train_l1_term", "L1 term: λ·Σ|w| per batch", False)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (col, label, logy) in zip(axes, panels):
        for i, s in enumerate(dict.fromkeys(hist["seed"])):
            t = hist[hist.seed == s]
            ax.plot(t["epoch"], t[col], color=SERIES[i % len(SERIES)], lw=1.6, label=f"seed {s}")
        if logy:
            ax.set_yscale("log")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Epoch")
    axes[0].legend()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def plot_weight_evolution(w_hist, names, path, title):
    """Average attention weight of every channel after each epoch (one line per channel, fixed palette)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ep = np.arange(1, len(w_hist) + 1)
    for j, n in enumerate(names):
        ax.plot(ep, w_hist[:, j], color=SERIES[j % len(SERIES)], lw=1.6, label=n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean band weight w (0-1)")
    ax.set_title(title, loc="left")
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, path)


def plot_seed_weights(final_w, names, path, title):
    """Grouped bars: channels on the x-axis, one bar per seed, final mean weight."""
    seeds = list(final_w)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    xs = np.arange(len(names))
    width = 0.8 / len(seeds)
    for i, s in enumerate(seeds):
        ax.bar(xs + (i - (len(seeds) - 1) / 2) * width, final_w[s], width=width * 0.92, color=SERIES[i],
               label=f"seed {s}", zorder=2)
    ax.set_xticks(xs, names, rotation=35, ha="right")
    ax.set_ylabel("Mean band weight w")
    ax.set_title(title, loc="left")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)
