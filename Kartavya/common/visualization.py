"""
Plots shared by Kartavya experiments (matplotlib, saved as PNG; no window is opened).

Style rules used everywhere (unchanged from the original repository):
  * one y-axis per chart (two measures with different units -> two panels, never a twin axis)
  * categorical colours in a fixed order, so "series 1" is the same colour in every graph
  * sequential (one hue, light -> dark) colour maps for error magnitude
  * light grid and axes so the data stands out
Generalisation: every plot now copes with any number of channels (labels/annotations are thinned out for
hyperspectral data) and with any channel names (no dataset-specific product names are assumed).
"""
import matplotlib

matplotlib.use("Agg")                     # draw to files only (works on servers without a screen)
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np

# Fixed categorical order (validated colour-blind-safe palette for adjacent series)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MAX_LABELLED = 40          # beyond this many channels tick labels are thinned
MAX_ANNOTATED = 16         # beyond this many channels correlation values are not printed
MAX_LINES = 12             # beyond this many channels only the most important are drawn as lines

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


def color(i):
    return SERIES[i % len(SERIES)]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _tick_positions(n):
    step = max(1, int(np.ceil(n / MAX_LABELLED)))
    return list(range(0, n, step))


def _short(names, n=18):
    return [s if len(s) <= n else s[:n - 1] + "…" for s in names]


def plot_explained_variance(ratio, path, title="Explained variance per principal component",
                            thresholds=(0.90, 0.95, 0.99)):
    """Bars: variance share of each component. Line: running total. Both in % on the same axis."""
    k = np.arange(1, len(ratio) + 1)
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(max(7.5, 0.12 * len(k)), 4.2))
    ax.bar(k, 100 * ratio, color=SERIES[0], width=0.6, label="Individual", zorder=2)
    ax.plot(k, 100 * cum, color=SERIES[1], marker="o" if len(k) <= 40 else None, label="Cumulative", zorder=3)
    for t in thresholds:
        ax.axhline(100 * t, color=TEXT2, lw=0.8, ls=":", zorder=1)
        ax.text(k[-1] + 0.45, 100 * t, f"{int(t * 100)}%", va="center", fontsize=8, color=TEXT2)
    if len(k) <= 20:
        for ki, c in zip(k, cum):
            ax.annotate(f"{100 * c:.1f}", (ki, 100 * c), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=TEXT)
    ax.set_xticks([k[i] for i in _tick_positions(len(k))])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title, loc="left")
    ax.legend(loc="center right")
    return _save(fig, path)


def plot_metric_panels(table, metrics, path, title, series_col="date", x_col="k",
                       x_label="Retained components (K)"):
    """
    Small multiples: one panel per metric, one line per series (e.g. per test sample).
    table: pandas DataFrame with columns x_col, series_col and every metric.
    """
    metrics = [(c, lab) for c, lab in metrics if c in table.columns]
    n = len(metrics)
    if n == 0:
        return None
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.3 * rows), squeeze=False)
    series = list(dict.fromkeys(table[series_col]))
    for ax, (col, label) in zip(axes.flat, metrics):
        for i, s in enumerate(series):
            t = table[table[series_col] == s].sort_values(x_col)
            ax.plot(t[x_col], t[col], color=color(i), marker="o", label=str(s))
        ax.set_title(label, loc="left")
        ax.set_xlabel(x_label)
        xs = sorted(table[x_col].unique())
        ax.set_xticks(xs if len(xs) <= 16 else xs[::int(np.ceil(len(xs) / 16))])
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    if 1 < len(series) <= 12:
        axes.flat[0].legend(title=series_col)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def plot_group_lines(table, value_col, groups, path, title, y_label, x_col="k",
                     x_label="Retained components (K)"):
    """One line per channel group (product or band region). The legend identifies them."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, g in enumerate(groups):
        t = table[table["group"] == g].sort_values(x_col)
        ax.plot(t[x_col], t[value_col], color=color(i), marker="o", label=g)
    ax.set_xticks(sorted(table[x_col].unique()))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left")
    ax.legend(loc="best", ncol=1 if len(groups) <= 8 else 2)
    return _save(fig, path)


def plot_tradeoff(table, x_col, y_col, path, title, x_label, y_label, label_col="k", series_col="date"):
    """Quality vs size: one point per K, labelled with K."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, s in enumerate(dict.fromkeys(table[series_col])):
        t = table[table[series_col] == s].sort_values(x_col)
        ax.plot(t[x_col], t[y_col], color=color(i), marker="o", label=str(s))
        if i == 0:
            for _, r in t.iterrows():
                ax.annotate(f"K={int(r[label_col])}", (r[x_col], r[y_col]), textcoords="offset points",
                            xytext=(5, 6), fontsize=7.5, color=TEXT2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left")
    if 1 < table[series_col].nunique() <= 12:
        ax.legend(title=series_col)
    return _save(fig, path)


def plot_correlation(corr, names, path, title="Channel correlation (training pixels)"):
    """Correlation matrix with a diverging map centred on 0."""
    n = len(names)
    size = min(14, max(6.4, 0.11 * n + 4))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(np.nan_to_num(corr), cmap="RdBu_r", vmin=-1, vmax=1)
    ticks = _tick_positions(n)
    ax.set_xticks(ticks, [_short(names)[i] for i in ticks], rotation=45, ha="right")
    ax.set_yticks(ticks, [_short(names)[i] for i in ticks])
    ax.grid(False)
    if n <= MAX_ANNOTATED:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr[i, j]) > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_loadings(components, names, path, n_show=4, title="PCA loadings (what each component is made of)",
                  prefix="PC"):
    """Bar chart of the weight of every input channel in the first components (one panel each)."""
    n_show = min(n_show, components.shape[0])
    c = len(names)
    fig, axes = plt.subplots(1, n_show, figsize=(3.3 * n_show, max(3.6, 0.06 * c + 2)), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(c)
    for i, ax in enumerate(axes):
        w = components[i]
        ax.barh(y, w, color=[SERIES[0] if v >= 0 else SERIES[1] for v in w], height=0.6)
        ax.axvline(0, color=TEXT2, lw=0.8)
        ax.set_title(f"{prefix}{i + 1}", loc="left")
        lim = max(1.0, float(np.abs(w).max()) * 1.05)
        ax.set_xlim(-lim, lim)
    ticks = _tick_positions(c)
    axes[0].set_yticks(ticks, [_short(names)[i] for i in ticks])
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def _stretch(img, valid, lo=2, hi=98, ref=None):
    """Percentile stretch for display only. ref: (a, b) per channel computed elsewhere (shared stretch)."""
    out = np.zeros(img.shape, dtype=np.float32)
    bounds = []
    for c in range(img.shape[-1]):
        if ref is not None:
            a, b = ref[c]
        else:
            v = img[..., c][valid]
            a, b = np.percentile(v, [lo, hi]) if v.size else (0, 1)
        bounds.append((a, b))
        out[..., c] = np.clip((img[..., c] - a) / max(b - a, 1e-9), 0, 1)
    out[~valid] = 1.0
    return out, bounds


def plot_component_images(reduced, valid, path, title, n_show=3, prefix="PC", labels=None):
    """Each reduced channel in grey, plus the first three as a colour composite."""
    n_show = min(n_show, reduced.shape[-1])
    panels = n_show + (1 if reduced.shape[-1] >= 3 else 0)
    aspect = reduced.shape[0] / max(reduced.shape[1], 1)
    fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3.6 * min(max(aspect, 0.4), 2.5) + 0.3))
    axes = np.atleast_1d(axes)
    for i in range(n_show):
        axes[i].imshow(_stretch(reduced[..., i:i + 1], valid)[0][..., 0], cmap="gray")
        axes[i].set_title(labels[i] if labels else f"{prefix}{i + 1}", loc="left", fontsize=9)
    if reduced.shape[-1] >= 3:
        axes[-1].imshow(_stretch(reduced[..., :3], valid)[0])
        axes[-1].set_title("first three as RGB", loc="left", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def display_rows(names):
    """
    Which channels to show in the reconstruction gallery.
      * the original repository's products, when present (RGB / False colour composites, NDVI, NDWI)
      * otherwise: a 3-band composite (bands 1-3 for 3-4 channel images; for hyperspectral cubes three bands
        spread over the spectrum, long -> short, a pseudo true colour) and one single band.
    """
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    known = [("RGB", ["RGB:R(B04)", "RGB:G(B03)", "RGB:B(B02)"]),
             ("False colour", ["False_Color:R(B08)", "False_Color:G(B04)", "False_Color:B(B03)"])]
    for label, chans in known:
        if all(c in idx for c in chans):
            rows.append((label, [idx[c] for c in chans], None))
    for single, cmap in [("NDVI", "RdYlGn"), ("NDWI", "RdBu")]:
        if single in idx:
            rows.append((single, [idx[single]], cmap))
    if rows:
        return rows
    c = len(names)
    if c >= 3:
        pick = [0, 1, 2] if c <= 4 else [int(round(c * f)) for f in (0.75, 0.45, 0.15)]
        pick = [min(p, c - 1) for p in pick]
        rows.append(("Composite " + "/".join(_short([names[i] for i in pick], 10)), pick, None))
    mid = c // 2
    rows.append((_short([names[mid]], 22)[0], [mid], "gray"))
    return rows


def plot_reconstruction_gallery(original, recons, valid, names, path, title):
    """
    Rows: composite(s) / single bands chosen by display_rows(), then the absolute error (all channels).
    Columns: original, then the reconstruction for every K in `recons` ({K: image}).
    Data are in [0, 1]. Composites of the original-repository products are shown exactly as stored; for
    other data one percentile stretch is computed on the ORIGINAL and applied unchanged to every
    reconstruction, so what you see is comparable across columns.
    """
    rows = display_rows(names) + [("|error|", None, "Reds")]
    plain = any(r[0] in ("RGB", "False colour") for r in rows)
    cols = ["Original"] + [f"K = {k}" for k in recons]
    imgs = [original] + list(recons.values())
    aspect = original.shape[0] / max(original.shape[1], 1)
    cell_w = 2.7
    fig, axes = plt.subplots(len(rows), len(cols), squeeze=False, layout="constrained",
                             figsize=(cell_w * len(cols) + 0.9, (cell_w * min(aspect, 2.5) + 0.35) * len(rows) + 0.5))
    err_max = max([float(np.abs(r - original)[valid].mean(axis=-1).max()) if valid.any() else 1
                   for r in recons.values()] or [1])
    err_max = min(max(err_max, 1e-6), 0.25)
    im = None
    stretches = {}
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
                if plain:
                    show = np.clip(img[..., ch], 0, 1).copy()
                    show[~valid] = 1
                else:
                    if r not in stretches:
                        stretches[r] = _stretch(original[..., ch], valid)[1]
                    show = _stretch(img[..., ch], valid, ref=stretches[r])[0]
                ax.imshow(show)
            else:
                show = np.where(valid, img[..., ch[0]], np.nan)
                ax.imshow(show, cmap=cmap, vmin=0, vmax=1)
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.text(-0.06, 0.5, label, rotation=90, ha="right", va="center", fontsize=9,
                        transform=ax.transAxes, color=TEXT)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.05, pad=0.02, shrink=0.35, anchor=(0, 0),
                     label="|error| (0-1 scale)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


def plot_convergence(histories, tol, path, title="FastICA convergence"):
    """One line per model: rotation change after each iteration (log scale), with the tolerance."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (label, h) in enumerate(histories.items()):
        h = np.maximum(np.asarray(h, dtype=float), 1e-8)        # exact zeros (e.g. K=1) drawn at 1e-8 on the log axis
        ax.plot(np.arange(1, len(h) + 1), h, color=color(i), marker="o", label=label)
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
                   color=color(i), label=lab, zorder=2)
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
    n = len(classes)
    fig, ax = plt.subplots(figsize=(min(16, 0.6 * n + 3), min(14, 0.55 * n + 2.4)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=45, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{int(cm[i, j])}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of true class")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_ga_convergence(runs, optimum, path, title, x_key="generation", best_key="best_so_far",
                        mean_key="gen_mean", unit="Generation", mean_label="population mean (first seed)"):
    """
    GA / PSO / ACO / SA convergence, one panel per K (small multiples, shared metric).
    runs   : {k: {seed: history list of dicts}}
    optimum: {k: best fitness found by exhaustive search} or {} (drawn as a dotted reference line)
    """
    ks = sorted(runs)
    cols = min(len(ks), 4)
    rows = int(np.ceil(len(ks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False)
    for ax, k in zip(axes.flat, ks):
        for i, (seed, h) in enumerate(runs[k].items()):
            g = [r[x_key] for r in h]
            ax.plot(g, [r[best_key] for r in h], color=color(i), marker="o",
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
    fig.suptitle(title + ("   (dotted = exhaustive-search optimum)" if optimum else ""), x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path)


def plot_selection_matrix(selected, names, ks, path, title):
    """Rows = K, columns = channels; a filled cell means the channel is kept at that K."""
    m = np.zeros((len(ks), len(names)))
    for r, k in enumerate(ks):
        m[r, selected[k]] = 1
    n = len(names)
    fig, ax = plt.subplots(figsize=(min(18, 0.8 * n + 2.2) if n <= 40 else 14, 0.45 * len(ks) + 1.9))
    ax.imshow(m, cmap=matplotlib.colors.ListedColormap([SURFACE, SERIES[0]]), vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ticks = _tick_positions(n)
    ax.set_xticks(ticks, [_short(names)[i] for i in ticks], rotation=40, ha="right")
    ax.set_yticks(range(len(ks)), [f"K = {k}" for k in ks])
    ax.grid(False)
    if n <= MAX_LABELLED:
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ks)), minor=True)
        ax.grid(which="minor", color=GRID, lw=2)
        ax.tick_params(which="minor", length=0)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_pheromone(histories, names, path, title):
    """
    ACO pheromone per channel across iterations, one panel per K (small multiples).
    With more than 12 channels only the 12 with the highest final pheromone (any K) are drawn.
    """
    ks = sorted(histories)
    final = np.max([histories[k][-1]["pheromone"] for k in ks], axis=0)
    show = list(range(len(names))) if len(names) <= MAX_LINES else sorted(np.argsort(-final)[:MAX_LINES].tolist())
    cols = min(len(ks), 4)
    rows = int(np.ceil(len(ks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False, sharey=True)
    for ax, k in zip(axes.flat, ks):
        h = histories[k]
        it = [r["iteration"] for r in h]
        tau = np.array([r["pheromone"] for r in h])
        for n_i, j in enumerate(show):
            ax.plot(it, tau[:, j], color=color(n_i), lw=1.6, marker="o", markersize=2.5, label=names[j])
        ax.set_title(f"K = {k}", loc="left")
        ax.set_xlabel("Iteration")
        ax.set_xlim(-0.5, max(max(it), 1) + 0.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in list(axes.flat)[len(ks):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Pheromone τ (after update)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title + ("" if len(show) == len(names) else f" (top {len(show)} of {len(names)} channels)"),
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _save(fig, path)


def plot_sa_temperature(histories, path, title, window=20):
    """Two panels: temperature per move (log scale) and the rolling share of accepted moves; one line per K."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for i, (k, h) in enumerate(sorted(histories.items())):
        it = np.array([r["iteration"] for r in h])
        T = np.array([r["temperature"] for r in h])
        acc = np.array([float(r["accepted"]) for r in h])
        roll = np.convolve(acc, np.ones(window) / window, mode="valid") if len(acc) >= window else acc
        axes[0].plot(it, T, color=color(i), lw=1.6, label=f"K = {k}")
        axes[1].plot(it[len(it) - len(roll):], roll, color=color(i), lw=1.6, label=f"K = {k}")
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
    """Neural-network training history, small multiples (one metric per panel, one line per seed)."""
    panels = [("train_loss", "Total loss (MSE + L1)", False), ("train_mse", "Training MSE (log)", True),
              ("val_mse", "Held-out patch MSE (log)", True), ("train_l1_term", "L1 term: λ·Σ|w| per batch", False)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (col, label, logy) in zip(axes, panels):
        for i, s in enumerate(dict.fromkeys(hist["seed"])):
            t = hist[hist.seed == s]
            ax.plot(t["epoch"], t[col], color=color(i), lw=1.6, label=f"seed {s}")
        if logy:
            ax.set_yscale("log")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Epoch")
    axes[0].legend()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def plot_weight_evolution(w_hist, names, path, title):
    """Average attention weight per channel after each epoch (top 12 channels if there are more)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ep = np.arange(1, len(w_hist) + 1)
    show = range(len(names)) if len(names) <= MAX_LINES else sorted(np.argsort(-w_hist[-1])[:MAX_LINES])
    for n_i, j in enumerate(show):
        ax.plot(ep, w_hist[:, j], color=color(n_i), lw=1.6, label=names[j])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean band weight w (0-1)")
    ax.set_title(title, loc="left")
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, path)


def plot_seed_weights(final_w, names, path, title):
    """Grouped bars: channels on the x-axis, one bar per seed, final mean weight."""
    seeds = list(final_w)
    n = len(names)
    fig, ax = plt.subplots(figsize=(min(18, max(9.5, 0.15 * n * len(seeds))), 4.0))
    xs = np.arange(n)
    width = 0.8 / len(seeds)
    for i, s in enumerate(seeds):
        ax.bar(xs + (i - (len(seeds) - 1) / 2) * width, final_w[s], width=width * 0.92, color=color(i),
               label=f"seed {s}", zorder=2)
    ticks = _tick_positions(n)
    ax.set_xticks(ticks, [_short(names)[i] for i in ticks], rotation=35, ha="right")
    ax.set_ylabel("Mean band weight w")
    ax.set_title(title, loc="left")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------------------------ comparison graphs (unified runner)
def plot_algorithms_vs_k(table, metric, label, direction, path, title):
    """One line per algorithm: metric vs K (the same K values for every algorithm)."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for i, (alg, t) in enumerate(table.groupby("algorithm", sort=False)):
        t = t.sort_values("k")
        if t[metric].notna().any():
            ax.plot(t["k"], t[metric], color=color(i), marker="o", label=alg)
    ax.set_xlabel("Retained channels / components (K)")
    ax.set_ylabel(label)
    ax.set_title(f"{title}  ({'higher' if direction == 'max' else 'lower'} is better)", loc="left")
    xs = sorted(table["k"].unique())
    ax.set_xticks(xs if len(xs) <= 16 else xs[::int(np.ceil(len(xs) / 16))])
    ax.legend(ncol=2)
    return _save(fig, path)


def plot_horizontal_bars(labels, values, path, title, x_label, highlight=None):
    """Sorted horizontal bars (one per algorithm); the best is highlighted."""
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(labels) + 1.6))
    cols = [SERIES[1] if labels[i] == highlight else SERIES[0] for i in order]
    ax.barh([labels[i] for i in order], [values[i] for i in order], color=cols, zorder=2)
    for y, i in enumerate(order):
        ax.text(values[i], y, f" {values[i]:.3g}", va="center", fontsize=8, color=TEXT2)
    ax.set_xlabel(x_label)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)
