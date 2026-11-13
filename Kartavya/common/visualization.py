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


def plot_loadings(components, names, path, n_show=4, title="PCA loadings (what each component is made of)"):
    """Bar chart of the weight of every input channel in the first components (one panel each)."""
    n_show = min(n_show, components.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(3.3 * n_show, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(names))
    for i, ax in enumerate(axes):
        w = components[i]
        ax.barh(y, w, color=[SERIES[0] if v >= 0 else SERIES[1] for v in w], height=0.6)
        ax.axvline(0, color=TEXT2, lw=0.8)
        ax.set_title(f"PC{i + 1}", loc="left")
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


def plot_component_images(reduced, valid, path, title, n_show=3):
    """Each principal-component image in grey, plus PC1-PC3 as a colour composite."""
    n_show = min(n_show, reduced.shape[-1])
    panels = n_show + (1 if reduced.shape[-1] >= 3 else 0)
    fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3.9))
    axes = np.atleast_1d(axes)
    for i in range(n_show):
        axes[i].imshow(_stretch(reduced[..., i:i + 1], valid)[..., 0], cmap="gray")
        axes[i].set_title(f"PC{i + 1}", loc="left")
    if reduced.shape[-1] >= 3:
        axes[-1].imshow(_stretch(reduced[..., :3], valid))
        axes[-1].set_title("PC1-PC3 as RGB", loc="left")
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
