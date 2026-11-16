"""
Live demo for the mentor: ranking-based band selection, interactive.

Run from the repo folder:
    pip install streamlit
    streamlit run darshita/app.py

A browser tab opens. Move the K slider and everything recomputes live.
"""
import os
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat

from ranking import entropy_scores, mvpca_scores, top_k, variance_scores   # our own file

IMAGE_FILE = "data/darshita/Indian_pines_corrected.mat"
IMAGE_URL = "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines_corrected.mat"
COLORS = {"Variance": "#2a78d6", "Entropy": "#eb6834", "MVPCA": "#1baf7a", "Uniform": "#8a8986"}

st.set_page_config(page_title="Band selection demo", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_image():
    if not os.path.exists(IMAGE_FILE):
        os.makedirs(os.path.dirname(IMAGE_FILE), exist_ok=True)
        urllib.request.urlretrieve(IMAGE_URL, IMAGE_FILE)
    return loadmat(IMAGE_FILE)["indian_pines_corrected"].astype(np.float32)


image = load_image()
H, W, B = image.shape
raw = image.reshape(-1, B)
low, high = raw.min(axis=0), raw.max(axis=0)
pixels = (raw - low) / (high - low)                       # every band scaled to 0..1
order = np.random.default_rng(0).permutation(len(pixels))
calib, test = order[: len(pixels) // 10], order[len(pixels) // 10:]


@st.cache_data
def select_all(k):
    """Run every method for this k; return chosen bands, scores and runtime (ms)."""
    methods = {"Variance": lambda: variance_scores(raw), "Entropy": lambda: entropy_scores(raw),
               "MVPCA": lambda: mvpca_scores(raw, k)}
    out, scores, times = {}, {}, {}
    for name, score_fn in methods.items():
        t = time.perf_counter()
        s = score_fn()
        out[name] = top_k(s, k)
        times[name] = 1000 * (time.perf_counter() - t)
        scores[name] = s
    out["Uniform"] = np.linspace(0, B - 1, k).round().astype(int).tolist()
    times["Uniform"] = 0.0
    return out, scores, times


def rebuild(bands):
    X = np.c_[pixels[:, bands], np.ones(len(pixels))]
    w, *_ = np.linalg.lstsq(X[calib], pixels[calib], rcond=None)
    return np.clip(X @ w, 0, 1)


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def sam(a, b):
    cos = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()


def redundancy(bands):
    k = len(bands)
    c = np.abs(np.corrcoef(pixels[:, bands], rowvar=False))
    return (c.sum() - k) / (k * (k - 1))


@st.cache_data
def evaluate(k):
    chosen, _, times = select_all(k)
    rows, rebuilt = [], {}
    for name, bands in chosen.items():
        r = rebuild(bands)
        rebuilt[name] = r
        rows.append({"Method": name, "Family": "Baseline" if name == "Uniform" else "Ranking",
                     "PSNR (dB) ↑": psnr(pixels[test], r[test]), "SAM (°) ↓": sam(pixels[test], r[test]),
                     "Redundancy ↓": redundancy(bands), "Time (ms)": times[name],
                     "Bands chosen": ", ".join(map(str, bands))})
    return pd.DataFrame(rows), rebuilt


@st.cache_data
def psnr_curve(ks):
    data = {m: [] for m in COLORS}
    for k in ks:
        chosen, _, _ = select_all(k)
        for m, bands in chosen.items():
            data[m].append(psnr(pixels[test], rebuild(bands)[test]))
    return data


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Controls")
K = st.sidebar.slider("K: bands the satellite may send", 2, 40, 10)
st.sidebar.metric("Data sent", f"{100 * K / B:.1f}%", f"{K} of {B} bands")
show_band = st.sidebar.slider("Band to inspect / rebuild", 0, B - 1, 99)
st.sidebar.caption("Dataset: Indian Pines, 145 × 145 pixels, 200 bands (AVIRIS).")

st.title("Band selection for satellite downlinking")
st.write("**Part 1: ranking-based methods.** Each band gets a score on its own; the top K are sent. "
         "On the ground all 200 bands are rebuilt from those K (linear regression, fitted on 10% of the "
         "pixels, scored on the other 90%).")

tab1, tab2, tab3 = st.tabs(["1 · The data", "2 · Which bands each method picks", "3 · Results and comparison"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        for ax, b in zip(axes, [10, show_band, 180]):
            ax.imshow(image[:, :, b], cmap="gray")
            ax.set_title(f"Band {b}")
            ax.axis("off")
        fig.suptitle("Same place, three different bands")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for r, c in [(20, 20), (70, 70), (120, 40)]:
            ax.plot(image[r, c, :], label=f"pixel ({r}, {c})")
        ax.set_xlabel("Band")
        ax.set_ylabel("Value")
        ax.set_title("Each pixel is a spectrum")
        ax.legend()
        st.pyplot(fig)
    a = np.corrcoef(raw[:, 50], raw[:, 51])[0, 1]
    b_ = np.corrcoef(raw[:, 10], raw[:, 180])[0, 1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Image size", f"{H}×{W}×{B}")
    m2.metric("Correlation, band 50 vs 51", f"{a:.3f}", "near copies")
    m3.metric("Correlation, band 10 vs 180", f"{b_:.3f}")
    st.info("Neighbouring bands are almost identical, so most of them can be dropped before downlinking.")

# ------------------------------------------------------------------ tab 2
chosen, scores, _ = select_all(K)
with tab2:
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for ax, name in zip(axes, ["Variance", "Entropy", "MVPCA"]):
        ax.plot(scores[name], color="gray", lw=1.1)
        ax.scatter(chosen[name], scores[name][chosen[name]], color=COLORS[name], s=28, zorder=3)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Band number")
    fig.suptitle(f"Score of every band; coloured dots = the {K} bands kept")
    fig.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(11, 2.4))
    for row, name in enumerate(COLORS):
        ax.vlines(chosen[name], row - 0.35, row + 0.35, color=COLORS[name], lw=3)
    ax.set_yticks(range(len(COLORS)))
    ax.set_yticklabels(list(COLORS))
    ax.invert_yaxis()
    ax.set_xlim(-2, B + 2)
    ax.set_xlabel("Band number")
    ax.set_title("Where the chosen bands fall: ranking methods bunch together, Uniform spreads out")
    fig.tight_layout()
    st.pyplot(fig)
    st.warning("Ranking scores each band alone, so the neighbours of a good band are also picked. "
               "They are near copies, which wastes the downlink.")

# ------------------------------------------------------------------ tab 3
with tab3:
    df, rebuilt = evaluate(K)
    best = df.sort_values("PSNR (dB) ↑", ascending=False).iloc[0]
    st.subheader(f"Sending {K} of {B} bands ({100 * K / B:.1f}% of the data)")
    cols = st.columns(len(df))
    for col, (_, r) in zip(cols, df.iterrows()):
        col.metric(r.Method, f"{r['PSNR (dB) ↑']:.2f} dB", f"SAM {r['SAM (°) ↓']:.2f}°", delta_color="off")
    st.dataframe(df.style.format({"PSNR (dB) ↑": "{:.2f}", "SAM (°) ↓": "{:.2f}", "Redundancy ↓": "{:.3f}",
                                  "Time (ms)": "{:.1f}"}), hide_index=True, width="stretch")
    st.success(f"Best at K = {K}: **{best.Method}** with {best['PSNR (dB) ↑']:.2f} dB.")

    fig, axes = plt.subplots(2, len(rebuilt) + 1, figsize=(13, 5.6))
    axes[0, 0].imshow(pixels[:, show_band].reshape(H, W), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Original band {show_band}")
    axes[1, 0].axis("off")
    for i, (name, r) in enumerate(rebuilt.items(), start=1):
        sent = " (sent)" if show_band in chosen[name] else ""
        axes[0, i].imshow(r[:, show_band].reshape(H, W), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"{name}{sent}")
        axes[1, i].imshow(np.abs(pixels[:, show_band] - r[:, show_band]).reshape(H, W),
                          cmap="magma", vmin=0, vmax=0.3)
        axes[1, i].set_title("error")
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Rebuilt band and its error map (brighter = worse)")
    fig.tight_layout()
    st.pyplot(fig)

    ks = [2, 5, 10, 15, 20, 25, 30, 40]
    curve = psnr_curve(ks)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, vals in curve.items():
        ax.plot(ks, vals, marker="o", lw=2, color=COLORS[name], ls="--" if name == "Uniform" else "-",
                label=name)
    ax.axvline(K, color="black", lw=1, ls=":")
    ax.set_xlabel("K, number of bands sent")
    ax.set_ylabel("PSNR (dB), higher is better")
    ax.set_title("Quality vs bands sent (dotted line = current K)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
