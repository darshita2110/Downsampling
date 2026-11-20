"""
EASY COMPARISON: turns the saved team-comparable results into tables a reader can follow, and counts which method
wins. Reads saved results only. Output: results/EASY_COMPARISON.md and .csv

Methods compared (all with the LINEAR decoder so the comparison is fair; PCA uses its exact inverse):
  PCA, Isomap, Kernel PCA, LLE.  (Isomap+SMACOF cannot embed new pixels, so it is not in this table.)
Measures: PSNR (rebuild quality, higher is better), SSIM (higher), SAM (angle error, LOWER is better),
          SVM OA % (can crops still be told apart, higher), neighbourhood preservation (higher).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENES = {"indian_pines": "Indian Pines", "salinas": "Salinas"}
NAMES = {"PCA (fit on sample)": "PCA", "Isomap + linear decoder": "Isomap",
         "Kernel PCA + linear decoder": "Kernel PCA", "LLE + linear decoder": "LLE"}
ORDER = ["PCA", "Isomap", "Kernel PCA", "LLE"]
MEASURES = [("PSNR_dB", "PSNR (dB)", True), ("SSIM", "SSIM", True), ("sam_deg", "SAM (deg)", False),
            ("OA", "SVM accuracy (%)", True), ("neighborhood_preservation", "Neighbourhood preservation", True)]


def load():
    frames = []
    for sc, label in SCENES.items():
        p = ROOT / "results" / "team_comparable" / sc / "team_comparable_all.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d["algorithm"].isin(NAMES)].copy()
        d["method"] = d["algorithm"].map(NAMES); d["scene"] = label
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    d = load()
    ks = sorted(d["k"].unique())
    md = ["# Easy comparison: PCA vs Isomap vs Kernel PCA vs LLE", "",
          "Every number comes from the saved runs (one random sample of 8,000 pixels, seed 42). 'k' = how many numbers are kept per pixel.", ""]
    rows = []
    for key, title, hi in MEASURES:
        md += [f"## {title}  ({'higher' if hi else 'LOWER'} is better)", ""]
        for sc in d["scene"].unique():
            x = d[d["scene"] == sc].pivot_table(index="method", columns="k", values=key).reindex(ORDER)
            best = x.max() if hi else x.min()
            hdr = ["Method"] + [f"k={k}" for k in x.columns]
            md += [f"**{sc}**", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for m in ORDER:
                cells = []
                for k in x.columns:
                    v = x.loc[m, k]
                    s = "-" if pd.isna(v) else (f"{v:.1f}" if key in ("OA",) else f"{v:.3f}" if key in ("SSIM", "neighborhood_preservation") else f"{v:.2f}")
                    if not pd.isna(v) and v == best[k]:
                        s = f"**{s}**"
                    cells.append(s)
                md.append(f"| {m} | " + " | ".join(cells) + " |")
            md.append("")
    # win counts
    wins = {m: 0 for m in ORDER}; total = 0
    detail = []
    for key, title, hi in MEASURES:
        for sc in d["scene"].unique():
            for k in ks:
                x = d[(d["scene"] == sc) & (d["k"] == k)].set_index("method")[key].reindex(ORDER)
                if x.isna().all():
                    continue
                w = x.idxmax() if hi else x.idxmin()
                wins[w] += 1; total += 1
    md += ["## Scoreboard: how many of the contests each method won", "",
           f"A contest = one measure, one scene, one k ({total} contests in total).", "",
           "| Method | Contests won |", "|---|---|"] + [f"| {m} | {wins[m]} of {total} |" for m in ORDER] + [""]
    # average gap to PCA
    md += ["## Average gap to PCA (positive = better than PCA)", "",
           "| Method | PSNR (dB) | SSIM | SVM accuracy (points) | Neighbourhood preservation |", "|---|---|---|---|---|"]
    for m in ORDER[1:]:
        gaps = []
        for key in ("PSNR_dB", "SSIM", "OA", "neighborhood_preservation"):
            g = []
            for (sc, k), grp in d.groupby(["scene", "k"]):
                s = grp.set_index("method")[key]
                if m in s and "PCA" in s and not pd.isna(s[m]) and not pd.isna(s["PCA"]):
                    g.append(s[m] - s["PCA"])
            gaps.append(sum(g) / len(g) if g else float("nan"))
        md.append(f"| {m} | {gaps[0]:+.2f} | {gaps[1]:+.3f} | {gaps[2]:+.1f} | {gaps[3]:+.3f} |")
    md.append("")
    # cost table
    md += ["## What each method costs (8,000-pixel fit, development machine, single core)", "",
           "| Method | Fit time (s), Indian Pines | Memory (MB) | Exact way back to the bands? | Needs a stored sample on the receiver to embed new pixels? |", "|---|---|---|---|---|"]
    try:
        b = json.loads((ROOT / "results" / "benchmarks" / "indian_pines_all.json").read_text())
        mp = {"PCA": "PCA baseline", "Isomap": "Isomap", "Kernel PCA": "Kernel PCA", "LLE": "LLE"}
        extra = {"PCA": ("Yes", "No (just 3 x 200 numbers)"), "Isomap": ("No, needs a learned decoder", "Yes"),
                 "Kernel PCA": ("No, needs a learned decoder", "Yes"), "LLE": ("No, needs a learned decoder", "Yes")}
        for m in ORDER:
            r = b[mp[m]]["metrics"]
            md.append(f"| {m} | {r['fit_time_seconds']:.1f} | {r['peak_memory_mb']:.0f} | {extra[m][0]} | {extra[m][1]} |")
    except Exception as exc:
        md.append(f"| (cost table unavailable: {exc}) | | | | |")
    md += ["", "LLE uses the dense solver here; with the default solver on our pinned libraries it broke on Indian Pines (see HOW_TO_RUN.md section 4).",
           "Times and memory are from one machine and one run each."]
    (ROOT / "results" / "EASY_COMPARISON.md").write_text("\n".join(md))
    d.to_csv(ROOT / "results" / "EASY_COMPARISON.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
