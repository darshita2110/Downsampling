"""
Collect results/team_comparable/<dataset>/team_comparable_all.csv into one table at the primary k (default 3)
and one markdown file. Reads saved CSVs only.
Output: results/team_comparable_summary.md / .csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["indian_pines", "salinas", "ksc", "botswana"]
COLS = ["PSNR_dB", "PSNR_dB_heldout", "SSIM", "sam_deg", "OA", "AA", "kappa", "ARI", "NMI",
        "neighborhood_preservation", "storage_saving_pct", "compression_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    frames = []
    for ds in ORDER:
        p = ROOT / "results" / "team_comparable" / ds / "team_comparable_all.csv"
        if p.exists():
            d = pd.read_csv(p)
            d = d[(d["k"] == a.k) & (d["algorithm"] != "Full bands (reference)")]
            frames.append(d)
    if not frames:
        print("No team_comparable results found; run experiments/09 first.")
        return
    d = pd.concat(frames)
    keep = ["dataset", "algorithm", "k"] + [c for c in COLS if c in d.columns]
    d = d[keep].round(3)
    d.to_csv(ROOT / "results" / "team_comparable_summary.csv", index=False)
    lines = ["| " + " | ".join(keep) + " |", "|" + "---|" * len(keep)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in d.values]
    notes = ["", f"k = {a.k}. 'PCA (fit on all pixels)' reproduces the teammates' PCA/PPCA protocol.",
             "OA/AA/kappa: RBF-SVM, random 10% per class (Abhishek's protocol), %. PSNR/SSIM: Kartavya's [0,1] protocol.",
             "'+ knn decoder' rows must be compared with 'PCA (fit on sample) + knn decoder', which uses the same decoder.",
             "ARI/NMI depend on k-means randomness and the scikit-learn version; treat as indicative (about +/-0.03)."]
    (ROOT / "results" / "team_comparable_summary.md").write_text("\n".join(lines + notes))
    print("\n".join(lines + notes))


if __name__ == "__main__":
    main()
