"""
Build the side-by-side comparison table: Raichur (real, 6 channels, no labels)
vs Indian Pines (benchmark, 200 bands, with ground truth), for every technique
that has results on both. Reads ONLY saved JSON - no refitting.

Run (project root):  python experiments/06_side_by_side_table.py
Output: results/side_by_side.md and results/side_by_side.csv

Cautions printed into the table itself:
 - Pixel counts differ for SMACOF (2,000) vs the others (8,000); its
   neighbourhood-preservation value is NOT directly comparable.
 - Reconstruction error is deliberately omitted: it is a different quantity
   for each technique (see src/evaluation.py).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAICHUR = {
    "Isomap": "01_Isomap/results/isomap_real_raichur.json",
    "Isomap with SMACOF": "02_IsomapSMACOF/results/isomap_smacof_real_raichur.json",
    "Kernel PCA": "03_KernelPCA/results/kernel_pca_real_raichur.json",
    "LLE": "04_LLE/results/lle_real_raichur.json",
}
IP_ALL = ROOT / "results" / "indian_pines_all.json"
COLS = ["trustworthiness", "continuity", "neighborhood_preservation", "fit_time_seconds", "peak_memory_mb"]


def fmt(v, name):
    if v is None:
        return "n/a"
    if name in ("fit_time_seconds", "peak_memory_mb"):
        return f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    return f"{v:.3f}" if name == "neighborhood_preservation" else f"{v:.4f}"


def main():
    if not IP_ALL.exists():
        print("Run experiments/05_indian_pines_all_techniques.py first.")
        return
    ip = json.loads(IP_ALL.read_text())
    rows = []
    for tech, rel in RAICHUR.items():
        p = ROOT / rel
        r = json.loads(p.read_text()) if p.exists() else None
        i = ip.get(tech)
        rows.append((tech, r, i))
    pca = ip.get("PCA baseline")

    header = ["Technique", "Dataset", "Pixels"] + ["Trust.", "Contin.", "Nbr. pres.", "Time (s)", "Memory (MB)"] \
             + ["5-NN OA (random)", "5-NN OA (spatial)", "5-NN kappa (spatial)"]
    table, csv_rows = [], []
    for tech, r, i in rows:
        if r:
            m = r["metrics"]
            line = [tech, "Raichur", r["n_pixels_used"]] + [fmt(m.get(c), c) for c in COLS] + ["no labels"] * 3
            table.append(line)
        else:
            table.append([tech, "Raichur", "MISSING"] + [""] * 8)
        if i:
            m = i["metrics"]; c = i["classification_5nn"]
            line = [tech, "Indian Pines", i["n_pixels_used"]] + [fmt(m.get(k), k) for k in COLS] + [
                f"{c['random_5fold']['overall_accuracy']:.3f}",
                f"{c['spatial_block_5fold']['overall_accuracy']:.3f}",
                f"{c['spatial_block_5fold']['kappa']:.3f}"]
            table.append(line)
        else:
            table.append([tech, "Indian Pines", "MISSING"] + [""] * 8)
    if pca:
        m = pca["metrics"]; c = pca["classification_5nn"]
        table.append(["PCA baseline", "Indian Pines", pca["n_pixels_used"]] + [fmt(m.get(k), k) for k in COLS] + [
            f"{c['random_5fold']['overall_accuracy']:.3f}", f"{c['spatial_block_5fold']['overall_accuracy']:.3f}",
            f"{c['spatial_block_5fold']['kappa']:.3f}"])
    raw = ip.get("Raw 200 bands (reference)")
    if raw:
        c = raw["classification"]
        table.append(["All 200 bands (no reduction)", "Indian Pines", raw["n_pixels_used"]] + ["-"] * 5 + [
            f"{c['random_5fold']['overall_accuracy']:.3f}", f"{c['spatial_block_5fold']['overall_accuracy']:.3f}",
            f"{c['spatial_block_5fold']['kappa']:.3f}"])

    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(x) for x in row) + " |" for row in table]
    notes = [
        "",
        "Notes:",
        "- SMACOF used 2,000 pixels on both datasets; the others used 8,000. Do not rank SMACOF's neighbourhood preservation against them.",
        "- Raichur uses 6 channels (LST unavailable); Indian Pines uses 200 bands. Metric values across datasets are not on a common scale; compare rankings within a dataset.",
        "- Reconstruction error is omitted: it measures a different thing for each technique.",
        "- 5-NN accuracy is on the 3-D embedding. 'spatial' = whole 29x29-pixel blocks held out; 'random' leaks neighbouring pixels across folds and is optimistic.",
        "- Embeddings are fitted on all sampled pixels (labels never used); this compares methods fairly but is not an out-of-sample deployment test.",
    ]
    out_md = ROOT / "results" / "side_by_side.md"
    out_md.write_text("\n".join(md + notes))
    with open(ROOT / "results" / "side_by_side.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(table)
    print("\n".join(md + notes))
    print(f"\nSaved {out_md}")


if __name__ == "__main__":
    main()
