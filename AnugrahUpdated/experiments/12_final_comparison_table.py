"""
FINAL COMPARISON TABLE: one row per (scene, method) for PCA, Isomap, Kernel PCA, LLE (and Isomap+SMACOF on Raichur).
Reads saved results only (no refitting):
  - neighbourhood metrics, time, memory, 5-NN spatial-block accuracy : results/benchmarks/<scene>_all.json (script 07)
  - Raichur neighbourhood metrics                                      : <technique>/results/*real_raichur.json
  - reconstruction (PSNR/SSIM/SAM) and SVM OA, k = 3, LINEAR decoder   : results/team_comparable/<scene>/team_comparable_all.csv (script 09)
Missing pieces are shown as '-' (never guessed). Writes results/FINAL_COMPARISON_TABLE.md and .csv.
Cautions: Raichur numbers were made on the author's laptop, the other scenes on a different machine, so compare
times only within a scene; SMACOF used 2,000 pixels (not comparable); 'linear decoder' is an addition made in this
study, not part of the published methods.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["indian_pines", "salinas", "ksc", "botswana"]
METHODS = [("PCA baseline", "PCA (fit on sample)"), ("Isomap", "Isomap + linear decoder"),
           ("Kernel PCA", "Kernel PCA + linear decoder"), ("LLE", "LLE + linear decoder")]
RAICHUR = {"Isomap": "01_Isomap/results/isomap_real_raichur.json",
           "Isomap with SMACOF": "02_IsomapSMACOF/results/isomap_smacof_real_raichur.json",
           "Kernel PCA": "03_KernelPCA/results/kernel_pca_real_raichur.json",
           "LLE": "04_LLE/results/lle_real_raichur.json"}


def f(v, nd):
    return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def main():
    rows = []
    for name, rel in RAICHUR.items():
        p = ROOT / rel
        if not p.exists():
            continue
        r = json.loads(p.read_text()); m = r["metrics"]
        rows.append(["raichur", name, r.get("n_pixels_used", "-"), f(m["trustworthiness"], 4), f(m["neighborhood_preservation"], 3),
                     "-", "-", "-", "-", "-", f(m["fit_time_seconds"], 2), f(m["peak_memory_mb"], 1)])
    for sc in SCENES:
        bj = ROOT / "results" / "benchmarks" / f"{sc}_all.json"
        tc = ROOT / "results" / "team_comparable" / sc / "team_comparable_all.csv"
        b = json.loads(bj.read_text()) if bj.exists() else {}
        t = pd.read_csv(tc) if tc.exists() else None
        for bname, tname in METHODS:
            r = b.get(bname)
            m = r["metrics"] if r and "metrics" in r else {}
            c = r.get("classification_5nn") if r else None
            psnr = ssim = sam = oa = None
            if t is not None:
                x = t[(t["algorithm"] == tname) & (t["k"] == 3)]
                if len(x):
                    x = x.iloc[0]; psnr, ssim, sam = x.get("PSNR_dB"), x.get("SSIM"), x.get("sam_deg"); oa = x.get("OA")
            rows.append([sc, bname, r["n_pixels_used"] if r else "-", f(m.get("trustworthiness"), 4),
                         f(m.get("neighborhood_preservation"), 3),
                         f(c["spatial_block_5fold"]["overall_accuracy"], 3) if c else "-",
                         f(oa, 1) if oa is not None else "-", f(psnr, 2), f(ssim, 3), f(sam, 2),
                         f(m.get("fit_time_seconds"), 2), f(m.get("peak_memory_mb"), 1)])
    hdr = ["Scene", "Method", "Pixels", "Trust.", "Nbr. pres.", "5-NN OA (spatial)", "SVM OA % (k=3)",
           "PSNR dB (k=3)", "SSIM (k=3)", "SAM deg (k=3)", "Time s", "Memory MB"]
    out = ROOT / "results"
    pd.DataFrame(rows, columns=hdr).to_csv(out / "FINAL_COMPARISON_TABLE.csv", index=False)
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    md += ["", "'-' = not run or not applicable (Raichur has no labels and no reconstruction run; Botswana has no labels; KSC and Botswana",
           "were not run through script 09). PSNR/SSIM/SAM/SVM use the LINEAR decoder (PCA: exact inverse) fitted on the same sample.",
           "SMACOF: 2,000 pixels, Raichur only, not converged; do not rank it against the 8,000-pixel rows.",
           "Raichur timings/memory are from the author's laptop; other scenes from a one-core development machine."]
    (out / "FINAL_COMPARISON_TABLE.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
