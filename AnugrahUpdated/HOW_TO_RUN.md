# How to run everything (Windows PowerShell)

Open PowerShell in the project folder, i.e. the folder that contains `src`, `experiments`, `01_Isomap`, ...
(on your machine: `C:\Users\anugr\OneDrive\Desktop\5th sem\DIP\Downsampling\Anugrah`). All scripts also work if you are
inside `experiments\`; then drop the `experiments/` part of the path. Type `python ...` (not `run python ...`).

```powershell
pip install -r requirements.txt
```
`requirements.txt` includes `scikit-image` (needed only for SSIM). If you skip it, scripts still run and print
`SSIM skipped`.

## 0. Data
- Indian Pines is already in `data/hsi/`.
- For the other scenes put these files in `data/hsi/` (or pass `--data-dir "C:\path\to\folder"`):
  `Salinas_corrected.mat`, `Salinas_gt.mat`, `KSC_data.mat`, `KSC_gt.mat`, `Botswana.mat`.
  Pavia University needs `PaviaU.mat` (the image), which was not supplied, so it is skipped.
- Raichur real data (`raichur_features.parquet`) was NOT found anywhere under `...\5th sem\DIP` on your machine
  when checked. The Raichur results for all four techniques are already saved as JSON in each technique's `results/`
  folder, so you do not need to re-run them. To re-run them you must download the file again
  (see `docs/DATA_ACCESS_GUIDE.md` in the repo).

## 1. Original study (synthetic cube; Raichur only if the parquet file exists)
```powershell
python experiments/00_make_synthetic_dataset.py
python 01_Isomap/experiments/03_isomap_on_synthetic_cube.py
python 02_IsomapSMACOF/experiments/02_on_synthetic_cube.py
python 03_KernelPCA/experiments/02_on_synthetic_cube.py
python 04_LLE/experiments/02_on_synthetic_cube.py
```

## 2. Hyperspectral scenes, neighbourhood metrics
```powershell
python experiments/07_benchmarks_all_techniques.py            # add --with-smacof for Isomap+SMACOF (slow)
python experiments/08_benchmark_summary.py                    # -> results/benchmarks_summary.md
```
`08` reads whatever JSON files exist in `results/benchmarks/`. If you only have some scenes' `.mat` files, the other
scenes' rows come from the JSON files that were shipped with this folder (generated on the development machine).

## 3. Team-comparable pipeline (same metrics as Kartavya and Abhishek)
```powershell
python experiments/09_team_comparable_pipeline.py --datasets indian_pines --ks 3 5 10
python experiments/09_team_comparable_pipeline.py --datasets salinas --ks 3 5 10
python experiments/09_team_comparable_pipeline.py --datasets ksc botswana --ks 3 --sample-size 4000   # may be slow
python experiments/10_team_comparable_summary.py --k 3        # -> results/team_comparable_summary.md
```
Check first: the row "PCA (fit on all pixels)" on Indian Pines at k = 3 must show PSNR 28.15 dB, SSIM 0.8786,
OA 64.9% (these equal your teammates' PCA numbers).

Outputs per scene in `results/team_comparable/<scene>/`:
`team_comparable_all.csv` (everything), `kartavya_format.csv` (column names like Kartavya's `comparison.csv`),
`abhishek_format.csv` (column names like Abhishek's `metrics.csv`), `psnr_vs_k.png`.
Options: `--no-ssim`, `--sample-size 8000`, `--repeats 5`.

## 3b. THE FINAL COMPARISON TABLE
```powershell
python experiments/12_final_comparison_table.py
```
Writes `results/FINAL_COMPARISON_TABLE.md` and `.csv`: one row per scene and method (PCA, Isomap, Kernel PCA, LLE; SMACOF on
Raichur) with trustworthiness, neighbourhood preservation, 5-NN spatial accuracy, SVM accuracy, PSNR, SSIM, SAM, time and
memory. It only reads saved results, so run scripts 07 and 09 first. Missing cells show `-`; nothing is guessed.

## 4. LLE solver note
On your machine (numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2) LLE with the default `eigen_solver="auto"` gave a
degenerate embedding on Indian Pines (trustworthiness 0.54; the same data gave 0.94 with `eigen_solver="dense"`).
The hyperspectral scripts (05, 07, 09) therefore use `dense`. The synthetic and Raichur LLE scripts still use `auto`;
their results were checked and are not affected. `python experiments/11_diagnose_lle.py` reproduces the comparison.

## 5. Pushing to GitHub
```powershell
git checkout -b nonlinear-final
git add -A
git status          # check that no large .mat or .parquet data files are listed except data/hsi/Indian_pines_*.mat
git commit -m "Non-linear feature extraction: Techniques 1-4, hyperspectral benchmarks, team-comparable pipeline"
git push -u origin nonlinear-final
```
Then open a pull request on GitHub. Do not `git add` the large Salinas, KSC or Botswana files.

## 6. What to read first when presenting
`documentation/05_team_comparable_pipeline.md`, then `results/team_comparable_summary.md`.
