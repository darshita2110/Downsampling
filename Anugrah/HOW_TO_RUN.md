# How to run everything (Windows PowerShell, from the project root)

```powershell
cd "C:\Users\anugr\OneDrive\Desktop\5th sem\DIP\Downsampling\Anugrah"   # the folder that contains src, experiments, 01_Isomap ... (use YOUR folder name)
pip install -r requirements.txt
```
Use Python 3.11. The pinned `requirements.txt` was NOT used in the development sandbox (it had newer scikit-learn),
so if a pinned-version error appears, send me the message. If Raichur loading fails with a pyarrow/fastparquet
error after installing, run `pip install "pyarrow==14.0.2"`.

## 0. Data
Put the `.mat` files in `data/hsi/` (Indian Pines is already there): `Salinas_corrected.mat`, `Salinas_gt.mat`,
`KSC_data.mat`, `KSC_gt.mat`, `Botswana.mat`. Pavia University needs `PaviaU.mat` (the image), which was not
supplied, so it is skipped. Or keep them elsewhere and add `--data-dir "C:\path\to\folder"`.

## 1. Original study (synthetic cube and Raichur) - already finished, reruns only if you want
```powershell
python experiments/00_make_synthetic_dataset.py
python 01_Isomap/experiments/03_isomap_on_synthetic_cube.py
python 02_IsomapSMACOF/experiments/02_on_synthetic_cube.py
python 03_KernelPCA/experiments/02_on_synthetic_cube.py
python 04_LLE/experiments/02_on_synthetic_cube.py
# Raichur (needs DIP_project\data\processed\ML_Dataset\combined\raichur_features.parquet):
python 01_Isomap/experiments/05_isomap_on_real_data.py
python 02_IsomapSMACOF/experiments/03_on_real_data.py
python 03_KernelPCA/experiments/03_on_real_data.py
python 04_LLE/experiments/03_on_real_data.py
```

## 2. Hyperspectral benchmark scenes, neighbourhood metrics (earlier protocol)
```powershell
python experiments/07_benchmarks_all_techniques.py            # add --with-smacof for Isomap+SMACOF (slow)
python experiments/08_benchmark_summary.py                    # -> results/benchmarks_summary.md
```

## 3. Team-comparable pipeline (same metrics as Kartavya and Abhishek)  <- the main new part
```powershell
# Indian Pines first; this also checks that your PCA row matches your teammates' PCA numbers
python experiments/09_team_comparable_pipeline.py --datasets indian_pines --ks 3 5 10
python experiments/09_team_comparable_pipeline.py --datasets salinas --ks 3 5 10
python experiments/09_team_comparable_pipeline.py --datasets ksc botswana --ks 3      # slow, see below
python experiments/10_team_comparable_summary.py --k 3        # -> results/team_comparable_summary.md
```
Outputs per scene in `results/team_comparable/<scene>/`:
- `team_comparable_all.csv` : everything
- `kartavya_format.csv` : columns named like Kartavya's `comparison.csv` (PSNR_dB, SSIM, MSE, RMSE, MAE, storage_saving_pct, compression_ratio, ...); append these rows to his table
- `abhishek_format.csv` : columns named like Abhishek's `metrics.csv` (var_explained, rmse, psnr_db, sam_deg, ARI, NMI, OA, AA, kappa); append to his
- `psnr_vs_k.png`

Useful options: `--no-ssim` (faster), `--sample-size 8000`, `--repeats 5`.

### Time and memory (measured on a single-core sandbox)
- Indian Pines, `--ks 3 5 10`: about 15 minutes. Salinas, same: about 35 minutes. Isomap needs about 1.5 GB RAM at 8,000 pixels.
- KSC (314k pixels) and Botswana (378k pixels): the run did NOT finish in my sandbox. It stalled in KSC while fitting; I did not find the cause (it is not duplicate pixels: all 8,000 sampled pixels are distinct). Results for these two scenes are therefore missing. Try `--sample-size 4000` first, and tell me what you see.

## 4. What to read first when presenting
`documentation/05_team_comparable_pipeline.md` (the pipeline, the protocol mapping, and the results).

## 5. LLE diagnostic (if LLE results look broken)
```powershell
python experiments/11_diagnose_lle.py
```
