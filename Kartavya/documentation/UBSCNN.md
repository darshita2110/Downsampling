# UBS-CNN: Unsupervised Band Selection with a CNN (implemented as BS-Net-Conv)

Code: `Kartavya/algorithms/combinatorial/ubscnn/` (`config.py`, `model.py`, `train.py`, `test.py`).
Outputs: `Kartavya/outputs/<category>/ubscnn/png/`.

## 1. Which method? (the ambiguity, stated first)

**No paper defines an architecture under the exact name "UBS-CNN".** "UBS" is the field's generic
abbreviation for *unsupervised band selection*. Several CNN-based UBS methods exist:

| Method | Reference |
|---|---|
| **BS-Net-Conv** | Cai, Liu & Cai, *IEEE TGRS* 58(3):1969–1984, 2020, arXiv:1904.08269 |
| DARecNet-BS | Roy et al., *IEEE GRSL* 2021 |
| CAE-UBS (concrete autoencoder) | Sun et al., *IEEE TGRS* 2022 |
| Masked convolutional autoencoder | *IEEE GRSL* 2022 |

**BS-Net-Conv is implemented**, because it is:
- the most cited of these;
- fully unsupervised;
- CNN-based;
- a direct band ranker;
- fully specified by the paper together with the authors' public code (github.com/AngryCai/BS-Nets).

It is called "BS-Net-Conv, used as the UBS-CNN method" everywhere. No architecture was invented.

## 2. The CNN from the basics

**What a convolution is.** A small window (here 3×3 pixels × all channels) slides over the
image. At each position it computes a weighted sum, and those weights are learned. Stacking
convolutions lets the network detect spatial-spectral patterns.

**Input representation.**
- Patches of 16×16 pixels × 8 channels, with values in [0, 1].
- Patches come only from the **training** region (top half of the image).
- A patch is kept only if all of its pixels are valid.

**Architecture** (paper Table IV, plus the authors' code where the paper is silent):

```
x (8×16×16) → BatchNorm → x_n
Band Attention Module:  Conv3×3(64, valid)+BN+ReLU → GlobalAvgPool → FC128 ReLU → FC8 Sigmoid → w (8 weights in 0..1)
Band re-weighting:      z = x_n ⊗ w        (every channel multiplied by its weight)
Reconstruction network: Conv3×3(128)+BN+ReLU → Conv3×3(64)+BN+ReLU → ConvT3×3(64)+BN+ReLU
                        → ConvT3×3(128)+BN+ReLU → Conv1×1(8) Sigmoid → x̂ (8×16×16)
```

This network has 209,888 parameters. The paper reports 590,288 because its data had
200 bands rather than 8.

**Loss function** (paper Eq. 10, matching the code):

```
L = MSE(x, x̂) + λ · Σ_patches Σ_bands |w|,   λ = 0.01
```

The network must rebuild the whole patch while keeping the band weights small. The idea is
that only bands carrying unique information keep a large weight.

**Selection** (paper Eq. 11): average w over all training patches, then keep the K bands
with the largest average weight. One trained network gives a ranking, so every K = 1–7 is
read off the same ranking.

**Training settings.**

| Setting | Value | Source |
|---|---|---|
| Optimiser | Adam | paper and code |
| Learning rate | 1e-4 | code (the paper's table says 2e-3) |
| Batch size | 32 | code |
| Epochs | 100 | paper and code |
| Seeds | 42, 43, 44 | the final ranking averages their weights |

**Deviations, and why.**
- **Patch stride 16 instead of 1.** Stride 1 would give about 500,000 overlapping patches,
  which is hours of CPU training. Stride 16 gives 2,112 non-overlapping patches that still
  cover the whole training half; 10 % of them (211) are held out to monitor over-fitting.
- **Rebuilding the image after selection.** BS-Net's decoder rebuilds the image from *all*
  re-weighted bands, not from the selected subset alone. So a training-fitted linear model
  (the same one used for GA/PSO/ACO/SA) rebuilds the image from the K selected channels.
  This makes the methods comparable.

## 3. Data notes

These are the same 8 channels as the other methods, and they are **not** original
Sentinel-2 bands:
- 6 are RGB/False-Color display levels;
- 2 are NDVI/NDWI decoded from colour maps.

Two pairs are exact duplicates (B04 and B03 appear in both composites). There is one date,
so the split is spatial: top half trains, bottom half tests.

## 4. Actual training results (3 seeds × 100 epochs, CPU, 2 threads)

### Training cost

- About 4 s per epoch, about 6.3 min per network, 19 min in total.
- The process's peak memory was about 1.3 GB.
- Training MSE fell from 0.035 to 0.0022.

### Held-out patches look unstable

The held-out patch MSE fluctuated between about 0.02 and 0.11 instead of tracking the
training MSE. The likely cause is that the batch-normalisation running statistics become
unreliable once the re-weighted input is tiny (see below).

### The band weights collapse

All 8 average weights fell to almost 0. They print as 0.000 at three decimals, and only
their tiny relative differences remain.

This is a property of the published design:
- The L1 penalty is summed over the whole batch (0.01 × 32 patches × 8 bands), so it
  outweighs the reconstruction error.
- The convolution right after the re-weighting is followed by batch normalisation, which
  rescales its output, so shrinking every weight together costs almost nothing in
  reconstruction.
- Only the *relative* sizes of the weights can therefore carry a ranking.

### The ranking is unstable across seeds

| Seed | Ranking (most to least important) |
|---|---|
| 42 | RGB:R(B04) > FC:B(B03) > NDWI > FC:R(B08) > NDVI > RGB:B > FC:G(B04) > RGB:G(B03) |
| 43 | FC:G(B04) > FC:R(B08) > NDWI > FC:B(B03) > RGB:B > NDVI > RGB:R(B04) > RGB:G(B03) |
| 44 | RGB:R(B04) > FC:R(B08) > RGB:B > RGB:G(B03) > NDWI > FC:G(B04) > NDVI > FC:B(B03) |
| Consensus | RGB:R(B04) > FC:G(B04) > FC:R(B08) > NDWI > FC:B(B03) > RGB:B > NDVI > RGB:G(B03) |

### Selections and their quality

Each selection was scored with the objective used by GA/PSO/ACO/SA, for evaluation only.

| K | Selected (consensus top-K) | Objective | Best possible | Rank among all subsets | Seeds with same top-K |
|---|---|---|---|---|---|
| 1 | RGB:R(B04) | 0.797 | 0.814 | 4/8 | 2/3 |
| 2 | RGB:R(B04), FC:G(B04) | 0.697 | 1.371 | **28/28 (worst)** | 0/3 |
| 3 | + FC:R(B08) | 1.281 | 1.408 | 46/56 | 0/3 |
| 4 | + NDWI | 1.387 | 1.432 | 31/70 | 0/3 |
| 5 | + FC:B(B03) | 1.421 | 1.436 | 15/56 | 0/3 |
| 6 | + RGB:B(B02) | 1.422 | 1.438 | 11/28 | 0/3 |
| 7 | + NDVI | 1.434 | 1.434 | 1/8 | 2/3 |

At K=2 the CNN picks **both copies of B04** (r = 1.0). That is the worst possible pair,
because the attention ranks each band on its own and nothing penalises redundancy between
the bands it picks.

### Test

`test.py` (written and lint-checked, **not run** here) does no training. It:
- applies the saved selections and linear reconstruction to the test half;
- reports PSNR, SSIM, RMSE, MAE, reduction %, storage and timing;
- runs the trained networks forward on the test patches to report test-region attention
  weights and the network's own reconstruction MSE;
- compares the results with PCA, GA, PSO, ACO and SA.

## 5. Verdict

The paper evaluated BS-Net-Conv on hyperspectral images with about 200 narrow,
gradually changing bands. On these 8 channels it is **not reliable**:
- The weights collapse because of the batch-normalised design.
- The rankings change from seed to seed.
- It does not handle duplicate channels.

For this dataset, the search methods with an explicit redundancy term (GA, PSO, ACO, SA)
find the optimal subsets; BS-Net-Conv does not.

**Possible fixes, not implemented** (each would change the published method):
- average the L1 term over the batch instead of summing it;
- remove the batch normalisation after the re-weighting, or normalise the weights;
- add an explicit redundancy penalty.

## 6. Run

```bash
pip install -r Kartavya/requirements.txt                    # includes torch (CPU is enough)
python Kartavya/algorithms/combinatorial/ubscnn/train.py    # ~20 min on 2 CPU cores; --epochs 20 for a quick run
python Kartavya/algorithms/combinatorial/ubscnn/test.py     # evaluation only
```

**Saved outputs:**

| Folder | Contents |
|---|---|
| `models/ubscnn/png/` | `bsnet_conv_seed*.pt` (network weights), `ubscnn_K{k}.npz` (selection plus reconstruction weights) |
| `metrics/ubscnn/png/` | `training_history.csv` (loss and weights per epoch), `channel_ranking.csv`, `seed_rankings.csv`, `selected_features.csv` |
| `graphs/ubscnn/png/` | `training_loss.png`, `band_weight_evolution.png`, `final_band_weights.png`, `selected_channels.png`, `objective_vs_optimum.png` |
