# Genetic Algorithm (GA) channel selection

Code: `Kartavya/algorithms/combinatorial/ga/` (`config.py`, `ga.py`, `train.py`, `test.py`).
Outputs: `Kartavya/outputs/<category>/ga/png/`.

## 1. The idea from the basics

PCA, ICA and LDA *transform* the 8 channels into new mixed components. A GA instead *selects*:
it keeps K of the original channels unchanged and throws the rest away. The kept values are
the original values, so they can be read directly ("this is NDWI").

The problem is choosing which K. With C channels there are C-choose-K possible subsets.
For 8 channels that is at most 70, but for 13 Sentinel-2 bands plus several indices it
grows to millions, so trying every subset stops being practical. A GA is a search method
borrowed from evolution:

| Biology | Here |
|---|---|
| individual | one candidate subset |
| chromosome | a binary vector of length 8: gene = 1 keeps the channel, 0 drops it |
| fitness | how good the subset is (section 2) |
| population | 12 candidate subsets at a time |
| selection | tournament: draw 3 at random, the fittest becomes a parent |
| crossover | the child keeps the channels **both** parents share, then fills the remaining places at random from channels that **one** parent has |
| mutation | with probability 0.2, swap one kept channel for one dropped channel |
| elitism | the best 2 individuals pass to the next generation unchanged |
| generation | evaluate, select, cross over, mutate, repeat |

Every experiment fixes K, so a chromosome always has exactly K ones, for example
`[0 0 1 1 0 0 0 1]` = keep `RGB:B(B02)`, `False_Color:R(B08)`, `NDWI`. The crossover
and mutation above are designed to preserve that count, so no repair step is needed.

**Termination.** The run stops on whichever of these comes first:
- 40 generations;
- 10 generations without improvement of the best fitness;
- every possible subset has been evaluated (then the best is provably optimal).

## 2. Fitness function and why

```
fitness(S) = mean_R2(S) + 0.5 * min_R2(S) - 0.1 * redundancy(S)
```

- **mean_R2 (information preservation).** Fit a least-squares linear model that predicts
  all 8 channels from the kept ones. Then R2_c = 1 - SSE_c / SST_c is the share of channel
  c's variance that survives the reduction. Its mean over the 8 channels is the term.
  - R2 is a ratio, so it is unit-free: 0–255 display levels and −1..1 indices count equally.
  - Kept channels have R2 = 1 by construction; dropped channels are scored on how well they
    can be rebuilt. This doubles as **reconstruction quality**.
- **min_R2 (worst channel).** Averaging can hide a product that is lost completely. At K=1
  the mean R2 is 0.69, but NDWI keeps only 0.25. This term makes the GA protect the
  weakest channel.
- **redundancy.** Mean |Pearson r| between the kept channels. Two near-copies carry the
  same information. The weight is small (0.1), so this term mostly breaks near-ties in
  favour of diverse subsets.

**No over-fitting to the scoring pixels.**
- R2 is measured on *validation* pixels (30 % of the training sample). The linear model is
  fitted on the other 70 %.
- Test pixels are never seen during selection.
- The final model for each chosen subset is refitted on the full training sample (400k pixels).

**Downstream utility (e.g. classification accuracy) is not used.** No valid ground-truth
labels exist (see LDA.md), and labels derived from the same images would be circular.

## 3. Data facts that shape the result

These are the same 8 channels as PCA/ICA. They are **not** original Sentinel-2 bands:
- 6 are 8-bit display levels of the RGB and False-Color composites;
- 2 are NDVI/NDWI decoded from colour-mapped PNGs.

The composites share two bands:
- `RGB:R(B04)` = `False_Color:G(B04)`, r = 1.000000
- `RGB:G(B03)` = `False_Color:B(B03)`, r = 1.000000

Swapping one copy for the other gives an **identical** fitness, so several subsets tie
for best. Whether the GA succeeded is therefore judged by fitness value, not by channel
names.

There is only one date, so the split is spatial: the top half of the image trains and
the bottom half tests. With 5 dates the scripts switch automatically to a temporal split
(indices 0–2 train, 3–4 test).

## 4. Results (actual run, 2026-09-17, 1058×1024 grid)

### Selected channels (training)

| K | Kept channels | Fitness | mean R2 | min R2 (channel) | Redundancy |
|---|---|---|---|---|---|
| 1 | RGB:B(B02) | 0.816 | 0.693 | 0.246 (NDWI) | 0.000 |
| 2 | FC:B(B03), NDWI | 1.371 | 0.961 | 0.907 (NDVI) | 0.433 |
| 3 | RGB:B(B02), FC:R(B08), NDVI | 1.408 | 0.981 | 0.946 (NDWI) | 0.457 |
| 4 | RGB:R(B04), RGB:B(B02), FC:R(B08), NDWI | 1.432 | 0.996 | 0.984 (NDVI) | 0.561 |
| 5 | RGB:R(B04), RGB:B(B02), FC:R(B08), NDVI, NDWI | 1.436 | 0.998 | 0.992 (B03) | 0.578 |
| 6 | RGB:B, FC:R(B08), FC:G(B04), FC:B(B03), NDVI, NDWI | 1.438 | 1.000 | 1.000 | 0.624 |
| 7 | 6 above + RGB:R(B04) (a duplicate) | 1.434 | 1.000 | 1.000 | 0.662 |

Reading the table:
- **K=2:** B03 (green) and NDWI. NDWI is computed from green and NIR, so together the two
  pin down both bands.
- **K=3:** blue, NIR and NDVI. NDVI adds red, so all four underlying bands are covered.
- **K=6:** every *distinct* band is kept and nothing more, so the reconstruction is lossless.
- **K=7 scores below K=6.** The 7th channel must be a duplicate. It adds no information,
  only redundancy.

### Convergence and verification

Each K was run with 5 seeds, and all subsets were scored exhaustively as a check.

| K | Subsets | Optimal (tied) | Seeds reaching optimum | Mean subsets evaluated per run | Generations |
|---|---|---|---|---|---|
| 1 | 8 | 1 | 5/5 | 8 | 0 (all in first population) |
| 2 | 28 | 2 | 4/5 | 19.6 | 10 |
| 3 | 56 | 1 | 3/5 | 26.4 | 10–14 |
| 4 | 70 | 2 | 3/5 | 26.0 | 10–11 |
| 5 | 56 | 2 | 5/5 | 26.2 | 10–15 |
| 6 | 28 | 4 | 5/5 | 20.2 | 10 |
| 7 | 8 | 2 | 5/5 | 8 | 0 |

- **The best run found the global optimum at every K.**
- **Single runs sometimes stopped early.** At K=3 and K=4, two of five seeds settled on a
  near-optimal subset, with fitness 0.01 lower at K=3 and 0.001 lower at K=4. This is
  why several seeds are used.
- **Cost.** GA run times were 0.3–1.9 s per run, with peak memory of 45–72 MB. The full
  train.py took 72 s, including image loading (9 s, peak 523 MB) and the exhaustive check.

### Test region (bottom half, 541,531 pixels, saved subsets reused)

| K | Reduction | PSNR (dB) | SSIM | RMSE | Reduced .npy | GA vs PCA PSNR / SSIM |
|---|---|---|---|---|---|---|
| 1 | 87.5 % | 20.16 | 0.778 | 0.098 | 2.07 MB | 21.36 / 0.813 |
| 2 | 75.0 % | 27.31 | 0.953 | 0.043 | 4.13 MB | 28.88 / 0.968 |
| 3 | 62.5 % | 30.71 | 0.966 | 0.029 | 6.20 MB | 35.23 / 0.985 |
| 4 | 50.0 % | 38.29 | 0.993 | 0.012 | 8.27 MB | 39.91 / 0.995 |
| 5 | 37.5 % | 38.91 | 0.995 | 0.011 | 10.33 MB | 45.29 / 0.997 |
| 6 | 25.0 % | lossless | 1.000 | 0.000 | 12.40 MB | lossless |
| 7 | 12.5 % | lossless | 1.000 | 0.000 | 14.46 MB | lossless |

- The original 8-channel float32 image is 16.53 MB.
- The model is tiny: (K+1)×8 weights plus K indices.
- Kept channels come back exactly (maximum error about 1e-15).
- Reducing takes 1–3 ms and rebuilding 14–27 ms.

## 5. Verdict

- **Quality.** GA selection is within 1–2 dB of PCA at K = 1, 2 and 4, behind by 4.5 dB at
  K = 3 and by 6.4 dB at K = 5, and equal (lossless) at K ≥ 6. This is expected: PCA is
  the best possible *linear* K-dimensional summary, and a subset of original channels is a
  constrained special case of it.
- **What GA gains.** The kept data are real, interpretable channels; there is no
  transform to apply before use; a reduced file can be read by anyone; and the selection
  says which products are worth storing.
- **Practical answer here.** Six channels reconstruct everything, because two
  channels are duplicates. The two copies can be dropped with no loss.

## 6. Limitations

- There is only one date, so the spatial split tests generalisation within one scene, not
  across seasons.
- With 8 channels, exhaustive search is cheap, so a GA is not *needed* here. It is used and
  verified here so that the same code scales to larger sets (raw bands plus indices) where
  exhaustive search is impossible.
- Downstream utility could not be evaluated, because no labels exist.
- Reduced data are stored as float32 to match PCA/ICA. Because selected channels are
  original 8-bit-derived values, they could be stored more compactly; this was not
  measured.
- The raw-band experiment (`--dataset raw`) exits cleanly, because no original band files
  are present. No bands are fabricated.

## 7. Run

```bash
pip install -r Kartavya/requirements.txt
python Kartavya/algorithms/combinatorial/ga/train.py
python Kartavya/algorithms/combinatorial/ga/test.py
```

Weights, GA settings, seeds and K values are in `config.py`.
