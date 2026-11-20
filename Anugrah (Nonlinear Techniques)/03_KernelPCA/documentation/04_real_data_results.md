# Kernel PCA - real Raichur data: results

Run by you, on your own machine, against the real `raichur_features.parquet`
(2026-09-27, 8,000-pixel subsample - same sample size as Technique 1's
real-data run, since Kernel PCA's low cost gave no reason to shrink it).
Ran clean on the first data-reading attempt once `PARQUET_PATH` was pointed
at the right folder - no code bugs surfaced this time, unlike Technique 2's
run. (Fixed afterwards anyway: this project's own default `PARQUET_PATH` now
includes `DIP_project` out of the box for every technique's real-data
script, so this manual edit should not be needed again going forward.)

## Headline: the trade-off seen on synthetic data holds up, and both sides of it get MORE pronounced on real data

| Metric | Isomap (8,000 px) | Kernel PCA (8,000 px) | KPCA/Isomap ratio |
|---|---|---|---|
| Trustworthiness | 0.9987 | 0.9945 | 0.996 |
| Continuity | 0.9995 | 0.9987 | 0.999 |
| Neighbourhood preservation (Jaccard) | 0.4265 | 0.2933 | 0.688 |
| Fit time | 14.13 s | 1.75 s | 0.124 (**8.1x faster**) |
| Peak memory | 1,541 MB | 1,028 MB | 0.667 |

Compared directly against the synthetic-data comparison
(`documentation/03_stage4_5_synthetic.md`):

| | Synthetic (1,600 px) | Real (8,000 px) |
|---|---|---|
| Isomap vs. KPCA neighbourhood-preservation gap (absolute) | 0.128 | 0.133 | 
| KPCA's speed advantage over Isomap | 6.5x | **8.1x** |
| KPCA's memory advantage over Isomap | 0.346x (uses 35% as much) | 0.667x (uses 67% as much - advantage SHRINKING) |

**Three honest readings, not one simple verdict:**
1. The neighbourhood-preservation gap (Isomap beats Kernel PCA) is stable in
   absolute terms across both datasets (~0.13 either way) - this looks like
   a genuine, consistent property of the RBF-kernel approach lacking
   Isomap's geodesic-distance mechanism, not a synthetic-data artifact.
2. Kernel PCA's SPEED advantage over Isomap actually grows at real-data
   scale (6.5x -> 8.1x) - a practically important finding for anyone
   choosing a technique under a time budget on a full satellite tile.
3. Kernel PCA's MEMORY advantage shrinks sharply at scale (uses 35% of
   Isomap's memory on 1,600 pixels, but 67% on 8,000) - both methods'
   dominant memory cost is an n x n matrix (geodesic-distance matrix for
   Isomap, kernel/Gram matrix for Kernel PCA), so as n grows, that shared
   O(n^2) term increasingly dominates both, and whatever fixed/smaller
   overhead previously favoured Kernel PCA becomes proportionally less
   significant. Worth projecting forward explicitly: at full-tile pixel
   counts (millions), expect this memory gap to continue narrowing, not
   widening - neither method's memory cost scales favourably, just at
   different constants.

## What we could NOT check on real data, and why - stated rather than skipped silently

Stage 4 on synthetic data found a real limitation: Kernel PCA's 3 output
dimensions, checked against KNOWN true latent variables `t` and `s`, did not
recover `t` at all (0.08 correlation even combining all 3 dimensions - see
`documentation/03_stage4_5_synthetic.md`). On real data, there is no known
ground truth to repeat this exact check against - we cannot say whether the
real embedding similarly "loses" some true underlying physical factor,
because unlike the synthetic cube, we never knew the true number or identity
of real underlying factors to begin with. This is not a gap we can close
with more real-data runs; it is a structural limit of evaluating on real
data at all, and is exactly why Stage 4's synthetic, known-ground-truth
check matters as its own step rather than something only real-data numbers
could tell us. The real embedding's `explained_variance_ratio` (0.646,
slightly higher than synthetic's 0.556) is the closest available real-data
signal, but it describes OVERALL captured variance, not whether that
variance is spread across independent true factors or concentrated the same
way Stage 4 found on synthetic data - a genuinely open question for this
technique that we are flagging rather than guessing an answer to.

## Status
Technique 3 (Kernel PCA) is now complete end-to-end: Stages 1-5, synthetic
and real data. No code bugs found on this run (first real-data success of
the project with zero bug-fixes needed mid-run), though the path-default fix
below benefits every technique going forward.
