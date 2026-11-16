# Kernel PCA - Stage 4 (synthetic cube) and Stage 5 (evaluation)

Script: `experiments/02_on_synthetic_cube.py`. `n_components=3`, RBF kernel
with scikit-learn's default `gamma` (`1/n_channels` = 1/7 ~ 0.143), seed=42,
standardised channels - same dataset/settings pattern as Technique 1/2.

## A contrast with Stage 2's negative result, worth being precise about

Stage 2 found RBF Kernel PCA could not unroll a pure Swiss roll at any
`gamma`. On THIS dataset (our satellite-like synthetic cube, not a pure
Swiss roll), Kernel PCA performs much better: trustworthiness 0.992,
continuity 0.997 - close to Isomap's 0.999/0.999, not far behind. This is an
important, concrete lesson rather than a contradiction: Stage 2's negative
result is specific to data that genuinely requires "walking around" a
surface (by construction, a Swiss roll does). Our synthetic cube's curvature,
while real (it came from a Swiss-roll-style construction too, see
`experiments/00_make_synthetic_dataset.py`), is evidently mild enough, or
differently shaped, that RBF's local-similarity approach still captures most
of the local structure. The lesson for the paper: a method's weakness on one
canonical toy example does not automatically carry over to every dataset -
worth testing on the actual data of interest, which is exactly why Stage 4
exists as a separate step from Stage 2.

## A real limitation, found and verified rather than assumed

`visualisations/02b_latent_recovery.png`: all three of Kernel PCA's output
dimensions correlate most strongly with latent `s` (0.29, -0.81, 0.26) and
weakly with `t`. For Isomap and SMACOF, a similarly weak single-dimension
correlation with `t` turned out to be a rotation artifact - the cyclic
latent was still perfectly recoverable from a joint combination of two
dimensions. We checked whether the same explanation applies here: a linear
regression predicting `t` from all three Kernel PCA dimensions together
achieves Spearman correlation **0.08** - no better than chance, not 0.9+
like Isomap/SMACOF's rotated recovery.

**This is a real, different finding, not the same rotation story again.**
Kernel PCA's 3 output dimensions have concentrated almost entirely on
representing `s` and essentially discarded `t`. A plausible reason, worth
stating as a hypothesis rather than a proven cause: RBF similarity is driven
by which points are closest in the ORIGINAL 7-D space, and if variation
along `s` happens to dominate the overall spread of the data more than
variation along `t` does, an RBF kernel - which has no explicit mechanism
forcing it to spread variance evenly across true underlying factors the way
Isomap's geodesic distances structurally tend to - may simply end up
representing the dominant direction repeatedly across multiple output
dimensions rather than finding a second, genuinely independent one.

## Stage 5: evaluation, compared to Technique 1 (same dataset/settings)

| Metric | Isomap | Kernel PCA | Reading |
|---|---|---|---|
| Trustworthiness | 0.9990 | 0.9916 | Both high; Isomap slightly higher |
| Continuity | 0.9993 | 0.9974 | Both high; Isomap slightly higher |
| Neighbourhood preservation (Jaccard) | 0.5917 | 0.4639 | Isomap noticeably higher - a stricter metric showing a real gap the softer trustworthiness/continuity numbers mostly hide |
| Explained variance ratio | not computable (Isomap truncates its spectrum - see Technique 1's correction) | **0.556** - genuinely computable here (verified: `KernelPCA` keeps its FULL eigenvalue spectrum when asked, unlike Isomap's internal step) | A real, interpretable number for Kernel PCA: these 3 components capture about 56% of the kernel-space "variance" by this measure |
| Fit time | 0.427 s | **0.087 s** | Kernel PCA ~5x faster here - no neighbourhood graph or shortest-path computation to pay for |
| Peak memory | 63.0 MB | **21.8 MB** | Kernel PCA uses less memory too |

**Headline finding:** on this dataset, Kernel PCA is substantially cheaper
(time and memory) than Isomap, with only a modest cost in trustworthiness/
continuity - but it has a real, verified blind spot for one of the two true
underlying latents (`t`) that Isomap and SMACOF do not share. This is
exactly the kind of concrete, technique-specific trade-off (not simply
"faster" or "worse") the eventual Section-5 table needs to report.

## Status
Stage 1 through 5 complete on synthetic data, with a direct comparison
against Technique 1 and a real, hypothesis-stated limitation (latent `t`
not recovered) distinct from the rotation-only story found for Isomap/SMACOF.
Next: real Raichur data.
