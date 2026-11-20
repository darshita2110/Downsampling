# Isomap+SMACOF — Stage 4 (synthetic cube) and Stage 5 (evaluation)

Script: `experiments/02_on_synthetic_cube.py`. Settings identical to Technique
1's synthetic-cube run (`n_neighbors=12`, `n_components=3`, seed=42,
standardised channels) specifically so the two are comparable like-for-like.

## What's NOT repeated here, and why
The spectral-vs-spatial-neighbour demonstration
(`01_Isomap/visualisations/03c_spectral_vs_spatial_neighbors.png`) tests
Steps 1-2 (the k-NN graph), which SMACOF does not change - re-running it here
would reproduce an identical result, not test this technique. Skipped
deliberately; see that figure for the (still valid, unchanged) evidence.

## Visual check
`visualisations/02a_smacof_embedding_composite.png` is spatially coherent and
structurally similar to Technique 1's composite - expected, since both start
from the same geodesic distances.

## Latent recovery, including a rotation subtlety worth being honest about
`visualisations/02b_latent_recovery.png`: dimension 0 shows the same
ring/annulus pattern as Technique 1's result for the cyclic latent `t`
(expected - same cause, see `01_Isomap/documentation/02_isomap_stage4_5.md`).
Latent `s`, however, looked WORSE here at first glance than in Technique 1:
split weakly across dimensions 1 and 2 (Spearman 0.45 and 0.52) rather than
concentrated strongly in one (Technique 1's dim 2 scored 0.94 alone). Checked
directly rather than reported as a weakness: a linear regression predicting
`s` from dimensions 1 and 2 jointly achieves Spearman correlation **0.95**
(R²=0.89) - essentially identical recovery to Technique 1, just **rotated**
across two axes instead of aligned with one. This is expected and not a
shortcoming of SMACOF: as noted in Stage 2, any MDS solution (classical or
SMACOF) is unique only up to rotation/reflection, since only relative
distances are optimised - nothing in either method's objective privileges any
particular axis orientation. A real difference between the two techniques
would need to show up in something rotation-INVARIANT (distances, neighbour
sets, stress) - which is exactly what the metrics below check.

## Stage 5: evaluation, directly compared to Technique 1 (same dataset/settings)

| Metric | Isomap (Technique 1) | Isomap+SMACOF (Technique 2) | Reading |
|---|---|---|---|
| Trustworthiness | 0.9990 | 0.9992 | Essentially identical |
| Continuity | 0.9993 | 0.9995 | Essentially identical |
| Neighbourhood preservation (Jaccard) | 0.5917 | 0.6288 | SMACOF slightly higher here - plausibly because it optimises raw distance preservation directly (Stage 1), which is closer to what this metric measures, rather than classical MDS's algebraic proxy. One dataset is not enough to call this a general result - revisit once real-data numbers are in. |
| Fit time | 0.618 s | 15.600 s | **~25x slower** - geodesic-distance computation (0.613s, identical cost to Technique 1) is a small fraction of the total; SMACOF's 76 iterations x `n_init=4` restarts dominates (14.988s) |
| Peak memory | 63.0 MB | 154.5 MB | ~2.5x more - SMACOF needs working storage for the iterative majorization steps on top of the same geodesic distance matrix |
| SMACOF stress (this technique only) | not applicable | 229,861 | Isomap has no equivalent number to compare against directly - see `reconstruction_error`/`explained_variance_ratio` below |
| Reconstruction error | 2.345 (Isomap's own definition) | **None - not computed** | `src/evaluation.reconstruction_error` only recognises `model.reconstruction_error()`, defined for `sklearn.manifold.Isomap`, not for an `MDS`/SMACOF model. We do not report Isomap's reconstruction-error number for SMACOF's embedding (they would not be measuring the same thing) - SMACOF's own stress is reported separately in the results JSON (`smacof_stress`) as the appropriate analogue, not folded into this shared field to avoid implying false comparability. |
| Explained variance ratio | None (not computable, see Technique 1's correction) | None (not applicable - MDS/SMACOF has no eigenvalue spectrum at all) | Consistent: neither technique reports a number here, for related but distinct reasons |

**Headline finding: on this dataset, SMACOF buys a small, inconclusive
trustworthiness/neighbourhood-preservation edge at a large (~25x) runtime
cost and ~2.5x memory cost**, for an embedding that is otherwise structurally
the same information, just differently rotated. This is consistent with
Stage 1's prediction (Stage 1: SMACOF optimises a "better" criterion directly
but pays for it with iteration) and is exactly the kind of quantified
trade-off the eventual Section-5 comparison table needs - not "SMACOF is
better" or "SMACOF is worse", but a specific, measured cost for a specific,
small, inconclusive benefit on this data.

## Computational complexity note (extends Technique 1's)
Steps 1-2 cost is identical to Technique 1 (confirmed: 0.613s here vs. 0.618s
total there, consistent). SMACOF adds its own per-iteration cost - each
majorization step is a dense `n x n` matrix operation - run `max_iter` times
per restart, `n_init` times. At the pixel counts Technique 1 already flagged
as requiring subsampling on real data (Stage 4/5 scalability note), SMACOF
will be substantially more expensive still, not less - this is worth stating
plainly for the real-data run rather than discovering it empirically with a
very long wait.

## Status
Stage 1 (maths) through Stage 5 (evaluation) complete on synthetic data, with
a direct, like-for-like comparison against Technique 1. Next: real Raichur
data, per the project's "Commitment: real data" section - expect a
substantially longer runtime than Technique 1's real-data run given the ~25x
synthetic-data slowdown found here, so the real-data script uses a smaller
subsample than Technique 1's to keep the run practical, stated explicitly in
that script rather than left as a silent difference.
