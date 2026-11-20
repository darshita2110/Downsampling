# LLE - Stage 4 (synthetic cube) and Stage 5 (evaluation)

Script: `experiments/02_on_synthetic_cube.py`. `n_neighbors=12` (same as
Technique 1/2's synthetic runs; also > 7 channels, so `reg=1e-3` matters per
Stage 1), `n_components=3`, seed=42, standardised channels.

## A found bug in shared code, fixed before this technique could be evaluated fairly

Before writing this technique's results, `src/evaluation.reconstruction_error()`
was checked against LLE directly (not assumed to already work because it
worked for Isomap): LLE exposes `.reconstruction_error_` as an ATTRIBUTE
(trailing underscore), not a callable METHOD the way Isomap's
`.reconstruction_error()` is. The original shared function only checked for
the method form, so it would have silently returned `None` for LLE - hiding
a real, meaningful number rather than reporting it. Fixed `src/evaluation.py`
to check for both forms explicitly, verified against both Isomap (still
returns its original value, 3.328 on a test array) and LLE (now returns a
real value, 8.76e-05 on that same test array) without changing Isomap's
established results.

**Important distinction documented in the code, not just here:** Isomap's
`reconstruction_error()` and LLE's `reconstruction_error_` are NOT the same
quantity, despite the shared name - Isomap's measures leftover geodesic-
distance stress, LLE's measures leftover LOCAL reconstruction cost. Do not
compare their raw values across techniques in the Section-5 table; only
trends within one technique (e.g. across settings) are meaningful for this
particular field.

## Visual check - noisier than every prior technique

`visualisations/02a_lle_embedding_composite.png` is visibly grainier/noisier
than Isomap's, SMACOF's, or Kernel PCA's composites on the same cube -
consistent with (not just coincidental alongside) the evaluation numbers below.

## Latent recovery - a genuine, different kind of partial result

`visualisations/02b_latent_recovery.png`: dimension 0 recovers `s` well
(0.88). Dimensions 1 and 2 show the same RING pattern associated with the
cyclic latent `t` that Isomap and SMACOF also showed - visually, the pattern
looks similar. But checking joint recovery (linear regression on all 3
dimensions together, the same check that rescued `t` to ~0.94 for Isomap and
SMACOF): **LLE's joint recovery of `t` is only 0.292** - not rescued the way
the other two techniques' results were. `s`'s joint recovery (0.947) is
comparable to the others. This is a genuinely different, specific result,
not a repeat of either prior pattern (Isomap/SMACOF's "it's just rotated,
actually fine" story, or Kernel PCA's "doesn't recover it at all" story):
the ring SHAPE associated with `t` is visible, suggesting LLE is representing
*something* about the cyclic structure, but not cleanly enough for a simple
joint linear combination to recover it well - plausibly connected to the
extra noise visible in the composite image and the lower trustworthiness/
continuity numbers below, though this is a hypothesis stated as such, not
proven here.

## Stage 5: evaluation, compared to Technique 1 (same dataset/settings)

| Metric | Isomap | LLE | Reading |
|---|---|---|---|
| Trustworthiness | 0.9990 | 0.9757 | A real, noticeably bigger gap than Kernel PCA's (0.9916) showed against Isomap |
| Continuity | 0.9993 | 0.9840 | Same pattern - bigger gap than any prior technique |
| Neighbourhood preservation (Jaccard) | 0.5917 | **0.2592** | Worse than even Kernel PCA's 0.4639 on this strict metric - the weakest result of any technique so far on this check |
| LLE's own reconstruction error | not applicable | 1.14e-07 | Extremely small - but see note below: this does NOT imply good global neighbourhood preservation, because it measures something purely local |
| Fit time | 0.588 s | 0.382 s | LLE slightly faster |
| Peak memory | 63.0 MB | **2.47 MB** | LLE uses dramatically less memory - no n x n distance or kernel matrix ever built, unlike Isomap/SMACOF/Kernel PCA |

**A genuinely important disconnect worth stating plainly:** LLE's own
reconstruction error (1.14e-07) is tiny - by its own local objective, LLE
did an excellent job. But the SAME embedding scores worst-of-all-techniques
on trustworthiness, continuity, and especially neighbourhood preservation -
metrics computed independently of LLE's own objective. This is not a
contradiction; it is exactly what Stage 1's maths predicts: LLE only ever
optimises a LOCAL reconstruction criterion, with no mechanism (unlike
Isomap's global geodesic distances or even Kernel PCA's global kernel
matrix) enforcing that the GLOBAL layout stays sensible. A method can
satisfy its own objective nearly perfectly while still producing a globally
noisier embedding than other techniques - a concrete illustration of why
Stage 5 uses several different metrics rather than trusting any one
technique's own native cost function as the full picture.

## Status
Stage 1 through 5 complete on synthetic data. A real shared-code bug
(`reconstruction_error` attribute-vs-method mismatch) was found and fixed
here, benefiting any future technique with the same naming pattern. Next:
real Raichur data.
