# 2 (continued). Isomap — Stage 4 (applied to our synthetic cube) and Stage 5 (evaluation)

Script: `experiments/03_isomap_on_synthetic_cube.py`. Settings: `n_neighbors=12`,
`n_components=3`, `random_state=42`, channels standardised first
(`src/preprocessing.standardize`). Code: `src/isomap.py` (`fit_isomap_cube`),
`src/evaluation.py` (`summarize`).

## Pixel-wise vs. spatial: what Isomap actually does here

Isomap is applied **pixel-wise**: the cube is flattened to `(1600, 7)`, Isomap's
k-NN graph is built purely on spectral (7-D) distance between pixels, and the
`(1600, 3)` output is reshaped back to `(40, 40, 3)`. This is stated plainly
because it is easy to assume otherwise from the output image alone.

We checked this is really true, concretely, rather than just restating the claim
from `documentation/01_concepts.md`: `03c_spectral_vs_spatial_neighbors.png` picks
one pixel near the centre of the image and plots its 10 actual spectral
nearest-neighbours (the ones Isomap's graph connects it to) against its immediate
3x3 spatial neighbourhood. **Result: 0 of the 10 spectral neighbours are also
immediate spatial neighbours** - they are scattered up to ~20 pixels away,
clustered instead by spectral similarity (several visible as tight little
sub-clusters elsewhere in the image, i.e. other locations with a similar
land-cover-like profile). This is direct evidence, not an assumption: spatial
coherence in the output composite (`03a`) comes entirely from the fact that our
synthetic latent fields are themselves spatially smooth, not from anything in
Isomap's algorithm that uses pixel position.

## Visual check: does the embedding look sensible?

`03a_isomap_embedding_composite.png` shows the original cube (B04,B03,B02 as RGB)
beside the 3-component Isomap embedding (false colour). The four "bump" regions
visible in the original data remain visually distinct and spatially coherent
in the embedding - again, because of the point above, not despite it.

## Does Isomap recover the TRUE latent structure? An honest, specific finding

`03b_latent_recovery.png` compares each of the 3 embedding dimensions against the
2 ground-truth latents (`t`, `s`) that generated the cube (see
`experiments/00_make_synthetic_dataset.py`). Results (Spearman correlation,
best-matching latent per dimension):

| Embedding dim | Best-matched latent | Spearman |
|---|---|---|
| 0 | s | 0.25 |
| 1 | t | -0.26 |
| 2 | s | -0.94 |

Dimension 2 recovers latent `s` very well. Dimensions 0 and 1, however, show a
**ring/annulus pattern**, not a smooth gradient matching `t` directly - and their
correlation with `t` is weak. This is not a failure of Isomap to report without
explanation; it is an expected consequence of how we built the dataset: our
synthetic generator (`swiss_roll()` in `00_make_synthetic_dataset.py`) turns `t`
into an **angle** `phi` (`phi = 1.5*pi*(1 + 2*t)`) before using `cos(phi)` and
`sin(phi)` to place points on the roll. An angle is a *cyclic* quantity - it
cannot be represented faithfully by a single linear axis (0 and 2*pi are the same
angle, but far apart on a number line), so Isomap, correctly, represents it using
*two* embedding dimensions jointly (hence the ring pattern in each one
individually), rather than "failing" to find `t`. We checked this directly rather
than leaving it as a plausible-sounding claim: taking `angle = atan2(dim1, dim0)`
and comparing `cos(angle)`/`sin(angle)` against `cos(phi)`/`sin(phi)` (the true
roll angle) gives Spearman correlations of **0.939** and **0.934** respectively -
versus 0.25 and -0.26 for dimensions 0 and 1 taken alone. Isomap recovered the
cyclic latent correctly; our single-dimension correlation check was simply the
wrong tool to see it. We note this as a concrete limitation of
per-dimension Spearman correlation as an evaluation check when any true latent is
cyclic, not of Isomap, and will account for cyclic latents more carefully if we
extend the synthetic generator later.

**Takeaway for the paper:** a single-dimension correlation check is not sufficient
evidence of latent recovery when any true latent is cyclic; multi-dimension joint
checks (or avoiding cyclic constructions in future synthetic datasets) are
needed. This is exactly the kind of thing a synthetic dataset with known ground
truth is supposed to surface before we ever trust an embedding on real data where
no ground truth exists.

## Stage 5: evaluation metrics

Computed via `src/evaluation.summarize` (k=12 for trustworthiness/continuity/
neighbourhood preservation, matching the Isomap graph's own `n_neighbors`):

| Metric | Value | Interpretation |
|---|---|---|
| Trustworthiness | 0.999 | Almost no "false" neighbours introduced by the embedding |
| Continuity | 0.999 | Almost no true neighbours lost |
| Neighbourhood preservation (Jaccard) | 0.592 | A stricter, unweighted check: on average ~59% of a pixel's spectral k-NN set survives unchanged in the embedding - lower than the other two because Jaccard does not give partial credit for a neighbour that moved to rank 13 instead of rank 12, while trustworthiness/continuity do |
| Reconstruction error | 2.345 | Isomap's own residual MDS stress (see `documentation/02_isomap.md`); only meaningful in comparison to other `n_neighbors`/`n_components` settings on the *same* dataset, not as an absolute number |
| "Explained variance ratio" | 1.000 | Reported via Isomap's internal kernel-PCA eigenvalues (see `src/evaluation.explained_variance_ratio`); a value of 1.000 here mainly reflects that very few of the geodesic-distance-derived eigenvalues are positive/significant at this scale - **flagged as needing closer inspection before being quoted in the paper as "100% variance explained"**, since that phrasing implies something stronger than what this quantity actually measures for Isomap. We will revisit this once we have Kernel PCA's more standard explained-variance numbers to compare against. |
| Fit time | 0.44 s | 1,600 pixels, laptop/sandbox CPU - see complexity note below |
| Peak memory | 63.0 MB | `tracemalloc`-measured Python-level peak during `fit_transform` |

**Metrics deliberately not computed:** downstream classification performance (no
labels in this synthetic dataset - will be revisited if/when we have labelled real
data).

## Computational complexity and scalability note

All-pairs shortest paths (Stage-1, Step 2) cost `O(n^2 log n)` with Dijkstra +
Fibonacci heaps per source, run for all `n` sources, i.e. effectively `O(n^2 log
n)` total for sparse graphs (scikit-learn's default), or worse with denser
graphs/Floyd-Warshall. At `n=1,600` pixels (our 40x40 synthetic tile) this is
trivial (well under a second, as measured above). A real Sentinel-2 tile is
typically far larger - e.g. a 1,000x1,000-pixel tile is `n=1,000,000` pixels,
roughly 625x more pixels than our synthetic test. Since cost scales worse than
linearly, this is not simply "625x slower" - it is likely infeasible to run exact
Isomap on a full tile without first subsampling pixels, tiling the image into
patches, or using an approximate/landmark Isomap variant (a known extension in
the literature, not implemented here). This will be addressed explicitly when we
reach the real-data stage per the project README's "Commitment: real data"
section - we will not silently subsample without reporting it.

## Research gap / possible improvement noted here

Because Section 4 of the project brief specifically asks us to look at
spatially-constrained formulations where appropriate: a natural extension for
Isomap (not implemented yet, flagged as future work for the paper) is to build
the k-NN graph from a **combined spectral + spatially-weighted distance**
(e.g. concatenating normalised row/col coordinates to the 7 spectral channels
before computing neighbours, with a tunable weight controlling how much spatial
proximity influences the graph), rather than spectral distance alone. This would
make the "neighbour" relationship partly reflect real spatial adjacency while
still allowing spectrally similar but spatially distant pixels to connect when
the weight is small. This is a modification to the published algorithm, so if
implemented later it must be clearly labelled "modified Isomap (ours)", distinct
from the exact reproduction used in this stage.
