# LLE - Stage 2: basic example, and a genuinely different failure shape

Script: `experiments/01_basic_example.py`. Same Swiss roll as Technique 1-3
(`make_swiss_roll(n_samples=1500, noise=0.05, random_state=42)`).

## Result (a): LLE vs. Isomap at n_neighbors=10 - `visualisations/01a_lle_vs_isomap_swissroll.png`

Nearly identical unrolled shape to Isomap (Spearman 0.999 vs 1.000), and
LLE's own `reconstruction_error_` is tiny (1.03e-07) - the local linear
recipes found in 3-D transferred to the 2-D embedding almost perfectly, as
Stage 1's maths predicts for well-sampled, locally-flat data.

## Result (b): does LLE's lack of a distance graph change its FAILURE shape? - `visualisations/01b_lle_k_sweep.png`

This is the real test of Stage 1's specific claim, not just a repeat of
Technique 1's sweep. Isomap's failure at large k is distinctive: the
embedding curls back into a closed LOOP, because bad edges jump across
windings and corrupt chained shortest-path distances (see
`01_Isomap/visualisations/02c_short_circuit_failure.png`). LLE has no
chained distances to corrupt this way - so does it fail the same way, a
different way, or not at all?

**Checked directly:** at k=30, correlation drops to 0.93 and the embedding
visibly PINCHES toward one corner - points converging into a narrow wedge,
not folding into a loop. At k=80, correlation collapses to 0.15, and the
shape becomes flat, interleaved horizontal colour stripes - very different
colours sitting at similar heights, but critically, NOT a closed loop the
way Isomap's failure is. This confirms Stage 1's prediction: LLE still
breaks down at large k (so "no distance graph" does not mean "immune to bad
k"), but through a qualitatively different mechanism - the local
reconstruction step becoming under-constrained/unstable as neighbourhoods
stop being locally flat, not geodesic distances short-circuiting across
windings. Interesting secondary observation: at k=6, correlation is already
slightly off-perfect (0.94, not ~1.0) with a visible pinch at one end too -
LLE's "good" k range for this data may be narrower than Isomap's, worth
keeping in mind when choosing k on our actual synthetic/real cubes in Stage 4.

## Reproducibility
`RANDOM_SEED=42`; `make_swiss_roll(n_samples=1500, noise=0.05, random_state=42)`;
`sklearn.manifold.LocallyLinearEmbedding(method="standard")`, `reg=1e-3` for
the k-sweep (needed since k up to 80 exceeds the roll's 3 ambient dimensions,
per Stage 1's note on regularisation being required whenever k > n_features).
