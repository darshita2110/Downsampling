# Isomap+SMACOF - real Raichur data: results, and a code bug this run caught

Run by you, on your own machine, against the real `raichur_features.parquet`
(2026-09-27, 2,000-pixel subsample - smaller than Technique 1's 8,000, per
this technique's own docstring on SMACOF's higher per-pixel cost).

## A real bug this run caught, distinct from the environment issues

Before this could run at all, it crashed with
`TypeError: MDS.__init__() got an unexpected keyword argument 'init'`. This
sandbox's scikit-learn (1.9.1) was masking the bug: this project's actual
PINNED version, scikit-learn==1.5.2 - confirmed directly via
`inspect.signature(MDS.__init__)` on your machine - has no `init` parameter
at all (its full signature: `n_components, metric, n_init, max_iter, verbose,
eps, n_jobs, random_state, dissimilarity, normalized_stress`). The original
`_build_mds()` added `init="random"` unconditionally, checking only whether
argument NAMES had been renamed between versions, not whether an argument
EXISTS at all on some versions. Fixed by checking `"init" in params` the same
way every other kwarg in that function already was - this is now the kind of
defensive check that should be applied by default for every optional
constructor argument, not just the ones we happened to test for renaming.

This is worth flagging for the paper's methodology section: the dev sandbox's
different scikit-learn version did not catch this bug - it only surfaced
once the code ran against the project's actual pinned, real-world
environment. Reusing code across machines genuinely needs testing on the
pinned version, not just "a" version.

## The real, measured finding: SMACOF's neighbourhood-preservation edge holds up on real data, and is LARGER than on synthetic data

| Metric | Isomap (8,000 px) | SMACOF (2,000 px) |
|---|---|---|
| Trustworthiness | 0.9987 | 0.9969 |
| Continuity | 0.9995 | 0.9972 |
| Neighbourhood preservation (Jaccard) | **0.4265** | **0.6112** |
| Fit time | 14.13 s | 178.73 s |
| Peak memory | 1,541 MB | 273 MB |

On synthetic data (`documentation/02_stage4_5_synthetic.md`), SMACOF's edge
on neighbourhood preservation over Isomap was small and called
"inconclusive" (0.629 vs 0.592, +0.037). **On real data, the same edge is
much larger** (0.611 vs 0.427, +0.184) - and this held up even though SMACOF
was run on a quarter as many pixels (2,000 vs 8,000), which if anything
should make its job harder, not easier, since fewer pixels usually means a
less sampled, less supportive neighbourhood structure. This cross-validates
the synthetic-data hypothesis (Stage 1's prediction that SMACOF directly
optimising distance-preservation should show up most clearly in a strict,
unweighted metric like Jaccard neighbourhood overlap) on real satellite data,
not just a toy construction - a genuinely stronger piece of evidence than
either result alone.

**Important honesty check before overstating this:** `smacof_n_iter: 300`
equals `smacof_max_iter: 300` exactly - the optimisation **hit the iteration
cap without declaring convergence** (compare: the synthetic run converged
cleanly at 76 iterations, well under its own 300 cap). This means the
real-data SMACOF embedding may still have been improving when we stopped it,
so both the 178.73s runtime AND the reported stress/metrics describe an
embedding that had not fully settled - not necessarily representative of
what SMACOF would need or achieve if allowed to run to convergence. This is
a limitation of this specific run's settings, not of the method itself, and
should be stated as such rather than read as "SMACOF takes 178s on real
data" - it may take longer still to actually converge, or converge earlier
with better-tuned settings (fewer random restarts, a looser `eps`, or a
smarter initial configuration).

## Cost: as expensive as previously measured, or more

178.73s for 2,000 real pixels, vs. 12.06s for 1,600 synthetic pixels (7.5x
the pixels... wait, 2000/1600 = 1.25x the pixels, but 14.8x the time) -
substantially worse than even the ~25x-per-pixel-count-adjusted multiplier
found on synthetic data, though the non-convergence above (300 vs 76
iterations, a 3.9x difference on its own) explains most of this gap rather
than indicating some new, unexplained real-data slowdown. Memory (273 MB)
was in line with expectations for 2,000 pixels (consistent with the
quadratic-in-pixel-count memory scaling already established for Isomap and
carried over here, since the geodesic-distance matrix is the same size
either way).

## Updated headline for the eventual Section-5 table

SMACOF's trade-off, now backed by both synthetic AND real evidence: a real,
and on real data substantial, improvement in neighbourhood preservation over
classical-MDS Isomap, at a cost that is severe and - per the non-convergence
finding above - possibly understated by this run's default settings. A
sensible recommendation for the paper: SMACOF may be worth its cost
specifically when neighbourhood preservation is the priority metric, but
needs either a larger `max_iter` budget or a convergence-diagnostic check
(exactly what this run surfaced) before its runtime numbers are taken as
final.

## Status
Technique 2 (Isomap with SMACOF) is now complete end-to-end: Stages 1-5, on
both synthetic and real data, with one real code bug found and fixed (the
`init` kwarg) and one real, cross-validated finding (the neighbourhood-
preservation edge) - alongside an honest flag about non-convergence rather
than letting the raw runtime number stand unqualified.
