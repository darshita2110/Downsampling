# Kernel PCA - Stage 2: basic example, and a real negative result

Script: `experiments/01_basic_example.py`. Same Swiss roll as Technique 1/2
(`sklearn.datasets.make_swiss_roll(n_samples=1500, noise=0.05, random_state=42)`).

## Result (a): RBF Kernel PCA vs. Isomap - `visualisations/01a_kpca_vs_isomap_swissroll.png`

Stage 1's maths flagged a real risk: RBF's similarity has no mechanism for
"walking around" a surface the way Isomap's geodesic distance does. We tested
this rather than assuming it mattered in practice. Result: it does matter, a
lot. Isomap scores 1.000 Spearman correlation with true position (clean,
monotonic unroll). RBF Kernel PCA at `gamma=0.01` scores **0.156** - its
output is still a closed loop, the same qualitative failure shape as plain
PCA's (`01_pca_vs_truth.png`), just reached by a different, nonlinear route.

## Result (b): gamma sweep - `visualisations/01b_gamma_sweep.png`

Swept `gamma` across 0.0001, 0.001, 0.01, 0.1, 1.0 to check whether some
other setting rescues the unroll:

| gamma | Spearman correlation | What it looks like |
|---|---|---|
| 0.0001 | 0.221 | A spiral - visually matches plain PCA's failure almost exactly, consistent with Stage 1's prediction that very small gamma behaves like linear PCA |
| 0.001 | 0.218 | Same spiral shape, slightly tighter |
| 0.01 | 0.156 | Loop, less structured |
| 0.1 | 0.141 | Starting to collapse - most points crowd toward one region |
| 1.0 | 0.415 | Collapsed - nearly all points pile into a small cluster near the origin, with a handful of outliers; "best" correlation here is driven by very few points, not a real unroll |

**No setting of gamma unrolls the Swiss roll.** This is a genuine negative
result for this method on this specific kind of data, found by testing
across the hyperparameter's whole reasonable range rather than reporting one
value and moving on. It's not a mistake in how we ran it - it's exactly what
Stage 1's maths predicted: RBF similarity has no concept of distance "along"
a curved surface, so on data that specifically requires that concept (a
Swiss roll, by construction), no amount of gamma-tuning substitutes for it.

## What this means for our actual project, and why we're not dropping Kernel PCA

A Swiss roll is an adversarial example *for this specific limitation* - it
was designed, in the original manifold-learning literature, to need
geodesic-style distance. Our real satellite-pixel data may or may not have
that same "must walk around, not through" structure; that's an empirical
question we answer in Stage 4 (the synthetic cube) and the real-data run,
not something to assume from this one toy result. We are documenting this
negative result precisely so the eventual Section-5 comparison table can say
something specific and honest: Kernel PCA's weakness is sharpest exactly
when data has Isomap-style curved structure, and we have direct evidence of
that from this stage, to be weighed against whatever Stage 4/5 finds on data
that actually resembles our real channels.

## Reproducibility
`RANDOM_SEED = 42`; `make_swiss_roll(n_samples=1500, noise=0.05, random_state=42)`;
`sklearn.decomposition.KernelPCA(kernel="rbf", ...)` used as-is (exact
reproduction, per Stage 1).
