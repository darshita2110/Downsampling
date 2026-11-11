# 2. Isomap

## Original publication
Tenenbaum, J. B., de Silva, V., Langford, J. C. (2000). "A Global Geometric
Framework for Nonlinear Dimensionality Reduction." *Science*, 290(5500), 2319-2323.
DOI: [10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)

## Stage 1: the mathematics

See the full derivation in the project log (and the chat record of this session),
summarised here for the paper:

1. **Build a k-NN graph.** Connect each point to its `k` nearest neighbours (or
   neighbours within radius `epsilon`), edge weight = Euclidean distance. This is
   the only place where "locally flat" is assumed.
2. **Compute graph shortest-path distances** `D_G(i, j)` between every pair of
   points (Dijkstra / Floyd-Warshall). This approximates the true geodesic
   (along-the-surface) distance by chaining trustworthy short hops.
3. **Classical MDS on `D_G`:** double-centre the squared distance matrix,
   `B = -1/2 * H * D_G^2 * H` with `H = I - (1/n) * 1 * 1^T`, eigendecompose
   `B = V * Lambda * V^T`, and take `Y = V_d * sqrt(Lambda_d)` as the embedding.

**Objective minimised:** `E = || tau(D_G) - tau(D_Y) ||_{L2}`, where `tau` is the
double-centering operator above and `D_Y` is the matrix of Euclidean distances in
the embedding `Y`. Classical MDS solves this exactly via eigendecomposition - no
iterative optimisation, no local optima. (Contrast with Technique 2, "Isomap with
SMACOF", which replaces this closed-form step with an iterative stress
minimisation - to be covered when we reach it.)

**Hyperparameters:** `n_neighbors` (k), `n_components` (d). See trade-off table in
`documentation/01_concepts.md` section 1.3 and the concrete demonstration below.

**Advantages:** globally consistent (not patched from local pieces); closed-form,
no local optima; asymptotic convergence guarantees under reasonable sampling
(Tenenbaum et al. 2000, supplementary proofs).

**Limitations:** needs a connected, well-sampled, roughly convex manifold;
pairwise shortest paths are expensive at scale (impractical on a full satellite
tile without subsampling or approximate/landmark variants); sensitive to
"short-circuit" edges in the neighbourhood graph (demonstrated below); as
established in `01_concepts.md`, is not inherently spatially aware when applied
to flattened pixels (to be examined for our cube in Stage 4).

## Stage 2: basic example (`experiments/02_isomap_basic_example.py`)

Used `sklearn.datasets.make_swiss_roll` - the literature's own canonical toy
example - rather than our satellite-style synthetic cube, so this stage tests our
understanding of Isomap itself with nothing project-specific to second-guess.

### Result (a): Isomap vs PCA — `visualisations/02a_isomap_vs_pca_swissroll.png`
PCA's 2-D projection loops the roll into a spiral (the true starting and ending
colour positions end up adjacent). Isomap's 2-D projection lays the same points
out left-to-right in a clean, monotonic colour gradient - the roll is "unrolled"
correctly. Measured numerically (best |Spearman rank correlation| between an
embedding axis and true position along the roll, n=1500 points,
`n_neighbors=10`, seed=42):

| Method | Spearman correlation with true position |
|---|---|
| PCA | 0.221 |
| Isomap | 1.000 |

A perfect 1.000 is expected here because `make_swiss_roll`'s "noise" in our run
(0.05) is small relative to the roll's scale and `n_neighbors=10` is large enough
to keep the graph connected but small enough to avoid short-circuiting (see next
result) - this is closer to Isomap's ideal regime than our own 7-channel cube
will be, which is exactly why we test on both.

### Result (b): the k-NN graph itself — `visualisations/02b_neighbor_graph.png`
Plotted the actual Step-1 graph in 3-D. Edges hug the roll's surface (they run
along the spiral) rather than cutting straight through the gap between windings -
this is the concrete, visual form of "locally flat, globally curved" and of why
chaining these edges approximates geodesic, not Euclidean, distance.

### Result (c): the short-circuit failure, demonstrated — `visualisations/02c_short_circuit_failure.png`
Ran Isomap at `n_neighbors` = 6, 10, 30, 80 on the same data. At 6-10 the roll
unrolls cleanly (consistent with result (a)). At 30 and especially 80, the
embedding visibly curls back into a loop - edges have started jumping across
windings of the roll, corrupting shortest-path distances between points that are
actually far apart along the surface, and the failure mode looks qualitatively
like PCA's failure in `01_pca_sanity_check.py`. This is not a hypothetical
limitation - it is reproduced here on the standard benchmark object, and we will
check for it again whenever we tune `n_neighbors` on the real satellite cube.

### Reproducibility
`RANDOM_SEED = 42` throughout; `make_swiss_roll(n_samples=1500, noise=0.05,
random_state=42)`; scikit-learn's `Isomap` and `PCA` used as-is (exact
reproduction of the standard algorithms, not a modified version at this stage).

## Status
Stage 1 (maths) and Stage 2 (basic example) complete and documented. Next:
Stage 3 - a modular `src/isomap.py` wrapper (input validation, reproducibility,
docstrings tying code to the maths above), then Stage 4 on our synthetic 7-channel
cube, then eventually the real Raichur data per the README's "Commitment: real
data" section.
