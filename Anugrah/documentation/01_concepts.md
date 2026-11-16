# 1. Concepts: non-linear dimensionality reduction, and why PCA is not enough

Status: conceptual foundation, written before any technique-specific implementation.
Evidence figure referenced below: `visualisations/01_pca_vs_truth.png`, produced by
`experiments/01_pca_sanity_check.py` on `data/synthetic_v1/cube.npy`.

## 1.1 What "non-linear dimensionality reduction" means, intuitively

Take a flat sheet of paper and roll it up into a scroll, like a Swiss roll cake sliced
open. Every point on the paper still only needs 2 numbers to describe where it is
*on the paper* (how far along, how far across). But once rolled up, those same points
sit in 3-D space, and two points that are close in straight-line (3-D, "Euclidean")
distance might actually be on completely different windings of the roll - far apart
if you had to walk along the paper's surface to get from one to the other.

- The **intrinsic dimensionality** of the data is 2 (the paper is 2-D).
- The **ambient dimensionality** is 3 (it is sitting in 3-D space).
- A technique that only looks at straight-line distances in the ambient space
  (like PCA) cannot tell the difference between "close along the paper" and
  "close because two windings happen to be near each other".
- A technique that instead measures distance *along the surface* (geodesic distance),
  or that only trusts distances between very close neighbours (where straight-line
  and along-surface distance roughly agree), can recover the 2 true coordinates.

This toy object is literally called a "Swiss roll" in the manifold-learning
literature, and it is exactly what `experiments/00_make_synthetic_dataset.py`
builds (see that file's docstring) before turning it into 7 pretend satellite bands.

**Satellite-imagery version of the same idea:** a pixel's 7 measured values (4 raw
bands + NDVI + NDWI + LST) are not 7 independent facts about that pixel. They are
7 different, non-linear "views" of a smaller number of real physical quantities -
things like vegetation vigour, soil moisture, surface temperature, canopy structure.
NDVI, for instance, is already a *non-linear* function of B08 and B04 (a ratio of a
difference to a sum), not a linear combination. If the true physical drivers combine
non-linearly to produce the 7 channels, then the 7-D cloud of pixels, viewed as
points in 7-D space, is expected to lie on or near some curved, lower-dimensional
surface - not a flat subspace. Non-linear dimensionality reduction tries to recover
coordinates on that curved surface directly, rather than projecting onto the nearest
flat approximation of it.

## 1.2 Why PCA can fail here (not just "PCA is linear" as a slogan)

PCA finds the flat (linear) subspace that keeps the most variance. Concretely, it:
1. Centres the data.
2. Computes the covariance matrix of the 7 channels.
3. Takes the top-k eigenvectors of that covariance matrix as new axes.
4. Projects every pixel onto those axes.

Every step above only ever adds and scales channels - it can rotate and stretch the
cloud of points, but it can never bend it. If the true structure is curved (like the
rolled-up paper), the best flat subspace through it necessarily cuts across different
windings of the roll, mixing points that are actually far apart along the true
surface.

We checked this concretely rather than asserting it: `experiments/01_pca_sanity_check.py`
runs PCA on our synthetic 7-channel cube and keeps 2 components, which already
capture 85% of the total variance (67% + 18%). That sounds like PCA is doing well.
But look at `visualisations/01_pca_vs_truth.png`: the right panel shows each pixel's
*true* position along the roll (ground truth, known because we built the data); the
left panel is the PCA projection, coloured by that same true position. If PCA had
recovered the manifold, the colours would vary smoothly across the scatter. Instead,
the scatter forms a loop where different true positions (different colours) sit right
next to each other - PCA has folded the roll's windings on top of each other. High
explained variance is not the same as recovering the right structure: variance
measures spread, not whether the spread is organised the way the true data is
organised.

**Why this matters for multispectral/hyperspectral pixels specifically:**
- Several of our channels are deliberately non-linear functions of others (NDVI,
  NDWI are ratios), so some non-linearity is baked into the feature definitions
  themselves, before any physics is considered.
- Reflectance response to physical drivers (chlorophyll content, soil moisture,
  canopy water content) is well documented in remote sensing to be non-linear and
  often saturating (e.g. NDVI saturates at high biomass).
- Mixed pixels, atmospheric effects and sensor saturation add further non-linear
  distortions.
None of this means PCA is useless - it is still a fast, interpretable, well-understood
baseline, and if the true structure happens to be close to flat, PCA will find it
efficiently. It means we cannot assume linearity and should check, which is what
the sanity check above does and what later evaluation (trustworthiness, continuity,
reconstruction error) will keep doing for every technique.

## 1.3 Isomap vs. Kernel PCA vs. LLE vs. Laplacian Eigenmaps

All four are "manifold learning" methods: they assume the high-dimensional data
(here, 7-D pixels) lies near a lower-dimensional curved surface, and they try to find
coordinates on that surface. They differ in *what structure they try to preserve* and
*how they do the math*.

| | Isomap | Kernel PCA | LLE | Laplacian Eigenmaps |
|---|---|---|---|---|
| **Core idea** | Preserve geodesic (along-the-surface) distances between *all* pairs of points | Apply PCA after lifting points into a (possibly infinite-dimensional) feature space via a kernel, so curved structure becomes "flat" there | Preserve, for each point, the *local linear reconstruction weights* from its nearest neighbours | Preserve local neighbourhood proximity (nearby points in input space should stay nearby in the embedding) |
| **What's built first** | A k-NN graph; graph shortest-path distances approximate geodesic distances | A kernel (similarity) matrix between all pairs of points | A k-NN graph; weights so each point is reconstructed as a linear combination of its neighbours | A k-NN (or epsilon-ball) graph; an affinity/weight matrix per edge |
| **Core computation** | Classical MDS (eigendecomposition) on the geodesic distance matrix | Eigendecomposition of the (centred) kernel matrix | Eigendecomposition of a sparse matrix built from reconstruction weights | Eigendecomposition of the graph Laplacian |
| **Preserves globally or locally?** | Global: all pairwise geodesic distances | Depends on kernel; RBF kernels emphasise local similarity, but the method is still a global eigendecomposition | Local: only neighbourhood relationships, global layout emerges indirectly | Local: explicitly only neighbourhood relationships |
| **Is it a "kernel method"?** | Can be written as kernel PCA with a specific geodesic-distance-derived kernel | Yes, by definition | No (no kernel trick; different objective) | Yes, with the graph Laplacian playing the kernel's role |
| **Needs a connected neighbourhood graph?** | Yes - disconnected graphs break shortest-path distances | No (kernel is defined for any pair directly) | Yes | Yes |
| **Sensitive to short-circuit edges (a bad neighbour link that bridges two distant regions)?** | Yes, and strongly - one bad edge can distort many shortest paths | Not applicable in the same way | Less so - only affects that point's local reconstruction | Less so, similarly local |
| **Typical use when the manifold is...** | ...a single, well-sampled, roughly convex curved surface (e.g. Swiss roll) | ...structure best captured by a *known similarity shape* (e.g. RBF kernel matching your prior about what "similar" means) | ...locally linear everywhere, non-convex or holey surfaces | ...locally linear everywhere, when you care about preserving clusters/neighbourhoods more than exact distances |

**One-line mental model for each:**
- **Isomap**: "Measure distances by walking along the surface, not jumping through
  the air; then do ordinary PCA/MDS on those corrected distances."
- **Kernel PCA**: "Warp the data into a new space, chosen by the kernel, in which it
  is (hopefully) flat; then do ordinary PCA there."
- **LLE**: "Describe every point by how it's linearly built from its neighbours; find
  low-dimensional coordinates that keep those same local recipes working."
- **Laplacian Eigenmaps**: "Build a graph of who's near whom; find low-dimensional
  coordinates where connected points stay close, using the graph's spectral
  structure."

### Key hyperparameters shared across all four
- `n_neighbors` (or kernel bandwidth, for Kernel PCA with RBF): controls how "local"
  the notion of structure is. Too small -> disconnected graphs / noisy local
  structure; too large -> starts to behave like PCA (over-smooths the true manifold).
- `n_components`: how many latent dimensions to keep (3, in our case).
These, plus method-specific hyperparameters, are documented per-technique as we
implement each one, with the exact values used in every experiment (required for
reproducibility - see `results/` once experiments exist).

### Important caveat we will keep repeating (Stage 4 of every technique)

All four of the methods above, in their standard scikit-learn form, treat each pixel
as an independent 7-D point - the k-NN graph (or kernel) is built over *spectral*
distance between pixels, not over spatial position in the image. This means: running
plain Isomap/Kernel PCA/LLE/Laplacian Eigenmaps on a flattened `(H*W, 7)` array does
**not**, by itself, preserve or use spatial structure (which pixel is next to which).
Two spectrally similar pixels from opposite corners of the image will be treated as
neighbours just as readily as two adjacent pixels. We will say this explicitly for
each technique rather than assume spatial coherence comes for free, and we will
separately look at spatially constrained variants (e.g. restricting neighbours to a
spatial window, or combining spectral and spatial coordinates before building the
neighbourhood graph) where the assigned technique calls for it (this is explicit for
"Constrained Laplacian Eigenmaps").

## 1.4 Sources for this section

- Tenenbaum, de Silva, Langford (2000), "A Global Geometric Framework for Nonlinear
  Dimensionality Reduction", *Science* 290(5500), pp. 2319-2323. (Isomap)
- Roweis, Saul (2000), "Nonlinear Dimensionality Reduction by Locally Linear
  Embedding", *Science* 290(5500), pp. 2323-2327. (LLE)
- Belkin, Niyogi (2003), "Laplacian Eigenmaps for Dimensionality Reduction and Data
  Representation", *Neural Computation* 15(6), pp. 1373-1396. (Laplacian Eigenmaps)
- Schölkopf, Smola, Müller (1998), "Nonlinear Component Analysis as a Kernel
  Eigenvalue Problem", *Neural Computation* 10(5), pp. 1299-1319. (Kernel PCA)
- Vaddi et al. (2024), *Egyptian Journal of Remote Sensing and Space Sciences*, 27,
  pp. 82-92. (our project's reference paper; exact formulation of FHNPW to be
  verified against its own cited source before implementation - see root README)

Each of these will be cited again, with DOI/links, in the per-technique write-up once
we build that technique (Stage 1 of the workflow in the project's main instructions).
