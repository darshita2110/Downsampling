# Technique 4: Locally Linear Embedding (LLE)

## Original publication
Roweis, S. T., Saul, L. K. (2000). "Nonlinear Dimensionality Reduction by
Locally Linear Embedding." *Science*, 290(5500), 2323-2327.
DOI: 10.1126/science.290.5500.2323

## Implementation status
EXACT REPRODUCTION via `sklearn.manifold.LocallyLinearEmbedding` (`method="standard"`,
the originally published algorithm), per the project guideline to prefer
established implementations.

## Stage 1: the mathematics

### Core idea - a different mechanism from Isomap and Kernel PCA
Isomap fixes distances (geodesic, not Euclidean) then does algebra on them.
Kernel PCA warps the whole space via a kernel, then does PCA there. LLE does
neither: it never measures a distance between two points directly. Instead,
it describes every point by how it is LINEARLY RECONSTRUCTED from its
neighbours, and finds low-dimensional coordinates where that same local
recipe still works. The intuition: a local "recipe" (e.g. "I am 40% of the
way from neighbour A to neighbour B, plus a bit of C") survives flattening a
curved surface, because any curved surface looks locally flat up close - the
same zoom-in intuition Isomap's Stage 1 used, applied differently here.

### Algorithm, step by step
1. **Neighbours**: for every point, find its k nearest neighbours (no graph
   of chained distances built - unlike Isomap, nothing here is a path).
2. **Reconstruction weights**: solve, per point, for weights `W_ij` minimising
   `|| x_i - sum_j W_ij x_j ||^2` subject to `sum_j W_ij = 1`. The sum-to-1
   constraint makes W invariant to rotation/scale/translation of the local
   neighbourhood - it captures pure local geometry. Closed-form per point
   (local least-squares, no iteration).
3. **Low-dimensional coordinates**: hold W fixed, solve for Y minimising
   `sum_i || y_i - sum_j W_ij y_j ||^2`. Reduces to an eigenvalue problem on
   `M = (I-W)^T(I-W)`, taking the smallest NON-ZERO eigenvectors (the very
   smallest eigenvalue is always 0, a trivial degenerate solution, discarded).
   Closed-form, no iteration - like the other techniques' final steps.

### Objective function
`Phi(Y) = sum_i || y_i - sum_j W_ij y_j ||^2`, subject to Y having zero mean
and unit covariance. Contrast with the other two techniques: Isomap
minimises stress on pairwise geodesic DISTANCES; Kernel PCA maximises
variance in a warped feature SPACE; LLE minimises how badly each point's
local linear RECONSTRUCTION breaks when squeezed into fewer dimensions. No
two of the three techniques optimise the same quantity.

### Hyperparameters
- `n_neighbors` (k): size of the assumed-locally-flat patch.
- `reg`: regularisation for Step 2's local least-squares solve, REQUIRED
  whenever k exceeds the input dimensionality (our case: 7 channels, and we
  will likely want k > 7 for a stable fit) - an extra hyperparameter neither
  Isomap nor Kernel PCA has, checked directly in Stage 2/4 rather than left
  at scikit-learn's default without comment.
- `n_components`: as always.
- `method`: `'standard'` (used here, the original algorithm); scikit-learn
  also offers `'modified'` (MLLE, addressing some regularisation
  instabilities), `'hessian'`, `'ltsa'` - not used, noted as alternatives if
  `'standard'` proves unstable on our data.

### Advantages / limitations for satellite imagery
**Advantages:** no global distance graph, so Isomap's specific short-circuit
failure (one bad edge corrupting many pairwise shortest-path distances)
structurally cannot occur in the same form; purely local per-point
computation.
**Limitations:** with only 7 input channels, a stable local fit will very
likely need `k > 7`, which REQUIRES the `reg` parameter - its effect needs
checking, not assuming the default is adequate; LLE can still distort data
where local patches genuinely aren't flat, just via a different mechanism
(ill-conditioning, "tearing") rather than Isomap's short-circuiting - tested
directly in Stage 2, not assumed safer just because one specific failure
mode doesn't apply.

See `experiments/01_basic_example.py` (Stage 2) for the comparison against
Isomap and Kernel PCA on the same Swiss roll, and a direct test of whether
LLE's lack of a distance graph changes its behaviour as `n_neighbors` grows.
