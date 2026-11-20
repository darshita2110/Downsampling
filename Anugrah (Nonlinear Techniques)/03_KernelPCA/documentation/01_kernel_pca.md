# Technique 3: Kernel PCA

## Original publication
Scholkopf, B., Smola, A., Muller, K.-R. (1998). "Nonlinear Component Analysis
as a Kernel Eigenvalue Problem." *Neural Computation*, 10(5), 1299-1319.
DOI: 10.1162/089976698300017467

## Implementation status
EXACT REPRODUCTION via `sklearn.decomposition.KernelPCA`, per the project
guideline to prefer established implementations. This module adds input
validation, a cube-shaped API, and docstrings tying arguments to the maths
below - it does not reimplement the kernel eigendecomposition.

## Stage 1: the mathematics

### Strategy, contrasted with Isomap
Isomap keeps PCA-style math but fixes the *distances* first (geodesic instead
of Euclidean), then runs classical MDS. Kernel PCA takes the reverse
approach: keep the original distances, but **warp the space** via a feature
map `phi` so that ordinary PCA, run inside that warped space, finds structure
that was curved in the original space. Concretely:

1. Define `phi: R^7 -> F` (F possibly very high- or infinite-dimensional).
2. Do ordinary PCA in F: eigendecompose `phi(x)`'s covariance.
3. **The kernel trick**: PCA only ever needs dot products `phi(x).phi(y)`,
   never `phi(x)` itself. A kernel function `k(x,y)` that computes this dot
   product directly, without ever materialising `phi`, gives all of PCA's
   benefit in F for the cost of working in the original space.

### Objective function
```
K_ij = k(x_i, x_j)                  (n x n kernel/Gram matrix)
K_centered = H K H                  (double-centering, same idea as Isomap Step 3)
eigendecompose K_centered, keep top-d eigenvectors
```
Closed-form and exact, like Isomap's classical-MDS step - no iteration.

### Kernel choice: RBF (Gaussian)
`k(x,y) = exp(-gamma * ||x-y||^2)` - a SOFT, continuous notion of similarity
(falls off smoothly with distance) rather than Isomap's hard k-nearest-
neighbour cutoff. `gamma` plays the role Isomap's `n_neighbors` plays: how
"local" similarity is judged to be.

### Hyperparameters
- `kernel`: the whole notion of similarity (RBF used here; polynomial,
  sigmoid, etc. are alternatives not used in this project).
- `gamma` (RBF): too small -> approaches linear PCA (everything looks
  similar); too large -> only near-duplicates look similar.
- `n_components`: as always.

### Advantages / limitations
**Advantages:** no neighbourhood graph to build or break - no Isomap-style
short-circuit failure mode; smooth similarity instead of a hard cutoff; fast,
closed-form.
**Limitations:** no natural "geodesic distance" to approximate, so there is
less principled guidance for picking `gamma` than Isomap's `n_neighbors` has
for picking k; RBF has no notion of "walking around" a surface the way
geodesic distance does, so on genuinely Swiss-roll-shaped data it may behave
very differently from Isomap - tested directly in Stage 2, not assumed.

See `experiments/01_basic_example.py` (Stage 2) for the comparison against
Isomap on the same Swiss roll.
