# Technique 2: Isomap with SMACOF

## Original publications
- Isomap's Steps 1-2 (unchanged here): Tenenbaum, de Silva, Langford (2000),
  *Science* 290(5500), 2319-2323. DOI: 10.1126/science.290.5500.2319
- SMACOF: de Leeuw, J. (1977). "Applications of Convex Analysis to
  Multidimensional Scaling." *Recent Developments in Statistics*, North-Holland,
  pp. 133-145.
- de Leeuw, J., Mair, P. (2009). "Multidimensional Scaling Using Majorization:
  SMACOF in R." *Journal of Statistical Software*, 31(3), 1-30.
  DOI: 10.18637/jss.v031.i03

## Implementation status
**Combination of two exact, established implementations**, not a reimplementation
of either: `sklearn.manifold.Isomap` for Steps 1-2 (k-NN graph, geodesic
distances via shortest paths - we fit it but discard its own classical-MDS
embedding, keeping only `dist_matrix_`), and `sklearn.manifold.MDS` with
`dissimilarity="precomputed"` (which runs SMACOF) for Step 3. This combination
itself - "take Isomap's geodesic distances, embed them with SMACOF instead of
classical MDS" - is a recognised variant in the MDS/manifold-learning
literature, not invented for this project.

## Stage 1: the mathematics

### What changes relative to plain Isomap (`01_Isomap/documentation/02_isomap.md`)
Steps 1-2 are identical: build a k-NN graph, compute geodesic distances `D_G`
via shortest paths. Step 3 changes:

- **Classical-MDS Isomap** (Technique 1): double-centre `D_G²`, eigendecompose,
  closed-form embedding. Minimises `E = ||tau(D_G) - tau(D_Y)||` - a criterion
  on an algebraic transform of the distances.
- **SMACOF Isomap** (this technique): minimise **raw Stress** directly:

```
sigma(Y) = sum_i sum_j w_ij * (D_G(i,j) - ||y_i - y_j||)^2
```

  via iterative **majorization**: at each step, build a simple quadratic upper
  bound that touches the true Stress at the current `Y`, minimise that bound
  exactly (closed form - the "Guttman transform", `Y_new = (1/n) B(Y_old) Y_old`),
  move there, repeat. Guaranteed non-increasing Stress at every step (a
  majorizer sitting above the true function, touching it at the current point,
  cannot have a lower minimum elsewhere without the true function being lower
  too) - but not guaranteed to reach the *global* minimum, since Stress is not
  convex in `Y`.

### Why this matters, not just "it's a different algorithm"
Classical MDS's criterion is an algebraic proxy (double-centered squared
distances behave like inner products, which is what makes the closed-form
eigendecomposition possible). Stress is the thing we actually want to be small
- the literal gap between target and achieved distances. SMACOF buys direct
optimisation of the criterion we care about, at the cost of losing the
closed-form guarantee classical MDS has.

### Hyperparameters specific to this technique
- `max_iter`, `eps`: iteration cap / convergence tolerance for majorization.
- `n_init`: number of random restarts (SMACOF can land in different local
  minima; classical MDS has no such issue, being exact).
- `random_state`: here it genuinely changes the initial configuration `Y`,
  unlike plain Isomap where it only nudges an eigen-solver's start vector.

### Advantages / limitations versus classical-MDS Isomap
**Advantages:** optimises actual distance preservation (not an algebraic
proxy); the majorization framework generalises to weighted/non-metric variants
classical MDS's closed form cannot express.
**Limitations:** no global-optimum guarantee (needs `n_init` restarts);
strictly more expensive per run for, on well-behaved data, a similar answer;
introduces genuine random-seed sensitivity that classical-MDS Isomap does not
have.

## Stage 2: basic example (`experiments/01_basic_example.py`)

A version-compatibility issue found and fixed first: the installed
scikit-learn (1.9.1) renamed `MDS`'s constructor arguments (`dissimilarity`,
boolean `metric` -> `metric="precomputed"`, `metric_mds=`), while this
project's pinned `requirements.txt` (scikit-learn==1.5.2) expects the old
names. `isomap_smacof.py`'s `_build_mds()` detects which API the installed
version exposes via `inspect.signature` rather than assuming one - this
matters because this project may run on different machines/versions (as
already happened once, see `01_Isomap/documentation/03_real_data_loader.md`
for an unrelated but similarly version/environment-driven surprise).

### An unplanned finding before the intended comparison: short-circuiting is density-dependent, not just k-dependent
Technique 1's Stage 2 demonstrated short-circuiting at `n=1500` by increasing
`n_neighbors`. Running the same Swiss roll at `n=500` (chosen here only to
keep SMACOF's heavier per-iteration cost fast for a basic example) at
Technique 1's `n_neighbors=10` **already short-circuits** (Spearman 0.828
instead of ~1.0), and gets markedly worse by `n_neighbors=12-20` (down to
0.16-0.52) - checked directly, not assumed. `n_neighbors=6` recovers a clean
unroll (0.999) at this sample size. This confirms, from a new angle,
Technique 1's documented limitation: short-circuiting depends on **sampling
density relative to k**, not k alone - the same k that is safe at one sample
size can already be too large at a sparser one. Used `n_neighbors=6` for the
rest of this script so a broken shared k-NN graph does not confound the
actual question being asked here (classical MDS vs SMACOF), since both
embedding methods are fed the identical (and here, deliberately clean)
geodesic distances.

### Result (a): classical-MDS Isomap vs. Isomap+SMACOF — `visualisations/01a_isomap_vs_smacof_swissroll.png`
With a clean k-NN graph (`n_neighbors=6`), both unroll the unmodified roll
successfully: Spearman correlation with true position **0.999 (classical) vs.
0.982 (SMACOF)**. The two embeddings are rotated/reflected relative to each
other in the figure - expected and not meaningful on its own, since any MDS
solution (classical or SMACOF) is only unique up to rotation, reflection and
translation; only *relative* distances are optimised. SMACOF's small gap
below classical MDS's near-perfect score here is consistent with the maths
(Stage 1): it is optimising a harder, non-convex criterion, directly, via a
locally-convergent algorithm, versus classical MDS's one-shot exact solve.

### Result (b): SMACOF's local minima, demonstrated — `visualisations/01b_smacof_restarts_stress.png`
Ran SMACOF from 8 different random initial configurations (`n_init=1` each,
rather than sklearn's own best-of-n_init, so every individual restart is
visible). 6 of 8 restarts converge to essentially the same, best stress;
**2 of 8 (seeds 1 and 6) get stuck at 4-5x worse stress** - a concrete,
reproducible instance of the local-minima behaviour Stage 1's maths predicts
and that classical-MDS Isomap structurally cannot exhibit (it has exactly one
answer, always, by construction). This is the direct justification for
`n_init` as a hyperparameter worth tuning, not just a default to leave alone.

### Reproducibility
`RANDOM_SEED = 42` for data generation and the main comparison;
`make_swiss_roll(n_samples=500, noise=0.05, random_state=42)`;
`n_neighbors=6` (see short-circuit finding above), `n_components=2`,
SMACOF `n_init=4, max_iter=300, eps=1e-6` for part (a); restart sweep in part
(b) explicitly uses `n_init=1` with `random_state=0..7` to expose individual
restarts rather than sklearn's own internal best-of-n_init selection.

## Status
Stage 1 (maths) and Stage 2 (basic example, including the version-compatibility
fix and the density/short-circuit finding) complete. Next: Stage 4 - apply to
our synthetic 7-channel cube and the real Raichur data, Stage 5 - evaluation,
directly comparable against Technique 1's existing results
(`01_Isomap/results/*.json`) since both use the identical `src/evaluation.py`.
