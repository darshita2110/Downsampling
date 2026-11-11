"""
Shared evaluation metrics, used identically across every technique (Isomap,
Kernel PCA, LLE, Laplacian Eigenmaps, FHNPW) so the Section-5 comparison table
is apples-to-apples. Metrics that are not mathematically meaningful for a given
technique are deliberately left out when that technique is evaluated (see each
technique's own evaluation script) rather than computed anyway.
"""
import numpy as np
from sklearn.manifold import trustworthiness as _sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors


def trustworthiness(X_high, X_low, n_neighbors=10):
    """How much the embedding can be trusted: penalises neighbours that appear
    close in the low-dimensional embedding but were NOT close in the original
    high-dimensional space (i.e. false/invented neighbour relationships).
    Range [0, 1], 1 = perfect. Thin wrapper around scikit-learn's
    implementation (Venna & Kaski's definition) for a single, shared
    implementation across all techniques.
    """
    return _sklearn_trustworthiness(X_high, X_low, n_neighbors=n_neighbors)


def continuity(X_high, X_low, n_neighbors=10):
    """The complementary metric to trustworthiness: penalises neighbours that
    were close in the original high-dimensional space but got separated in the
    low-dimensional embedding (i.e. true neighbour relationships that were
    lost). Range [0, 1], 1 = perfect.

    Implementation note: continuity(X_high, X_low) is mathematically
    trustworthiness with the two spaces' roles exchanged - trustworthiness(A, B)
    already computes "neighbours-in-B not in A, penalised by rank-in-A", and
    continuity wants "neighbours-in-X_high not in X_low, penalised by
    rank-in-X_low", which is exactly trustworthiness(X_low, X_high). There is
    no separate scikit-learn function for this, so we reuse trustworthiness
    with swapped arguments rather than reimplementing the rank-penalty logic.
    """
    return _sklearn_trustworthiness(X_low, X_high, n_neighbors=n_neighbors)


def neighborhood_preservation(X_high, X_low, n_neighbors=10):
    """Simple, symmetric companion metric: average Jaccard overlap between each
    point's k-NN set in the original space and its k-NN set in the embedding.
    1 = identical neighbour sets, 0 = completely different. Easier to interpret
    directly than trustworthiness/continuity's rank-weighted penalties, at the
    cost of not weighting by how badly a neighbour was displaced.
    """
    nn_high = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_low)
    # +1 then drop index 0: a point is always its own nearest neighbour (distance 0)
    neighbors_high = nn_high.kneighbors(return_distance=False)[:, 1:]
    neighbors_low = nn_low.kneighbors(return_distance=False)[:, 1:]

    scores = np.empty(len(X_high))
    for i in range(len(X_high)):
        set_high, set_low = set(neighbors_high[i]), set(neighbors_low[i])
        union = set_high | set_low
        scores[i] = len(set_high & set_low) / len(union) if union else 1.0
    return scores.mean()


def reconstruction_error(model):
    """Reconstruction error, only meaningful for techniques that define one
    directly (Isomap exposes `reconstruction_error()`: the residual MDS stress
    left after keeping n_components dimensions, i.e. how much of the geodesic
    distance structure was NOT captured). Returns None, with an explanation,
    for models that do not define this quantity - do not fabricate a
    reconstruction error for methods where one isn't mathematically defined
    (e.g. plain Laplacian Eigenmaps has no natural reconstruction map).
    """
    if hasattr(model, "reconstruction_error"):
        return model.reconstruction_error()
    return None


def explained_variance_ratio(model):
    """Only meaningful for eigenvalue-based methods that expose eigenvalues
    (classical-MDS-style methods, e.g. Isomap's internal kernel PCA step, or
    Kernel PCA directly). Returns None if not applicable - we do not compute an
    'explained variance' for methods (LLE, Laplacian Eigenmaps) where the
    underlying eigenproblem does not correspond to variance in any meaningful
    sense.
    """
    kernel_pca = getattr(model, "kernel_pca_", None)
    eigenvalues = getattr(kernel_pca, "eigenvalues_", None)
    if eigenvalues is None:
        return None
    eigenvalues = np.asarray(eigenvalues)
    positive = eigenvalues[eigenvalues > 0]
    if positive.sum() == 0:
        return None
    n_kept = getattr(model, "n_components", len(positive))
    return float(positive[:n_kept].sum() / positive.sum())


def summarize(X_high, X_low, model=None, diagnostics=None, n_neighbors=10):
    """Convenience wrapper: run every metric that is well-defined given what
    `model` exposes, and fold in the timing/memory diagnostics already
    collected at fit time (see isomap.py's fit_isomap_pixels). Returns a flat
    dict suitable for one row of the Section-5 comparison table.
    """
    summary = {
        "trustworthiness": trustworthiness(X_high, X_low, n_neighbors),
        "continuity": continuity(X_high, X_low, n_neighbors),
        "neighborhood_preservation": neighborhood_preservation(X_high, X_low, n_neighbors),
        "reconstruction_error": reconstruction_error(model) if model is not None else None,
        "explained_variance_ratio": explained_variance_ratio(model) if model is not None else None,
    }
    if diagnostics:
        summary["fit_time_seconds"] = diagnostics.get("fit_time_seconds")
        summary["peak_memory_mb"] = diagnostics.get("peak_memory_mb")
    return summary
