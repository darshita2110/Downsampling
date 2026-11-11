"""
Isomap for spatial-spectral dimensionality reduction.

Original algorithm: Tenenbaum, de Silva, Langford (2000), "A Global Geometric
Framework for Nonlinear Dimensionality Reduction", Science 290(5500), 2319-2323.
DOI: 10.1126/science.290.5500.2319

Implementation status: EXACT REPRODUCTION. We use scikit-learn's
`sklearn.manifold.Isomap`, which implements the published algorithm (k-NN graph,
Dijkstra/Floyd-Warshall shortest paths, classical MDS) - per the project's own
guideline to prefer established implementations over reinventing algorithms that
already have one. This module adds: input validation, a cube-shaped API
(pixel-wise application to an (H, W, C) image with reshape back to (H, W, d)),
and docstrings connecting each argument to the maths in
documentation/02_isomap.md. It does not reimplement Isomap's internals.

See documentation/02_isomap.md for the full mathematical derivation.
"""
import time
import tracemalloc

import numpy as np
from sklearn.manifold import Isomap

from .preprocessing import cube_to_pixels, pixels_to_cube, standardize, validate_cube


def fit_isomap_pixels(pixels, n_neighbors=10, n_components=3, eigen_solver="auto",
                       random_state=42, **isomap_kwargs):
    """Fit Isomap on a (n_pixels, n_channels) matrix. Thin, validated wrapper.

    Parameters
    ----------
    pixels : array, shape (n_pixels, n_channels)
        Each row is one pixel's spectrum. Standardise first (see
        preprocessing.standardize) if channels are on very different scales.
    n_neighbors : int
        k in the k-NN graph (Stage-1, Step 1). See documentation/02_isomap.md
        for the bias/variance trade-off (too small -> disconnected graph;
        too large -> short-circuit edges across the manifold).
    n_components : int
        Output dimensionality d (Stage-1, Step 3: top d eigenvectors kept).
    eigen_solver : str
        Passed through to scikit-learn ('auto', 'arpack', 'dense'). 'dense' is
        exact but memory-heavy; use it for small/medium pixel counts where
        reproducibility of the exact eigendecomposition matters more than speed.
    random_state : int
        Only affects the 'arpack' solver's internal iterative start vector.
        Classical MDS itself is deterministic given the distance matrix - this
        argument exists for reproducibility of the solver, not of the maths.

    Returns
    -------
    embedding : array, shape (n_pixels, n_components)
    model : the fitted sklearn.manifold.Isomap instance (kept for
        model.reconstruction_error() and model.nbrs_, used by evaluation.py)
    diagnostics : dict with fit_time_seconds, peak_memory_mb, and a warning
        if the k-NN graph was disconnected (Isomap fits anyway using
        np.inf-safe shortest paths, but a disconnected graph means the
        geodesic-distance assumption has broken down for some pixel pairs)
    """
    pixels = np.asarray(pixels, dtype="float64")
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2-D (n_pixels, n_channels) array, got shape {pixels.shape}")
    n_pixels, n_channels = pixels.shape
    if n_pixels <= n_neighbors:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be smaller than the number of "
            f"pixels ({n_pixels})."
        )
    if n_components >= n_channels:
        raise ValueError(
            f"n_components ({n_components}) should be smaller than the number "
            f"of input channels ({n_channels}) for this to be a reduction."
        )
    if not np.isfinite(pixels).all():
        raise ValueError("pixels contains NaN/inf; handle missing data before fitting.")

    model = Isomap(n_neighbors=n_neighbors, n_components=n_components,
                    eigen_solver=eigen_solver, **isomap_kwargs)

    tracemalloc.start()
    start = time.perf_counter()
    embedding = model.fit_transform(pixels)
    fit_time = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # A disconnected neighbourhood graph shows up as infinite entries in the
    # geodesic distance matrix sklearn computed internally; sklearn replaces
    # them with the max finite distance rather than erroring, which silently
    # weakens the geodesic-distance guarantee for the affected pixel pairs.
    # We surface this rather than letting it pass unnoticed.
    dist_matrix = model.dist_matrix_
    n_infinite = int(np.isinf(dist_matrix).sum())
    warning = None
    if n_infinite > 0:
        warning = (
            f"Neighbourhood graph was disconnected: {n_infinite} pixel-pair "
            f"geodesic distances were unreachable before sklearn's fallback. "
            f"Consider increasing n_neighbors."
        )

    diagnostics = {
        "fit_time_seconds": fit_time,
        "peak_memory_mb": peak_bytes / 1e6,
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "n_pixels": n_pixels,
        "n_channels": n_channels,
        "disconnected_graph_warning": warning,
    }
    return embedding, model, diagnostics


def fit_isomap_cube(cube, n_neighbors=10, n_components=3, standardize_channels=True,
                     random_state=42, **isomap_kwargs):
    """Stage-4 entry point: apply Isomap to a (H, W, C) image cube, pixel-wise.

    IMPORTANT (see documentation/01_concepts.md section 1.3 and
    documentation/02_isomap.md Stage 4): this treats every pixel as an
    independent point in C-dimensional spectral space. The k-NN graph is built
    purely on spectral distance between pixels - NOT on which pixels are
    spatially adjacent in the image. Two spectrally similar pixels from
    opposite corners of the scene can become graph neighbours here. This
    function does NOT, by itself, produce a spatially-regularised embedding;
    see experiments/03_isomap_on_synthetic_cube.py for a concrete
    demonstration of this and documentation/02_isomap.md for the
    spatially-constrained discussion.

    Returns
    -------
    embedding_cube : array, shape (H, W, n_components)
    model : the fitted sklearn.manifold.Isomap instance
    diagnostics : dict (see fit_isomap_pixels)
    """
    cube = validate_cube(cube)
    pixels, spatial_shape = cube_to_pixels(cube)
    if standardize_channels:
        pixels, _, _ = standardize(pixels)
    embedding, model, diagnostics = fit_isomap_pixels(
        pixels, n_neighbors=n_neighbors, n_components=n_components,
        random_state=random_state, **isomap_kwargs
    )
    embedding_cube = pixels_to_cube(embedding, spatial_shape)
    return embedding_cube, model, diagnostics
