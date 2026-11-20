"""
Locally Linear Embedding (LLE) for spatial-spectral dimensionality reduction.

Original algorithm: Roweis, Saul (2000), "Nonlinear Dimensionality Reduction
by Locally Linear Embedding", Science 290(5500), 2323-2327.
DOI: 10.1126/science.290.5500.2323

Implementation status: EXACT REPRODUCTION via
sklearn.manifold.LocallyLinearEmbedding(method="standard") - the originally
published algorithm, not scikit-learn's other variants (modified/hessian/ltsa).
See documentation/01_lle.md for the full derivation and
documentation/02_stage2_basic_example.md for how LLE's failure mode at large
k differs qualitatively from Isomap's (found by direct comparison, not assumed).
"""
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src`
from src.preprocessing import cube_to_pixels, pixels_to_cube, standardize, validate_cube


def fit_lle_pixels(pixels, n_neighbors=12, n_components=3, reg=1e-3, random_state=42, eigen_solver="auto"):
    """Fit LLE on a (n_pixels, n_channels) matrix.

    Parameters
    ----------
    pixels : array, shape (n_pixels, n_channels)
    n_neighbors : int
        Size of the locally-flat patch (Stage 1, Step 1). With only 7 input
        channels, n_neighbors > 7 is likely needed for a stable local fit -
        this REQUIRES `reg` (below) rather than leaving it at risk of being
        ill-conditioned; checked directly in documentation, not assumed safe.
    n_components : int
    eigen_solver : str
        'auto' (default, used for the synthetic and Raichur results) or 'dense'. On the Indian Pines
        data (200 bands) with scikit-learn 1.5.2 / SciPy 1.13.1, 'auto' (ARPACK) returned a degenerate
        embedding (trustworthiness 0.54) while 'dense' gave 0.94, so the hyperspectral scripts use 'dense'.
    reg : float
        Regularisation added to the local covariance matrix in Stage 1's
        Step 2, required whenever n_neighbors > n_channels. See
        documentation/01_lle.md for why this hyperparameter exists at all -
        neither Isomap nor Kernel PCA has an equivalent.
    random_state : int
        Used by the eigen-solver.

    Returns
    -------
    embedding : array, shape (n_pixels, n_components)
    model : the fitted sklearn.manifold.LocallyLinearEmbedding instance
        (exposes `.reconstruction_error_`, the Stage-1 Step-2 reconstruction
        cost - see documentation/01_lle.md's objective function)
    diagnostics : dict with timing/memory and the settings used
    """
    pixels = np.asarray(pixels, dtype="float64")
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2-D (n_pixels, n_channels) array, got shape {pixels.shape}")
    n_pixels, n_channels = pixels.shape
    if n_pixels <= n_neighbors:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be smaller than n_pixels ({n_pixels}).")
    if n_components >= n_channels:
        raise ValueError(f"n_components ({n_components}) should be smaller than n_channels ({n_channels}).")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels contains NaN/inf; handle missing data before fitting.")
    if n_neighbors > n_channels and reg <= 0:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) exceeds n_channels ({n_channels}) - the local "
            f"reconstruction step (Stage 1, Step 2) will be ill-conditioned without "
            f"regularisation. Pass reg > 0 (e.g. 1e-3)."
        )

    model = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                                     method="standard", reg=reg, random_state=random_state,
                                     eigen_solver=eigen_solver)

    tracemalloc.start()
    start = time.perf_counter()
    embedding = model.fit_transform(pixels)
    fit_time = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    diagnostics = {
        "fit_time_seconds": fit_time,
        "peak_memory_mb": peak_bytes / 1e6,
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "n_pixels": n_pixels,
        "n_channels": n_channels,
        "reg": reg,
        "reconstruction_error": float(model.reconstruction_error_),
    }
    return embedding, model, diagnostics


def fit_lle_cube(cube, n_neighbors=12, n_components=3, reg=1e-3,
                  standardize_channels=True, random_state=42):
    """Stage-4 entry point: apply LLE to a (H, W, C) image cube, pixel-wise.

    Same pixel-wise caveat as every other technique: neighbours are found by
    spectral distance only, with no spatial position information surviving
    the cube_to_pixels flattening. Unlike Isomap, there is no chained-distance
    graph here, but the k-NN search itself (Stage 1, Step 1) is still purely
    spectral - see 01_Isomap/visualisations/03c_spectral_vs_spatial_neighbors.png
    for the concrete demonstration (not re-run here; the underlying k-NN
    search mechanism is the same kind of spectral-only lookup).
    """
    cube = validate_cube(cube)
    pixels, spatial_shape = cube_to_pixels(cube)
    if standardize_channels:
        pixels, _, _ = standardize(pixels)
    embedding, model, diagnostics = fit_lle_pixels(
        pixels, n_neighbors=n_neighbors, n_components=n_components, reg=reg, random_state=random_state
    )
    embedding_cube = pixels_to_cube(embedding, spatial_shape)
    return embedding_cube, model, diagnostics
