"""
Kernel PCA for spatial-spectral dimensionality reduction.

Original algorithm: Scholkopf, Smola, Muller (1998), "Nonlinear Component
Analysis as a Kernel Eigenvalue Problem", Neural Computation 10(5), 1299-1319.
DOI: 10.1162/089976698300017467

Implementation status: EXACT REPRODUCTION via sklearn.decomposition.KernelPCA.
See documentation/01_kernel_pca.md for the full derivation and
documentation/02_stage2_basic_example.md for a real negative result (RBF
Kernel PCA cannot unroll a Swiss roll at any gamma tested) found before this
module was applied to our actual data.
"""
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.decomposition import KernelPCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src`
from src.preprocessing import cube_to_pixels, pixels_to_cube, standardize, validate_cube


def fit_kernel_pca_pixels(pixels, n_components=3, kernel="rbf", gamma=None, random_state=42):
    """Fit Kernel PCA on a (n_pixels, n_channels) matrix.

    Parameters
    ----------
    pixels : array, shape (n_pixels, n_channels)
    n_components : int
    kernel : str
        'rbf' used throughout this project (see documentation/01_kernel_pca.md);
        scikit-learn also supports 'poly', 'sigmoid', 'cosine', 'linear'.
    gamma : float or None
        RBF kernel bandwidth. None -> scikit-learn's default (1/n_features).
        See the gamma-sweep finding in documentation/02_stage2_basic_example.md
        before assuming a default is adequate for new data.
    random_state : int
        Used by the eigen-solver when it is randomised (e.g. 'arpack' or
        'randomized'); the eigendecomposition itself is otherwise deterministic
        given the kernel matrix, same caveat as Isomap's random_state.

    Returns
    -------
    embedding : array, shape (n_pixels, n_components)
    model : the fitted sklearn.decomposition.KernelPCA instance
    diagnostics : dict with timing/memory and the settings used
    """
    pixels = np.asarray(pixels, dtype="float64")
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2-D (n_pixels, n_channels) array, got shape {pixels.shape}")
    n_pixels, n_channels = pixels.shape
    if n_components >= n_channels:
        raise ValueError(f"n_components ({n_components}) should be smaller than n_channels ({n_channels}).")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels contains NaN/inf; handle missing data before fitting.")

    model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma,
                       random_state=random_state, eigen_solver="auto")

    tracemalloc.start()
    start = time.perf_counter()
    embedding = model.fit_transform(pixels)
    fit_time = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    diagnostics = {
        "fit_time_seconds": fit_time,
        "peak_memory_mb": peak_bytes / 1e6,
        "n_components": n_components,
        "n_pixels": n_pixels,
        "n_channels": n_channels,
        "kernel": kernel,
        "gamma_used": model.gamma if model.gamma is not None else 1.0 / n_channels,
    }
    return embedding, model, diagnostics


def fit_kernel_pca_cube(cube, n_components=3, kernel="rbf", gamma=None,
                         standardize_channels=True, random_state=42):
    """Stage-4 entry point: apply Kernel PCA to a (H, W, C) image cube, pixel-wise.

    Same pixel-wise caveat as every other technique in this project: the
    kernel similarity is computed from spectral values only, with no spatial
    position information surviving the cube_to_pixels flattening.
    """
    cube = validate_cube(cube)
    pixels, spatial_shape = cube_to_pixels(cube)
    if standardize_channels:
        pixels, _, _ = standardize(pixels)
    embedding, model, diagnostics = fit_kernel_pca_pixels(
        pixels, n_components=n_components, kernel=kernel, gamma=gamma, random_state=random_state
    )
    embedding_cube = pixels_to_cube(embedding, spatial_shape)
    return embedding_cube, model, diagnostics


def explained_variance_ratio(pixels, n_components, kernel="rbf", gamma=None, random_state=42):
    """Explained-variance ratio for Kernel PCA - UNLIKE Isomap (see
    src/evaluation.explained_variance_ratio's long comment on why this is not
    computable there), this genuinely IS computable for Kernel PCA: verified
    directly that sklearn.decomposition.KernelPCA keeps the FULL eigenvalue
    spectrum when fit with n_components=None (checked: 199 eigenvalues for
    200 samples, not truncated to whatever n_components was requested, unlike
    Isomap's internal KernelPCA step which always truncates). So the "kept /
    total" ratio here is a real measurement, not an artifact - but it
    requires a SEPARATE fit with n_components=None to see the full spectrum,
    since the main model (fit with n_components=3) only ever exposes 3
    eigenvalues, same truncation Isomap has, just for a different reason
    (that model was never asked to keep more).

    This second fit repeats the same kernel-eigendecomposition cost as the
    main fit (not negligible at real-data pixel counts) - called separately,
    not automatically inside fit_kernel_pca_pixels, so that cost is only paid
    when this specific metric is wanted (see Stage 5 evaluation scripts).
    """
    pixels = np.asarray(pixels, dtype="float64")
    full_model = KernelPCA(n_components=None, kernel=kernel, gamma=gamma,
                            random_state=random_state, eigen_solver="auto")
    full_model.fit(pixels)
    eigenvalues = np.asarray(full_model.eigenvalues_)
    positive = eigenvalues[eigenvalues > 0]
    if positive.sum() == 0:
        return None
    return float(positive[:n_components].sum() / positive.sum())
