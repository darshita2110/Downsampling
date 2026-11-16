"""
Shared preprocessing utilities: converting a spatial-spectral image cube
(Height, Width, Channels) to a flat (n_pixels, Channels) matrix and back, plus
basic scaling and validity checks. Every technique module (isomap.py,
kernel_pca.py, ...) uses these same functions, so preprocessing is identical
across the whole comparison (required by the project's Section 5: "keep
preprocessing consistent").
"""
import numpy as np


def cube_to_pixels(cube):
    """(H, W, C) -> (H*W, C). Row-major flatten, so pixel index = row*W + col.

    Keeping this one function as the single source of truth for the flattening
    order matters: every technique must flatten and (later) reshape the same
    way, or comparisons between techniques' output images would be silently
    misaligned.
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a (H, W, C) cube, got shape {cube.shape}")
    height, width, n_channels = cube.shape
    pixels = cube.reshape(height * width, n_channels)
    return pixels, (height, width)


def pixels_to_cube(pixels, spatial_shape):
    """(n_pixels, C) -> (H, W, C), inverse of cube_to_pixels."""
    height, width = spatial_shape
    pixels = np.asarray(pixels)
    if pixels.shape[0] != height * width:
        raise ValueError(
            f"pixels has {pixels.shape[0]} rows, expected {height * width} "
            f"for spatial shape {spatial_shape}"
        )
    n_components = pixels.shape[1]
    return pixels.reshape(height, width, n_components)


def validate_cube(cube, expected_channels=None):
    """Basic input validation shared by every technique's Stage-4 entry point."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D (H, W, C) array, got {cube.ndim}-D shape {cube.shape}")
    if expected_channels is not None and cube.shape[-1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {cube.shape[-1]}")
    if not np.isfinite(cube).all():
        n_bad = (~np.isfinite(cube)).sum()
        raise ValueError(
            f"Cube contains {n_bad} non-finite values (NaN/inf). "
            f"Handle missing data explicitly before dimensionality reduction "
            f"(e.g. mask and exclude, or impute) - do not silently drop/zero-fill."
        )
    return cube


def standardize(pixels, mean=None, std=None):
    """Zero-mean, unit-variance scaling per channel. Returns (scaled, mean, std)
    so the same mean/std can be reapplied to held-out data (e.g. a test tile).

    Manifold-learning methods build neighbourhoods from distances, and raw
    satellite channels (e.g. B02 ~ 0-3000 scale, NDVI ~ -1 to 1) have wildly
    different numeric ranges - without standardising, distance would be
    dominated by whichever channel has the largest raw numbers, regardless of
    its actual information content.
    """
    pixels = np.asarray(pixels, dtype="float64")
    if mean is None:
        mean = pixels.mean(axis=0)
    if std is None:
        std = pixels.std(axis=0)
        std[std == 0] = 1.0  # guard against a constant channel
    return (pixels - mean) / std, mean, std
