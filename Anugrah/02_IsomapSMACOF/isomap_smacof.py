"""
Isomap with SMACOF: geodesic distances (Isomap Steps 1-2) embedded via
iterative Stress majorization (SMACOF) instead of classical MDS's closed-form
eigendecomposition (Isomap Step 3). See documentation/01_isomap_smacof.md for
the full derivation and citations.

Implementation status: COMBINATION OF TWO EXACT, ESTABLISHED IMPLEMENTATIONS.
`sklearn.manifold.Isomap` computes the geodesic distance matrix (we discard its
own classical-MDS embedding and keep only `dist_matrix_`); `sklearn.manifold.MDS`
with a precomputed dissimilarity matrix runs SMACOF on those distances. Neither
algorithm is reimplemented here.

Version-compatibility note: scikit-learn renamed MDS's constructor arguments
between versions - older releases (this project's own requirements.txt pins
scikit-learn==1.5.2) use `dissimilarity="precomputed"` and a boolean `metric=`
flag for metric-vs-non-metric MDS; newer releases (>=1.7-ish) use
`metric="precomputed"` and a separate `metric_mds=` boolean, with the old
names deprecated. Both forms exist across machines this project might run on
(verified directly: this sandbox has scikit-learn 1.9.1, which already uses
the new names), so `_build_mds` below detects which one the installed version
expects via introspection rather than assuming either - avoids a confusing
FutureWarning/TypeError depending on whose machine this runs on.
"""
import inspect
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.manifold import MDS, Isomap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `src`
from src.preprocessing import cube_to_pixels, pixels_to_cube, standardize, validate_cube


def _build_mds(n_components, n_init, max_iter, eps, random_state):
    """Construct sklearn.manifold.MDS for a precomputed, metric (not
    non-metric) SMACOF run, using whichever constructor argument names the
    installed scikit-learn version expects.

    Found on the real-data run (not just the dev sandbox): this project's own
    pinned version, scikit-learn==1.5.2, has NO `init` constructor parameter
    at all (confirmed via `inspect.signature`: its MDS.__init__ exposes
    n_components, metric, n_init, max_iter, verbose, eps, n_jobs,
    random_state, dissimilarity, normalized_stress - no `init`). The original
    version of this function added `init="random"` unconditionally, which
    crashed with `TypeError: MDS.__init__() got an unexpected keyword
    argument 'init'` on exactly the version this project is pinned to - the
    version-compatibility check below was incomplete, checking only for
    `metric_mds` (which args are RENAMED) and missing that `init` may not
    EXIST at all on some versions. Fixed by checking "init" in params before
    adding it, the same defensive pattern as every other kwarg here; omitting
    it is harmless since scikit-learn 1.5.2's MDS already initialises
    randomly internally when none is given.
    """
    params = inspect.signature(MDS.__init__).parameters
    kwargs = dict(n_components=n_components, n_init=n_init, max_iter=max_iter,
                  eps=eps, random_state=random_state)
    if "init" in params:
        kwargs["init"] = "random"
    if "metric_mds" in params:  # newer scikit-learn
        kwargs["metric"] = "precomputed"
        kwargs["metric_mds"] = True
        if "normalized_stress" in params:
            kwargs["normalized_stress"] = False
    else:  # older scikit-learn (this project's requirements.txt pin, 1.5.2)
        kwargs["dissimilarity"] = "precomputed"
        kwargs["metric"] = True
        if "normalized_stress" in params:
            kwargs["normalized_stress"] = False
    return MDS(**kwargs)


def fit_isomap_smacof_pixels(pixels, n_neighbors=10, n_components=3, n_init=4,
                              max_iter=300, eps=1e-6, random_state=42):
    """Fit Isomap-with-SMACOF on a (n_pixels, n_channels) matrix.

    Parameters
    ----------
    pixels : array, shape (n_pixels, n_channels)
    n_neighbors, n_components : as in plain Isomap (see 01_Isomap/isomap.py) -
        control the geodesic-distance graph (Steps 1-2, unchanged from Isomap).
    n_init : int
        SMACOF random restarts; the best-stress result is kept. Unlike plain
        Isomap, SMACOF genuinely has local minima - see documentation.
    max_iter, eps : SMACOF's own iteration cap / convergence tolerance.
    random_state : int
        Here this really changes the SMACOF initial configuration (not just an
        eigen-solver start vector, as in plain Isomap).

    Returns
    -------
    embedding : array, shape (n_pixels, n_components)
    info : dict with the fitted Isomap model (`isomap_model`, for its
        `dist_matrix_` / graph diagnostics) and the fitted MDS model
        (`mds_model`, for `.stress_`, `.n_iter_`)
    diagnostics : dict - timing/memory for the whole pipeline, split into the
        geodesic-distance step and the SMACOF step so we can see which
        dominates cost (see Stage 5 evaluation)
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

    tracemalloc.start()

    # Steps 1-2 (shared with plain Isomap): k-NN graph + geodesic distances.
    # We fit a full Isomap model (simplest way to get an exact, established
    # geodesic-distance computation) but use ONLY its dist_matrix_ - its own
    # classical-MDS embedding is discarded; SMACOF below replaces it.
    start = time.perf_counter()
    isomap_model = Isomap(n_neighbors=n_neighbors, n_components=n_components).fit(pixels)
    geodesic_time = time.perf_counter() - start
    dist_matrix = isomap_model.dist_matrix_
    n_infinite = int(np.isinf(dist_matrix).sum())
    disconnected_warning = None
    if n_infinite > 0:
        disconnected_warning = (
            f"Neighbourhood graph was disconnected: {n_infinite} pixel-pair "
            f"geodesic distances were unreachable. Consider increasing n_neighbors."
        )

    # Step 3 (this technique's change): SMACOF instead of classical MDS.
    start = time.perf_counter()
    mds_model = _build_mds(n_components, n_init, max_iter, eps, random_state)
    embedding = mds_model.fit_transform(dist_matrix)
    smacof_time = time.perf_counter() - start

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    diagnostics = {
        "geodesic_distance_time_seconds": geodesic_time,
        "smacof_time_seconds": smacof_time,
        "fit_time_seconds": geodesic_time + smacof_time,
        "peak_memory_mb": peak_bytes / 1e6,
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "n_pixels": n_pixels,
        "n_channels": n_channels,
        "n_init": n_init,
        "max_iter": max_iter,
        "smacof_n_iter": int(mds_model.n_iter_),
        "smacof_stress": float(mds_model.stress_),
        "disconnected_graph_warning": disconnected_warning,
    }
    info = {"isomap_model": isomap_model, "mds_model": mds_model}
    return embedding, info, diagnostics


def fit_isomap_smacof_cube(cube, n_neighbors=10, n_components=3, standardize_channels=True,
                            n_init=4, max_iter=300, eps=1e-6, random_state=42):
    """Stage-4 entry point: apply Isomap-with-SMACOF to a (H, W, C) image cube,
    pixel-wise. Same spatial-blindness caveat as plain Isomap applies here too
    (see 01_Isomap/isomap.py's fit_isomap_cube docstring) - Steps 1-2 are
    identical, so the neighbourhood graph is built on spectral distance only.
    """
    cube = validate_cube(cube)
    pixels, spatial_shape = cube_to_pixels(cube)
    if standardize_channels:
        pixels, _, _ = standardize(pixels)
    embedding, info, diagnostics = fit_isomap_smacof_pixels(
        pixels, n_neighbors=n_neighbors, n_components=n_components,
        n_init=n_init, max_iter=max_iter, eps=eps, random_state=random_state
    )
    embedding_cube = pixels_to_cube(embedding, spatial_shape)
    return embedding_cube, info, diagnostics
