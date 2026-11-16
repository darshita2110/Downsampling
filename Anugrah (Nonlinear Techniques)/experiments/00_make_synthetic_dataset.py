"""
Generate a small synthetic 7-channel "pretend satellite image" for our first experiments.

Why synthetic, and why built this way
--------------------------------------
Before touching real Sentinel-2 data, we want a dataset where WE know the ground
truth: how many independent drivers actually generated the 7 channels, and how
those drivers were combined. That lets us check, for every technique, whether it
recovers something close to the true number of drivers and their shape - something
we can never fully verify on real data because we don't know its true dimensionality.

The construction below:
  1. Starts from 2 independent LATENT variables per pixel, laid out smoothly over
     a spatial grid (so neighbouring pixels have similar latent values, like a
     real landscape):
         t  -> a "vegetation / moisture" driver, varying as a smooth 2-D wave
         s  -> a "temperature-like" driver, varying as a smoother, slower wave
  2. Wraps (t, s) through a NON-LINEAR map (a "Swiss-roll"-style twist) to get an
     intermediate 3-D curved surface embedded in 3-D space. This is the classic
     manifold-learning toy example: two numbers (t, s) are enough to describe
     every point, but the surface they trace out is curved, so a straight
     (linear) ruler through the 3-D space does not measure distance along it
     correctly. This is exactly why PCA struggles here - more in README.md.
  3. Passes that curved 3-D surface through 7 different non-linear readout
     functions to produce our 7 pretend bands: B02, B03, B04, B08, NDVI, NDWI, LST.
     The readout functions are simple, fixed, and DOCUMENTED below - they are not
     meant to physically model real reflectance, only to give 7 channels that are
     redundant (because they all come from the same 2 underlying drivers) AND
     non-linearly entangled (because of the Swiss-roll twist and saturating
     functions like tanh/sigmoid).
  4. Adds a small amount of Gaussian sensor noise.

Ground truth saved alongside the data (never used by the dimensionality-reduction
code - only for evaluation/debugging):
    - the 2 latent drivers (t, s) per pixel
    - the "unrolled" 2-D coordinates used to lay out the Swiss-roll (for plotting)

Output
------
data/synthetic_v1/
    cube.npy          float32 array, shape (H, W, 7)   <- the "image"
    channel_names.txt one name per line, same order as the last axis
    latents.npy        float32 array, shape (H, W, 2)   <- ground truth, debugging only
    swiss_roll_color.npy  float32 array, shape (H, W)   <- the usual Isomap "unrolled position" colour, for later visual checks
    metadata.json      generation parameters, for reproducibility

Run
---
    cd "Anugrah (Nonlinear Techniques)"
    python experiments/00_make_synthetic_dataset.py
"""
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- settings
HEIGHT, WIDTH = 40, 40          # 1,600 pixels total: big enough to see structure, small enough to be fast
RANDOM_SEED = 42
NOISE_STD = 0.02                # Gaussian sensor noise added to every channel, after scaling to ~unit range

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic_v1"
CHANNEL_NAMES = ["B02", "B03", "B04", "B08", "NDVI", "NDWI", "LST"]


def make_latents(height, width, rng):
    """Two smooth latent fields over a (height, width) grid, each roughly in [0, 1].

    Low-frequency sine/cosine fields are used (rather than independent random noise
    per pixel) so that nearby pixels have similar latent values - mimicking how real
    land cover changes smoothly across a scene rather than jumping randomly pixel to
    pixel. This is what later lets Isomap-style methods use "neighbouring pixels are
    related" as a meaningful signal.
    """
    row = np.linspace(0, 1, height)
    col = np.linspace(0, 1, width)
    grid_row, grid_col = np.meshgrid(row, col, indexing="ij")

    # t: a vegetation/moisture-like driver - two bumps across the scene
    t = 0.5 + 0.5 * np.sin(2.3 * np.pi * grid_row) * np.cos(1.6 * np.pi * grid_col)
    # s: a temperature-like driver - a slower, single gradient plus mild texture
    s = 0.5 + 0.35 * np.cos(1.1 * np.pi * grid_col) + 0.15 * np.sin(3.0 * np.pi * grid_row)
    s = (s - s.min()) / (s.max() - s.min())

    # small independent jitter so pixels are not perfectly deterministic functions
    # of row/col alone (keeps PCA vs. non-linear comparisons from being too easy)
    t = np.clip(t + rng.normal(0, 0.02, size=t.shape), 0, 1)
    s = np.clip(s + rng.normal(0, 0.02, size=s.shape), 0, 1)
    return t.astype("f4"), s.astype("f4")


def swiss_roll(t, s):
    """Map two latent coordinates (t, s) onto a curved 3-D surface (the classic
    'Swiss roll' manifold used throughout the manifold-learning literature).

    phi is the angle of the roll: as t goes from 0 to 1, we sweep through 1.5 turns
    of a spiral, so points that are close in 3-D space (small turns apart) can
    actually be far apart along the surface (several turns apart) - this is the
    structure that trips up straight-line (Euclidean / PCA) distances and that
    Isomap's geodesic distance is designed to fix.
    """
    phi = 1.5 * np.pi * (1 + 2 * t)          # angle, ~1.5 full turns
    radius = 1.0 + phi / (1.5 * np.pi * 3)    # radius grows slowly with angle
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    z = 2.0 * s
    return x, y, z, phi  # phi is returned as the "ground-truth unrolled colour"


def readout_channels(x, y, z, rng):
    """Turn the curved 3-D surface (x, y, z) into 7 correlated, non-linearly
    entangled 'bands'. Each formula below is an arbitrary but FIXED non-linear
    mix - the point is only that all 7 channels are functions of the SAME 3
    underlying numbers, so the true intrinsic dimensionality is low (<=3, and by
    construction actually 2, since x,y,z themselves came from (t,s)), while the
    observed channels look like 7 independent-ish numbers unless you account for
    the non-linear relationship.
    """
    def norm01(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-8)

    b02 = norm01(np.tanh(1.2 * x + 0.3 * z))
    b03 = norm01(np.tanh(0.8 * y - 0.4 * x))
    b04 = norm01(1 / (1 + np.exp(-(2.0 * z - 1.0 * x))))     # logistic/sigmoid mix
    b08 = norm01(np.sin(1.3 * x) * np.cos(0.9 * y) + 0.5 * z)
    ndvi = norm01((b08 - b04) / (b08 + b04 + 1e-3))           # kept here for narrative realism
    ndwi = norm01((b03 - b08) / (b03 + b08 + 1e-3))
    lst = norm01(np.exp(-((x - 0.5) ** 2 + (y + 0.3) ** 2)) + 0.6 * z)

    channels = np.stack([b02, b03, b04, b08, ndvi, ndwi, lst], axis=-1).astype("f4")
    channels += rng.normal(0, NOISE_STD, size=channels.shape).astype("f4")
    return np.clip(channels, 0, 1)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    t, s = make_latents(HEIGHT, WIDTH, rng)
    x, y, z, phi = swiss_roll(t, s)
    cube = readout_channels(x, y, z, rng)

    assert cube.shape == (HEIGHT, WIDTH, 7), cube.shape
    assert not np.isnan(cube).any(), "NaNs produced during synthesis - check NDVI/NDWI denominators"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "cube.npy", cube)
    np.save(OUT_DIR / "latents.npy", np.stack([t, s], axis=-1).astype("f4"))
    np.save(OUT_DIR / "swiss_roll_color.npy", phi.astype("f4"))
    (OUT_DIR / "channel_names.txt").write_text("\n".join(CHANNEL_NAMES) + "\n")
    metadata = {
        "height": HEIGHT, "width": WIDTH, "n_channels": 7,
        "channel_names": CHANNEL_NAMES, "random_seed": RANDOM_SEED,
        "noise_std": NOISE_STD, "true_latent_dimensionality": 2,
        "generator": "Swiss-roll-style synthetic cube, see docstring of this file",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved synthetic cube: {cube.shape} -> {OUT_DIR / 'cube.npy'}")
    print(f"Channel order: {CHANNEL_NAMES}")
    print(f"Value range per channel (min, max):")
    for name, idx in zip(CHANNEL_NAMES, range(7)):
        channel = cube[..., idx]
        print(f"  {name:5s}  min={channel.min():.3f}  max={channel.max():.3f}  std={channel.std():.3f}")
    print(f"\nTrue intrinsic dimensionality used to build this cube: 2 "
          f"(latents t, s) -> 3 (Swiss-roll x,y,z) -> 7 observed channels.")


if __name__ == "__main__":
    main()
