"""
Step 2: Explore one Sentinel-2 L1C scene (.SAFE)

What it does:
  1. Finds the .SAFE folder (or .zip) in the current folder
  2. Prints info about every band (size, resolution, data type)
  3. Shows: true-colour image, false-colour image, NDVI map, cloud mask
  4. Prints how big the full scene is and how long it would take to downlink

Setup (once):
  pip install rasterio numpy matplotlib

Run:
  python scripts/explore_s2.py                      # auto-finds the .SAFE in the current folder
  python scripts/explore_s2.py path/to/scene.SAFE   # or give the path
"""

import glob
import os
import re
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

DISPLAY_SIZE = 1098      # read bands shrunk 10x for quick viewing (full is 10980)
DOWNLINK_MBPS = 320      # EnMAP X-band downlink, used as an example link speed
BITS_PER_PIXEL = 12      # Sentinel-2 stores 12-bit values

BAND_INFO = {
    "B01": ("Coastal blue", 60), "B02": ("Blue", 10), "B03": ("Green", 10),
    "B04": ("Red", 10), "B05": ("Red-edge 1", 20), "B06": ("Red-edge 2", 20),
    "B07": ("Red-edge 3", 20), "B08": ("NIR", 10), "B8A": ("Narrow NIR", 20),
    "B09": ("Water vapour", 60), "B10": ("Cirrus", 60), "B11": ("SWIR 1", 20),
    "B12": ("SWIR 2", 20),
}


# ---------- finding files ----------

def find_safe(arg=None):
    if arg:
        path = arg
    else:
        safes = glob.glob("*.SAFE") or glob.glob("*/*.SAFE")
        if safes:
            path = safes[0]
        else:
            zips = glob.glob("S2*_MSIL1C_*.zip")
            if not zips:
                sys.exit("No .SAFE folder or S2 .zip found. Put this script next to it.")
            print(f"Unzipping {zips[0]} ...")
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall(".")
            path = glob.glob("*.SAFE")[0]
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(".")
        path = glob.glob("*.SAFE")[0]
    return path


def band_paths(safe):
    files = glob.glob(os.path.join(safe, "GRANULE", "*", "IMG_DATA", "*_B*.jp2"))
    out = {}
    for f in files:
        m = re.search(r"_(B\d\d|B8A)\.jp2$", f)
        if m:
            out[m.group(1)] = f
    return dict(sorted(out.items()))


def read_offset(safe):
    """Newer products (baseline 04.00+) store value = reflectance*10000 + 1000."""
    mtd = os.path.join(safe, "MTD_MSIL1C.xml")
    quant, offset = 10000.0, 0.0
    if os.path.exists(mtd):
        txt = open(mtd, encoding="utf-8").read()
        q = re.search(r"<QUANTIFICATION_VALUE[^>]*>([\d.]+)<", txt)
        o = re.search(r"<RADIO_ADD_OFFSET[^>]*>(-?[\d.]+)<", txt)
        if q:
            quant = float(q.group(1))
        if o:
            offset = float(o.group(1))
    return quant, offset


def read_metadata(safe):
    mtd = os.path.join(safe, "MTD_MSIL1C.xml")
    info = {}
    if os.path.exists(mtd):
        txt = open(mtd, encoding="utf-8").read()
        for tag in ["PRODUCT_START_TIME", "SPACECRAFT_NAME", "PROCESSING_BASELINE",
                    "Cloud_Coverage_Assessment"]:
            m = re.search(rf"<{tag}[^>]*>([^<]+)<", txt)
            if m:
                info[tag] = m.group(1)
    return info


# ---------- reading ----------

def read_band(path, size=DISPLAY_SIZE):
    with rasterio.open(path) as src:
        return src.read(1, out_shape=(size, size), resampling=Resampling.average)


def to_reflectance(raw, quant, offset):
    refl = (raw.astype("float32") + offset) / quant
    refl[raw == 0] = np.nan          # 0 = no data (black swath edge)
    return refl


def stretch(img, lo=2, hi=98):
    """Brighten an image for display by clipping extreme values."""
    valid = img[~np.isnan(img)]
    if valid.size == 0:
        return np.zeros_like(img)
    a, b = np.percentile(valid, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-9), 0, 1)


def cloud_mask(safe, size=DISPLAY_SIZE):
    """MSK_CLASSI: band 1 = opaque clouds, band 2 = cirrus (baseline 04.00+)."""
    f = glob.glob(os.path.join(safe, "GRANULE", "*", "QI_DATA", "MSK_CLASSI_B00.jp2"))
    if not f:
        return None
    with rasterio.open(f[0]) as src:
        m = src.read(out_shape=(src.count, size, size), resampling=Resampling.nearest)
    return (m[0] > 0) | (m[1] > 0) if m.shape[0] > 1 else m[0] > 0


# ---------- main ----------

def main():
    safe = find_safe(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nScene: {os.path.basename(safe)}\n")

    for k, v in read_metadata(safe).items():
        print(f"  {k:28s} {v}")

    paths = band_paths(safe)
    quant, offset = read_offset(safe)
    print(f"\n  Reflectance = (value + {offset:g}) / {quant:g}\n")

    # band table
    print(f"  {'Band':5s} {'Name':14s} {'Res':>5s} {'Pixels':>13s} {'Type':>7s} {'File MB':>8s}")
    total_bits = 0
    crs = None
    for b, p in paths.items():
        with rasterio.open(p) as src:
            h, w, dt = src.height, src.width, src.dtypes[0]
            crs = crs or src.crs
        name, res = BAND_INFO.get(b, ("?", "?"))
        mb = os.path.getsize(p) / 1e6
        total_bits += h * w * BITS_PER_PIXEL
        print(f"  {b:5s} {name:14s} {res:>4}m {h:>6}x{w:<6} {dt:>7s} {mb:8.1f}")

    raw_mb = total_bits / 8 / 1e6
    print(f"\n  Coordinate system: {crs}")
    print(f"\n  Raw scene size (all 13 bands, 12-bit): {raw_mb:,.0f} MB")
    print(f"  Time to downlink at {DOWNLINK_MBPS} Mbit/s: {total_bits / 1e6 / DOWNLINK_MBPS:.1f} s")
    print("  -> this is the number your project tries to shrink.\n")

    # load the 4 key bands (shrunk for display)
    need = ["B02", "B03", "B04", "B08"]
    missing = [b for b in need if b not in paths]
    if missing:
        sys.exit(f"Missing bands: {missing}")
    B = {b: to_reflectance(read_band(paths[b]), quant, offset) for b in need}

    true_colour = np.dstack([stretch(B["B04"]), stretch(B["B03"]), stretch(B["B02"])])
    false_colour = np.dstack([stretch(B["B08"]), stretch(B["B04"]), stretch(B["B03"])])
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (B["B08"] - B["B04"]) / (B["B08"] + B["B04"])
    clouds = cloud_mask(safe)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].imshow(true_colour);  ax[0, 0].set_title("True colour (B04, B03, B02)")
    ax[0, 1].imshow(false_colour); ax[0, 1].set_title("False colour (B08, B04, B03)\nplants = red")
    im = ax[1, 0].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    ax[1, 0].set_title("NDVI (green = vegetation)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)
    if clouds is not None:
        ax[1, 1].imshow(clouds, cmap="gray")
        ax[1, 1].set_title(f"Cloud mask (white = cloud, {100 * clouds.mean():.1f}%)")
    else:
        ax[1, 1].text(0.5, 0.5, "No cloud mask found", ha="center")
    for a in ax.flat:
        a.axis("off")
    plt.suptitle(os.path.basename(safe)[:60], fontsize=10)
    plt.tight_layout()
    plt.savefig("scene_overview.png", dpi=120)
    print("  Saved scene_overview.png")
    plt.show()


if __name__ == "__main__":
    main()