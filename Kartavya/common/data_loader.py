"""
Loading the Raichur Sentinel-2 products for Kartavya experiments.

IMPORTANT facts about the PNG files in data/processed/sentinel2/ (they come from scripts/main.py):

* They are *rendered matplotlib figures*, not raw rasters. Every PNG has a title and white
  margins, and the NDVI/NDWI PNGs also have a colour bar. So the folders have different
  pixel sizes (RGB/False_Color 3024 x 2852, NDVI/NDWI 2962 x 3181 for 2026-09-17), and a
  pixel at (row, col) in one PNG is NOT the same ground point as (row, col) in another.
  -> We cut out the map panel of every figure and resample all panels onto one common grid.

* RGB  = (B04, B03, B02) and False_Color = (B08, B04, B03), each band contrast-stretched
  to 0-255 with a 2-98 percentile stretch. These are 8-bit *display composites*, not the
  original Sentinel-2 spectral bands (no reflectance values, clipped, non-linear).

* NDVI and NDWI are *derived indices* drawn with a colour map (RdYlGn / RdBu, range -1..1).
  We invert the colour map to get the index value back (256 levels, step 2/256 = 0.0078).
  Pixels with no data were drawn white and become NaN.

The real reflectance bands (B02, B03, B04, B08) exist only in the raw NetCDF files
(data/raw/sentinel2_l1c/), which are loaded separately by `load_raw_scene`.
"""
import re
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # the figures are ~9 Mpx; switch off PIL's bomb guard

DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


# --------------------------------------------------------------------------- file listing
def list_sorted_images(folder, extension=".png"):
    """
    All image files in `folder`, sorted by their COMPLETE filename in ascending
    lexicographical (character-by-character) order. Python's sorted() on str does exactly that.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    names = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == extension)
    return [folder / n for n in names]


def build_index_mapping(dataset_dir, folders, expected_per_folder=None, logger=None):
    """
    Give every image an index (0, 1, 2, ...) from its sorted position inside its folder.
    The same index in different folders must be the same acquisition date, otherwise the
    channels of one "sample" would come from different days - we check that here.

    Returns {"folders": {folder: [{"index", "filename", "date"}, ...]}, "samples": [...]}.
    """
    mapping = {"dataset_dir": str(dataset_dir), "sort_rule": "ascending lexicographic order of full filename",
               "folders": {}, "samples": []}
    for f in folders:
        paths = list_sorted_images(Path(dataset_dir) / f)
        if expected_per_folder is not None and len(paths) != expected_per_folder:
            msg = f"{f}/ has {len(paths)} PNG files, expected {expected_per_folder}."
            if logger:
                logger.warning(msg + " Continuing with the files that exist.")
        entries = []
        for i, p in enumerate(paths):
            m = DATE_PATTERN.search(p.name)
            entries.append({"index": i, "filename": p.name, "date": m.group(0) if m else None})
        mapping["folders"][f] = entries

    counts = {f: len(v) for f, v in mapping["folders"].items()}
    n = min(counts.values()) if counts else 0
    if len(set(counts.values())) > 1 and logger:
        logger.warning(f"Folders have different image counts {counts}; using the first {n} indices of each.")

    for i in range(n):
        dates = {f: mapping["folders"][f][i]["date"] for f in folders}
        if len(set(dates.values())) != 1:
            raise ValueError(f"Index {i} points to different dates in different folders: {dates}. "
                             "Sorted orders do not line up, so channels would be mixed across days.")
        mapping["samples"].append({"index": i, "date": dates[folders[0]],
                                   "files": {f: mapping["folders"][f][i]["filename"] for f in folders}})
    return mapping


# --------------------------------------------------------------------- figure handling
def read_png_rgb(path):
    """Read a PNG as uint8 (H, W, 3). The alpha channel (all 255 in these figures) is dropped."""
    with Image.open(path) as im:
        info = {"mode": im.mode, "size_wh": im.size}
        arr = np.asarray(im.convert("RGBA"))
    info.update({"shape": arr.shape, "dtype": str(arr.dtype),
                 "min": int(arr[..., :3].min()), "max": int(arr[..., :3].max()),
                 "alpha_min": int(arr[..., 3].min()), "alpha_max": int(arr[..., 3].max())})
    return arr[..., :3], info


def _longest_run(mask):
    """Start and end (exclusive) of the longest run of True values in a 1-D boolean array."""
    best, start, best_span = 0, None, (0, 0)
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, best_span = i - start, (start, i)
            start = None
    return best_span


def find_map_panel(rgb, min_fill=0.5):
    """
    Locate the map inside a matplotlib figure: the largest block of rows/columns that are
    mostly non-white. The title (thin text) and colour bar (separated by a white gap, and
    narrower than the map) are excluded automatically.
    Returns (top, bottom, left, right) with bottom/right exclusive.
    """
    non_white = np.any(rgb != 255, axis=2)
    left, right = _longest_run(non_white.mean(axis=0) > min_fill)
    top, bottom = _longest_run(non_white[:, left:right].mean(axis=1) > min_fill)
    return top, bottom, left, right


def colormap_lut(cmap_name, n=256):
    """The n RGB colours (uint8) that matplotlib uses for this colour map."""
    import matplotlib
    cmap = matplotlib.colormaps[cmap_name].resampled(n)
    return np.round(cmap(np.arange(n))[:, :3] * 255).astype(np.int32)


def decode_colormap(rgb, cmap_name, vmin=-1.0, vmax=1.0, n=256, max_distance=6.0):
    """
    Turn a colour-mapped picture back into numbers.
    Each pixel colour is matched to the nearest of the n colour-map colours; colour index i
    stands for the value interval [vmin + i*w, vmin + (i+1)*w] with w = (vmax - vmin) / n, and we
    return its centre. Pure white (no data) and colours far from the map become NaN.

    Returns (values float32 with NaN, stats dict).
    """
    lut = colormap_lut(cmap_name, n)
    flat = rgb.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    uniq, inverse = np.unique(packed, return_inverse=True)        # decode each distinct colour once
    ucol = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.int32)
    d2 = ((ucol[:, None, :] - lut[None, :, :]) ** 2).sum(-1)      # (n_unique, n) squared distances
    nearest = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(uniq)), nearest])
    width = (vmax - vmin) / n
    uval = (vmin + (nearest + 0.5) * width).astype(np.float32)
    is_white = uniq == 0xFFFFFF
    bad = is_white | (dist > max_distance)
    uval[bad] = np.nan
    values = uval[inverse].reshape(rgb.shape[:2])
    stats = {"exact_match_pct": float(100 * (dist[inverse] == 0).mean()),
             "white_nodata_pct": float(100 * is_white[inverse].mean()),
             "unmatched_pct": float(100 * (bad & ~is_white)[inverse].mean()),
             "quantization_step": width}
    return values, stats


def resize_float(arr, size_wh):
    """Area-average resize of a float32 2-D array (PIL 'F' mode, BOX filter)."""
    return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize(size_wh, Image.Resampling.BOX))


def resize_with_mask(values, valid, size_wh, min_coverage=0.99):
    """
    Resize an array that has invalid pixels without letting them leak in:
    average (value * valid) and valid separately, then divide.
    A target pixel is valid only if >= min_coverage of its source area was valid.
    """
    filled = np.where(valid, values, 0).astype(np.float32)
    num = resize_float(filled, size_wh)
    cov = resize_float(valid.astype(np.float32), size_wh)
    out = np.where(cov > 0, num / np.maximum(cov, 1e-6), 0).astype(np.float32)
    return out, cov >= min_coverage


# ----------------------------------------------------------------- one PNG sample
def load_png_sample(dataset_dir, sample, folder_specs, target_width, logger=None):
    """
    Load one acquisition date from all four folders and put every channel on one grid.

    folder_specs: {"RGB": {"kind": "composite", "channels": [...]},
                   "NDVI": {"kind": "colormap", "cmap": "RdYlGn", "vmin": -1, "vmax": 1, "channel": "NDVI"}, ...}

    Returns dict with
      channels  : float32 (H, W, C) on the common grid, natural units
                  (composites 0..255 display levels, indices -1..1)
      names     : channel names, e.g. "RGB:R(B04)"
      valid     : bool (H, W) - pixel had data in every product
      inspection: per-file facts (size, mode, dtype, ranges, panel box, decode quality)
    """
    panels, inspection = {}, {}
    for folder, spec in folder_specs.items():
        path = Path(dataset_dir) / folder / sample["files"][folder]
        rgb, info = read_png_rgb(path)
        top, bottom, left, right = find_map_panel(rgb)
        info["map_panel_box_tblr"] = (top, bottom, left, right)
        info["map_panel_hw"] = (bottom - top, right - left)
        panels[folder] = rgb[top:bottom, left:right]
        inspection[folder] = info

    # Common grid: width from the config, height from the aspect ratio of the first composite panel
    ref = panels[next(f for f, s in folder_specs.items() if s["kind"] == "composite")]
    target_height = int(round(target_width * ref.shape[0] / ref.shape[1]))
    size_wh = (target_width, target_height)

    layers, names, masks = [], [], []
    for folder, spec in folder_specs.items():
        panel = panels[folder]
        if spec["kind"] == "colormap":
            vals, dstats = decode_colormap(panel, spec["cmap"], spec["vmin"], spec["vmax"])
            inspection[folder]["colormap_decode"] = dstats
            valid = np.isfinite(vals)
            inspection[folder]["value_range_decoded"] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
            out, ok = resize_with_mask(vals, valid, size_wh)
            layers.append(out)
            names.append(spec["channel"])
            masks.append(ok)
        else:
            # Display composites carry no explicit no-data marker after rendering (no-data was drawn
            # white, but bright clipped clouds are white too), so their validity comes from the
            # colour-mapped products of the same date, which mark no-data unambiguously.
            for c, cname in enumerate(spec["channels"]):
                layers.append(resize_float(panel[..., c].astype(np.float32), size_wh))
                names.append(f"{folder}:{cname}")

    valid = np.logical_and.reduce(masks) if masks else np.ones((target_height, target_width), bool)
    return {"channels": np.stack(layers, axis=-1), "names": names, "valid": valid,
            "inspection": inspection, "grid_hw": (target_height, target_width), "date": sample["date"]}


def alignment_check(sample_data, max_shift=4):
    """
    Check that the resampled panels line up on the ground.

    1. RGB:R and False_Color:G are both band B04 with the same stretch, and RGB:G and False_Color:B
       are both B03 - so they should be (almost) identical if the grids align.
    2. NDVI vs the pseudo-NDVI built from False_Color (B08 - B04)/(B08 + B04) checks the
       colour-map figure against the composite figure (different figure layout).
    For each pair we search integer shifts in [-max_shift, max_shift] and report the shift with
    the highest correlation. Best shift (0, 0) means the grids are aligned to within a pixel.
    """
    x, names, valid = sample_data["channels"], sample_data["names"], sample_data["valid"]
    idx = {n: i for i, n in enumerate(names)}

    def best_shift(a, b):
        m = max_shift
        a_c = a[m:-m, m:-m]
        v_c = valid[m:-m, m:-m]
        best = (-2.0, (0, 0))
        zero = None
        for dy in range(-m, m + 1):
            for dx in range(-m, m + 1):
                b_c = b[m + dy:b.shape[0] - m + dy, m + dx:b.shape[1] - m + dx]
                r = np.corrcoef(a_c[v_c], b_c[v_c])[0, 1]
                if (dy, dx) == (0, 0):
                    zero = r
                if r > best[0]:
                    best = (r, (dy, dx))
        return {"best_shift_dy_dx": best[1], "corr_at_best": float(best[0]), "corr_at_zero": float(zero)}

    report = {}
    pairs = [("RGB:R(B04)", "False_Color:G(B04)"), ("RGB:G(B03)", "False_Color:B(B03)")]
    for a, b in pairs:
        if a in idx and b in idx:
            report[f"{a} vs {b}"] = best_shift(x[..., idx[a]], x[..., idx[b]])
            diff = np.abs(x[..., idx[a]] - x[..., idx[b]])[valid]
            report[f"{a} vs {b}"]["mean_abs_diff_levels"] = float(diff.mean())
    if {"False_Color:R(B08)", "False_Color:G(B04)", "NDVI"} <= idx.keys():
        nir, red = x[..., idx["False_Color:R(B08)"]], x[..., idx["False_Color:G(B04)"]]
        pseudo = (nir - red) / (nir + red + 1e-6)
        report["NDVI vs pseudo-NDVI(False_Color)"] = best_shift(x[..., idx["NDVI"]], pseudo)
    return report


# ------------------------------------------------------------- raw Sentinel-2 bands
def find_raw_scenes(raw_dir, pattern="raichur_s2_*.nc"):
    """Raw single-date NetCDF files (multi-date 'range' files are skipped), sorted by full name."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    return sorted((p for p in raw_dir.glob(pattern) if "_range_" not in p.name), key=lambda p: p.name)


def load_raw_scene(path, bands, scale=10000.0, downsample=1):
    """
    Load the ORIGINAL Sentinel-2 L1C bands from one NetCDF file written by
    scripts/download_s2_many_days.py. Values are top-of-atmosphere reflectance x 10000, so we divide
    by `scale`. A pixel is valid only if every band is present and > 0 (same rule as scripts/main.py).
    `downsample` > 1 averages k x k blocks to save memory/time (valid only if the whole block is valid).
    """
    import xarray as xr
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        missing = [b for b in bands if b not in ds]
        if missing:
            raise KeyError(f"{Path(path).name} has no band(s) {missing}; available: {list(ds.data_vars)}")
        arrays, valids = [], []
        for b in bands:
            a = ds[b].values
            fill = ds[b].attrs.get("_FillValue", None)
            v = a > 0 if fill is None else (a != fill) & (a > 0)
            arrays.append(a.astype(np.float32) / scale)
            valids.append(v)
        date = str(ds["t"].values)[:10] if "t" in ds else None
    x = np.stack(arrays, axis=-1)
    valid = np.logical_and.reduce(valids)
    if downsample > 1:
        k = downsample
        h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
        x = x[:h, :w].reshape(h // k, k, w // k, k, -1)
        vb = valid[:h, :w].reshape(h // k, k, w // k, k)
        x = (x * vb[..., None]).sum((1, 3)) / np.maximum(vb.sum((1, 3)), 1)[..., None]
        valid = vb.all((1, 3))
    x[~valid] = 0
    info = {"file": Path(path).name, "shape": x.shape, "dtype": str(x.dtype),
            "min": [float(x[..., i][valid].min()) if valid.any() else None for i in range(x.shape[-1])],
            "max": [float(x[..., i][valid].max()) if valid.any() else None for i in range(x.shape[-1])],
            "valid_pct": float(100 * valid.mean())}
    return {"channels": x.astype(np.float32), "names": list(bands), "valid": valid,
            "inspection": {"raw": info}, "grid_hw": x.shape[:2], "date": date}


# ------------------------------------------------------------ experiment-level helpers
def discover_samples(experiment, cfg, logger=None):
    """
    List the samples (acquisition dates) of an experiment in a fixed, reproducible order.
    cfg is the algorithm's config module. Returns the index mapping dict.
    """
    if experiment == "png":
        return build_index_mapping(cfg.PNG_DATASET_DIR, list(cfg.PNG_FOLDERS),
                                   cfg.EXPECTED_IMAGES_PER_FOLDER, logger)
    if experiment == "raw":
        files = find_raw_scenes(cfg.RAW_DATASET_DIR)
        samples = []
        for i, p in enumerate(files):
            m = DATE_PATTERN.search(p.name)
            samples.append({"index": i, "date": m.group(0) if m else None, "files": {"raw": p.name}})
        return {"dataset_dir": str(cfg.RAW_DATASET_DIR), "sort_rule": "ascending lexicographic order of full filename",
                "folders": {"raw": [{"index": s["index"], "filename": s["files"]["raw"], "date": s["date"]}
                                    for s in samples]},
                "samples": samples}
    raise ValueError(f"Unknown experiment '{experiment}' (use 'png' or 'raw')")


def load_normalized_sample(experiment, sample, cfg):
    """Load one sample and scale every channel to [0, 1]. Returns the sample dict with
    'channels' replaced by the normalised array and 'normalization' added."""
    from Kartavya.common.preprocessing import normalize_channels
    if experiment == "png":
        data = load_png_sample(cfg.PNG_DATASET_DIR, sample, cfg.PNG_FOLDERS, cfg.TARGET_WIDTH)
        ranges = cfg.PNG_NORMALIZATION
    else:
        data = load_raw_scene(Path(cfg.RAW_DATASET_DIR) / sample["files"]["raw"], cfg.RAW_BANDS,
                              cfg.RAW_SCALE, cfg.RAW_DOWNSAMPLE)
        ranges = cfg.RAW_NORMALIZATION
    data["channels_natural_range"] = [(float(data["channels"][..., c][data["valid"]].min()),
                                       float(data["channels"][..., c][data["valid"]].max()))
                                      if data["valid"].any() else (None, None)
                                      for c in range(data["channels"].shape[-1])]
    data["channels"], data["normalization"] = normalize_channels(data["channels"], data["names"], ranges)
    return data


def channel_groups(names):
    """{"RGB": [0, 1, 2], "False_Color": [3, 4, 5], "NDVI": [6], ...} from channel names."""
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(n.split(":")[0], []).append(i)
    return groups


def make_split(n_samples, cfg):
    """
    Decide which samples train and which test.
    Returns {"mode": "temporal"|"spatial", "train": [...], "test": [...]} where each test entry is
    {"index": i, "rows": "all" | "bottom"} and each train entry {"index": i, "rows": "all" | "top"}.
    """
    if n_samples == 0:
        raise FileNotFoundError("No samples found - check the dataset folder in config.py")
    if cfg.TRAIN_INDICES is not None and cfg.TEST_INDICES is not None:
        overlap = set(cfg.TRAIN_INDICES) & set(cfg.TEST_INDICES)
        if overlap:
            raise ValueError(f"Indices {overlap} are in both TRAIN_INDICES and TEST_INDICES")
        return {"mode": "temporal",
                "train": [{"index": i, "rows": "all"} for i in cfg.TRAIN_INDICES],
                "test": [{"index": i, "rows": "all"} for i in cfg.TEST_INDICES]}
    if n_samples == 1:
        return {"mode": "spatial", "spatial_train_fraction": cfg.SPATIAL_TRAIN_FRACTION,
                "train": [{"index": 0, "rows": "top"}], "test": [{"index": 0, "rows": "bottom"}]}
    n_train = min(max(1, int(np.ceil(n_samples * cfg.TRAIN_FRACTION))), n_samples - 1)
    return {"mode": "temporal",
            "train": [{"index": i, "rows": "all"} for i in range(n_train)],
            "test": [{"index": i, "rows": "all"} for i in range(n_train, n_samples)]}


def rows_slice(height, rows, fraction):
    """Row range for 'all', 'top' (training part) or 'bottom' (test part) of a spatial split."""
    cut = int(round(height * fraction))
    return {"all": slice(0, height), "top": slice(0, cut), "bottom": slice(cut, height)}[rows]


def load_split_pixels(experiment, mapping, entries, cfg, fraction, logger=None, with_alignment=True):
    """
    Load the samples listed in `entries` (train or test part of a split), normalised to [0, 1], and
    return the stacked feature matrix of their valid pixels plus an inspection report per date.
    Only the rows belonging to each entry ('all' / 'top' / 'bottom') are used.
    """
    from Kartavya.common.preprocessing import image_to_matrix
    blocks, inspection, names = [], {}, None
    for entry in entries:
        s = mapping["samples"][entry["index"]]
        data = load_normalized_sample(experiment, s, cfg)
        names = data["names"]
        align = alignment_check(data) if (experiment == "png" and with_alignment) else {}
        rs = rows_slice(data["valid"].shape[0], entry["rows"], fraction)
        valid = data["valid"].copy()
        valid[:rs.start] = False
        valid[rs.stop:] = False
        blocks.append(image_to_matrix(data["channels"], valid))
        inspection[s["date"]] = {"files": data["inspection"], "common_grid_hw": data["grid_hw"],
                                 "valid_pct_on_grid": float(100 * data["valid"].mean()),
                                 "channel_natural_range": dict(zip(names, data["channels_natural_range"])),
                                 "normalization": data["normalization"], "alignment": align,
                                 "rows_used": (rs.start, rs.stop), "pixels_used": int(blocks[-1].shape[0])}
        if logger:
            logger.info(f"  loaded {s['date']}: grid {data['grid_hw']}, valid {100 * data['valid'].mean():.3f}%, "
                        f"pixels used {blocks[-1].shape[0]:,} (rows {rs.start}-{rs.stop})")
            for f, info in data["inspection"].items():
                extra = ""
                if "colormap_decode" in info:
                    d = info["colormap_decode"]
                    extra = (f", decoded range {info['value_range_decoded']}, exact colour match "
                             f"{d['exact_match_pct']:.1f}%, no-data {d['white_nodata_pct']:.3f}%")
                logger.info(f"    {f}: {info.get('mode', '')} {info.get('shape')} {info.get('dtype')} "
                            f"min {info.get('min')} max {info.get('max')}"
                            + (f", map panel {info['map_panel_hw']}" if "map_panel_hw" in info else "") + extra)
            for pair, r in align.items():
                logger.info(f"    alignment {pair}: best shift {r['best_shift_dy_dx']}, r={r['corr_at_best']:.4f}")
    return np.concatenate(blocks), names, inspection
