"""
Dataset-independent loading for every Kartavya algorithm.

The ONLY required input is a path. It may be

  * one image file                    -> one sample (H x W x C)
  * a folder of image files           -> one sample per file (same number of channels each; spatial sizes may
                                         differ), or - if every file is single-band with the same shape - the
                                         files are stacked as the bands of ONE image (e.g. B02.tif, B03.tif ...)
  * a folder of sub-folders           -> "product" layout (the layout of the original repository:
                                         RGB/, False_Color/, NDVI/ ... with the same number of images each):
                                         the i-th image of every sub-folder (sorted filename order) is stacked
                                         into sample i; products of different size are resampled to one grid.

Supported formats (detected from the file extension and, for ambiguous cases, from the content):
  PNG, JPG/JPEG, BMP, GIF, WebP            (Pillow; 8/16-bit, palette, alpha -> validity mask)
  TIFF / GeoTIFF, JPEG2000, VRT, ERDAS .img (rasterio if installed - keeps georeferencing + nodata;
                                           tifffile/Pillow fallback for plain TIFF)
  ENVI (.hdr + data file)                  (spectral if installed, else rasterio)
  NumPy .npy / .npz                        (2-D or 3-D arrays, any dtype)
  MATLAB .mat (v5 and v7.3/HDF5)           (scipy / h5py; e.g. Indian Pines, Pavia, Salinas cubes)
  NetCDF .nc                               (xarray; every 2-D/3-D data variable becomes channel(s))
  HDF5 .h5 / .hdf5 / .he5                  (h5py; the largest numeric 2-D/3-D dataset)

What is detected automatically: image size, number of channels/bands, channel order (H,W,C vs C,H,W),
data type, value range, band names (where the format stores them), no-data pixels (NaN, the format's nodata
value, a fully transparent alpha channel). Data are kept in their native dtype until normalisation, so
16-bit and floating-point scientific data lose no precision.

Label rasters for the supervised algorithm (LDA) are recognised by their file name (see config.LABEL_KEYWORDS)
and are never used as image data.
"""
import re
from pathlib import Path

import numpy as np

from common import config as gcfg
from common.preprocessing import Normalizer, image_to_matrix
from common.utils import fingerprint, safe_name

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None        # large scientific rasters are legitimate; switch off the bomb guard
except ImportError:                      # pragma: no cover
    Image = None

PIL_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".ppm", ".pgm"}
TIFF_EXT = {".tif", ".tiff", ".gtiff"}
RASTERIO_EXT = {".jp2", ".vrt", ".img", ".j2k"}
ENVI_HDR = ".hdr"
ENVI_DATA_EXT = {"", ".img", ".dat", ".raw", ".bsq", ".bil", ".bip", ".bin", ".envi"}
NUMPY_EXT = {".npy", ".npz"}
MAT_EXT = {".mat"}
NETCDF_EXT = {".nc", ".nc4", ".cdf"}
HDF5_EXT = {".h5", ".hdf5", ".he5", ".hdf"}
SUPPORTED_EXT = PIL_EXT | TIFF_EXT | RASTERIO_EXT | {ENVI_HDR} | NUMPY_EXT | MAT_EXT | NETCDF_EXT | HDF5_EXT


class DatasetError(ValueError):
    """Raised with a human-readable explanation when a dataset cannot be used."""


# ============================================================================ helpers
def _is_label_file(path, keywords=None):
    words = [w.lower() for w in (keywords or gcfg.LABEL_KEYWORDS)]
    stem = Path(path).stem.lower()
    tokens = re.split(r"[^a-z0-9]+", stem)
    return any(w in tokens for w in words) or any(len(w) > 3 and stem.endswith(w) for w in words)


def to_hwc(arr, hint=None):
    """
    Bring an array into (H, W, C) order.
      2-D             -> (H, W, 1)
      hint == "chw"   -> transposed (formats that are known to be band-first, e.g. rasterio)
      otherwise       -> the SMALLEST axis is taken as the channel axis (an image is never thinner than it
                         has bands in practice); ties keep the (H, W, C) interpretation.
    Returns (array, description of the decision).
    """
    a = np.asarray(arr)
    a = np.squeeze(a) if a.ndim > 3 else a
    if a.ndim == 2:
        return a[..., None], "2-D array -> 1 channel"
    if a.ndim != 3:
        raise DatasetError(f"Expected a 2-D or 3-D array, got shape {a.shape}")
    if hint == "chw":
        return np.moveaxis(a, 0, -1), "band-first format -> (H, W, C)"
    s = a.shape
    if s[2] <= s[0] and s[2] <= s[1]:
        return a, f"shape {s}: last axis smallest -> (H, W, C)"
    if s[0] == s[1]:
        return a, f"shape {s}: first two axes tied -> keep (H, W, C)"
    if s[0] < s[1] and s[0] < s[2]:
        return np.moveaxis(a, 0, -1), f"shape {s}: first axis smallest -> (C, H, W) transposed"
    return np.moveaxis(a, 1, -1), f"shape {s}: middle axis smallest -> (H, C, W) transposed"


def _numeric(a):
    return isinstance(a, np.ndarray) and a.dtype.kind in "biuf" and a.ndim >= 2 and a.size > 1


def _pick_array(candidates, want_labels=False):
    """From {name: array} pick the image cube (largest 3-D, else largest 2-D) or the label map."""
    items = [(k, v) for k, v in candidates.items() if _numeric(v)]
    if not items:
        return None, None
    if want_labels:
        two_d = [(k, v) for k, v in items if np.squeeze(v).ndim == 2]
        ints = [(k, v) for k, v in two_d if v.dtype.kind in "biu" or np.all(np.mod(v, 1) == 0)]
        pool = ints or two_d
        if not pool:
            return None, None
        return max(pool, key=lambda kv: kv[1].size)
    three = [(k, v) for k, v in items if np.squeeze(v).ndim == 3]
    pool = three or items
    return max(pool, key=lambda kv: kv[1].size)


# ============================================================================ per-format readers
def _read_pil(path):
    with Image.open(path) as im:
        mode, n_frames = im.mode, getattr(im, "n_frames", 1)
        info = {"reader": "Pillow", "mode": mode, "size_wh": im.size, "frames": n_frames}
        if mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
            mode = im.mode
        elif mode in ("1",):
            im = im.convert("L")
            mode = "L"
        arr = np.asarray(im)
    names = list(mode) if mode in ("RGB", "RGBA", "CMYK", "LA", "YCbCr", "HSV", "LAB") else None
    mask = None
    a, how = to_hwc(arr)
    if mode in ("RGBA", "LA"):
        alpha = a[..., -1]
        a = a[..., :-1]
        names = names[:-1] if names else None
        if alpha.min() < alpha.max() or alpha.max() == 0:
            mask = alpha > 0                         # fully transparent pixels carry no data
            info["alpha_used_as_validity_mask"] = True
        else:
            info["alpha_dropped"] = "constant alpha channel carries no information"
    info["orientation"] = how
    return a, names, mask, info


def _read_rasterio(path):
    import rasterio
    with rasterio.open(path) as ds:
        arr = ds.read()                                # (bands, H, W), native dtype
        nodata = ds.nodata
        names = [d if d else None for d in ds.descriptions]
        msk = None
        try:
            m = ds.dataset_mask()                      # 0 = no data (nodata value, alpha, internal mask)
            if m.min() == 0:
                msk = m > 0
        except Exception:                              # pragma: no cover
            pass
        info = {"reader": "rasterio", "driver": ds.driver, "crs": str(ds.crs) if ds.crs else None,
                "transform": list(ds.transform)[:6] if ds.transform else None, "nodata": nodata,
                "count": ds.count, "dtypes": list(ds.dtypes)}
    a = np.moveaxis(arr, 0, -1)
    if nodata is not None and msk is None:
        msk = ~np.any(np.isclose(a, nodata) if a.dtype.kind == "f" else a == nodata, axis=-1)
    names = names if any(names) else None
    info["orientation"] = "band-first (rasterio) -> (H, W, C)"
    return a, names, msk, info


def _read_tiff(path):
    try:
        import rasterio  # noqa: F401
        return _read_rasterio(path)
    except ImportError:
        pass
    except Exception as e:                             # rasterio could not open it: try tifffile
        first_error = e
    else:
        first_error = None
    try:
        import tifffile
        arr = tifffile.imread(path)
        a, how = to_hwc(arr)
        return a, None, None, {"reader": "tifffile", "orientation": how,
                               "rasterio_error": str(first_error) if first_error else None}
    except ImportError:
        return _read_pil(path)


def _read_envi(path):
    p = Path(path)
    hdr = p if p.suffix.lower() == ".hdr" else p.with_suffix(".hdr")
    try:
        import spectral.io.envi as envi
        img = envi.open(str(hdr))
        arr = np.asarray(img.load())                   # (rows, cols, bands) in native dtype
        meta = img.metadata
        names = meta.get("band names")
        if not names and meta.get("wavelength"):
            names = [f"{w}{meta.get('wavelength units', '')}".strip() for w in meta["wavelength"]]
        mask = None
        ignore = meta.get("data ignore value")
        if ignore is not None:
            mask = ~np.any(arr == float(ignore), axis=-1)
        return arr, names, mask, {"reader": "spectral (ENVI)", "interleave": meta.get("interleave"),
                                  "orientation": "ENVI (rows, cols, bands)"}
    except ImportError:
        data = _envi_data_file(hdr)
        return _read_rasterio(data if data else hdr)


def _envi_data_file(hdr):
    for ext in sorted(ENVI_DATA_EXT):
        c = hdr.with_suffix(ext) if ext else hdr.with_suffix("")
        if c.exists() and c != hdr:
            return c
    return None


def _read_numpy(path, want_labels=False):
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=False)
        key = None
    else:
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        preferred = [k for k in ("data", "image", "img", "cube", "x", "X", "arr_0") if k in d and _numeric(d[k])]
        key, arr = (preferred[0], d[preferred[0]]) if preferred and not want_labels else _pick_array(d, want_labels)
        if arr is None:
            raise DatasetError(f"{p.name}: no numeric 2-D/3-D array inside (keys {list(d)})")
    a, how = to_hwc(arr)
    return a, None, None, {"reader": "numpy", "key": key, "orientation": how}


def _read_mat(path, want_labels=False):
    try:
        from scipy.io import loadmat
        d = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        reader, transposed = "scipy.io.loadmat", False
    except NotImplementedError:                        # MATLAB v7.3 = HDF5
        import h5py
        d = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                    d[name] = np.asarray(obj).T        # MATLAB stores column-major: reverse the axes
            f.visititems(visit)
        reader, transposed = "h5py (MATLAB v7.3)", True
    key, arr = _pick_array(d, want_labels)
    if arr is None:
        raise DatasetError(f"{Path(path).name}: no numeric 2-D/3-D variable found (variables {list(d)})")
    a, how = to_hwc(arr)
    return a, None, None, {"reader": reader, "variable": key, "orientation": how,
                           "all_variables": {k: list(np.shape(v)) for k, v in d.items()}, "v73": transposed}


def _read_netcdf(path):
    import xarray as xr
    with xr.open_dataset(path, mask_and_scale=True) as ds:
        layers, names, shape = [], [], None
        for v in ds.data_vars:
            a = np.squeeze(np.asarray(ds[v].values))
            if a.ndim == 2:
                parts = [(v, a)]
            elif a.ndim == 3:                          # (band/time, y, x)
                parts = [(f"{v}[{i}]", a[i]) for i in range(a.shape[0])]
            else:
                continue
            for n, layer in parts:
                if shape is None:
                    shape = layer.shape
                if layer.shape != shape or layer.dtype.kind not in "biuf":
                    continue
                layers.append(layer)
                names.append(n)
    if not layers:
        raise DatasetError(f"{Path(path).name}: no 2-D/3-D numeric data variables")
    a = np.stack(layers, axis=-1)
    return a, names, None, {"reader": "xarray (NetCDF)", "variables": names, "orientation": "(y, x) per variable"}


def _read_hdf5(path, want_labels=False):
    import h5py
    d = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 and obj.dtype.kind in "biuf":
                d[name] = np.asarray(obj)
        f.visititems(visit)
    key, arr = _pick_array(d, want_labels)
    if arr is None:
        raise DatasetError(f"{Path(path).name}: no numeric 2-D/3-D dataset")
    a, how = to_hwc(arr)
    return a, None, None, {"reader": "h5py", "dataset": key, "orientation": how}


def read_raster(path, want_labels=False):
    """
    Read ONE file in its native dtype. Returns
      data  : ndarray (H, W, C)
      names : list of band names or None
      mask  : bool (H, W) validity from the format (alpha, nodata) or None
      info  : dict describing how the file was read
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in PIL_EXT:
        out = _read_pil(p)
    elif ext in TIFF_EXT:
        out = _read_tiff(p)
    elif ext == ENVI_HDR:
        out = _read_envi(p)
    elif ext in RASTERIO_EXT:
        out = _read_envi(p) if p.with_suffix(".hdr").exists() else _read_rasterio(p)
    elif ext in NUMPY_EXT:
        out = _read_numpy(p, want_labels)
    elif ext in MAT_EXT:
        out = _read_mat(p, want_labels)
    elif ext in NETCDF_EXT:
        out = _read_netcdf(p)
    elif ext in HDF5_EXT:
        out = _read_hdf5(p, want_labels)
    elif p.with_suffix(".hdr").exists():
        out = _read_envi(p)
    else:
        raise DatasetError(f"Unsupported file type '{ext}' ({p.name}). Supported: {sorted(SUPPORTED_EXT)}")
    data, names, mask, info = out
    if data.dtype.kind not in "biuf":
        raise DatasetError(f"{p.name}: unsupported data type {data.dtype}")
    if data.dtype == bool:
        data = data.astype(np.uint8)
    info.update({"file": p.name, "shape_hwc": list(data.shape), "dtype": str(data.dtype)})
    return data, names, mask, info


# ============================================================================ resampling
def _resize_stack(a, valid, size_hw):
    """Area-average resampling of every channel (and of the validity mask) to size_hw. A target pixel is
    valid only if >= 99 % of its source area was valid (no-data never leaks into valid pixels)."""
    from skimage.transform import resize
    h, w = size_hw
    if a.shape[:2] == (h, w):
        return a, valid
    af = a.astype(np.float64)
    vf = valid.astype(np.float64)
    filled = np.where(valid[..., None], af, 0.0)
    shrink = h < a.shape[0] or w < a.shape[1]
    kw = dict(order=1, mode="edge", anti_aliasing=shrink, preserve_range=True)
    num = resize(filled, (h, w, a.shape[2]), **kw)
    cov = resize(vf, (h, w), **kw)
    out = np.where(cov[..., None] > 0, num / np.maximum(cov[..., None], 1e-9), 0.0)
    if a.dtype.kind in "iu":
        info = np.iinfo(a.dtype)
        out = np.clip(np.round(out), info.min, info.max)
        out = out.astype(a.dtype)
    else:
        out = out.astype(a.dtype)
    return out, cov >= 0.99


def resize_labels(lab, size_hw, max_aspect_error=0.02, name="labels"):
    """Nearest-neighbour resampling for class ids (they must never be averaged); aspect ratio is checked."""
    h, w = size_hw
    if lab.shape == (h, w):
        return lab
    err = abs(lab.shape[0] / lab.shape[1] - h / w) / (h / w)
    if err > max_aspect_error:
        raise DatasetError(f"{name}: shape {lab.shape} does not match the image {size_hw} "
                           f"({100 * err:.1f}% aspect difference) - it probably covers a different area")
    rows = np.minimum((np.arange(h) * lab.shape[0] / h).astype(int), lab.shape[0] - 1)
    cols = np.minimum((np.arange(w) * lab.shape[1] / w).astype(int), lab.shape[1] - 1)
    return lab[rows][:, cols]


# ============================================================================ the dataset
class Dataset:
    """
    A dataset = an ordered list of samples (images) with the same channels, plus optional label rasters.

    Attributes
      name, dataset_id      : folder-safe name and name + content fingerprint (used for output folders)
      layout                : "single_file" | "files" | "band_stack" | "products"
      samples               : [{"index", "name", "files"}] in sorted-filename order
      channel_names, n_channels, dtype
    """

    def __init__(self, path, label_path=None, logger=None, max_image_side=None):
        self.path = Path(path).expanduser().resolve()
        self.logger = logger
        self.max_image_side = gcfg.MAX_IMAGE_SIDE if max_image_side is None else max_image_side
        if not self.path.exists():
            raise DatasetError(f"Dataset path does not exist: {self.path}")
        self._cache, self._cache_bytes = {}, 0
        self.label_files = []
        self._discover()
        self._discover_labels(label_path)
        self.name = safe_name(self.path.stem if self.path.is_file() else self.path.name)
        self.dataset_id = f"{self.name}_{fingerprint(self.all_files)}"
        self._probe()
        self.normalizer = None

    # ------------------------------------------------------------------ discovery
    def _log(self, msg, level="info"):
        if self.logger:
            getattr(self.logger, level)(msg)

    def _candidates(self, folder, recursive=False):
        it = folder.rglob("*") if recursive else folder.iterdir()
        files = sorted((p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
                        and not p.name.startswith(".")), key=lambda p: str(p.relative_to(folder)))
        # ENVI: the .hdr represents the pair; a data file with a sibling .hdr is not a separate image
        files = [p for p in files if p.suffix.lower() == ".hdr" or not p.with_suffix(".hdr").exists()]
        images = [p for p in files if not _is_label_file(p)]
        labels = [p for p in files if _is_label_file(p)]
        return images, labels

    def _discover(self):
        p = self.path
        if p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXT and not p.with_suffix(".hdr").exists():
                raise DatasetError(f"Unsupported file type '{p.suffix}'. Supported: {sorted(SUPPORTED_EXT)}")
            self.layout = "single_file"
            self.samples = [{"index": 0, "name": p.stem, "files": [p]}]
            images, labels = self._candidates(p.parent)
            self.label_files = [lab for lab in labels if lab != p]
            self.all_files = [p]
            return
        images, labels = self._candidates(p)
        self.label_files = labels
        if images:
            self.layout = "files"
            self.samples = [{"index": i, "name": f.stem, "files": [f]} for i, f in enumerate(images)]
            self.all_files = list(images)
            return
        subdirs = sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))
        per_dir = {d: self._candidates(d)[0] for d in subdirs}
        per_dir = {d: f for d, f in per_dir.items() if f}
        for d in subdirs:
            self.label_files += self._candidates(d)[1]
        if len(per_dir) >= 2 and len({len(f) for f in per_dir.values()}) == 1:
            self.layout = "products"
            n = len(next(iter(per_dir.values())))
            self.products = [d.name for d in per_dir]
            self.samples = [{"index": i, "name": _common_tag([per_dir[d][i] for d in per_dir]) or f"sample_{i:03d}",
                             "files": [per_dir[d][i] for d in per_dir]} for i in range(n)]
            self.all_files = [f for fs in per_dir.values() for f in fs]
            return
        images, labels = self._candidates(p, recursive=True)
        self.label_files = labels
        if not images:
            raise DatasetError(f"No supported image files in {p} (supported extensions: {sorted(SUPPORTED_EXT)})")
        if per_dir and len({len(f) for f in per_dir.values()}) > 1:
            self._log(f"Sub-folders have different image counts {[len(f) for f in per_dir.values()]}; "
                      "treating every file as its own sample", "warning")
        self.layout = "files"
        self.samples = [{"index": i, "name": f.stem, "files": [f]} for i, f in enumerate(images)]
        self.all_files = list(images)

    def _discover_labels(self, label_path):
        if label_path:
            lp = Path(label_path).expanduser().resolve()
            if not lp.exists():
                raise DatasetError(f"Label path does not exist: {lp}")
            self.label_files = [lp] if lp.is_file() else sorted(
                q for q in lp.rglob("*") if q.is_file() and q.suffix.lower() in SUPPORTED_EXT)
        self.label_map = {}
        if not self.label_files:
            return
        if len(self.samples) == 1 and len(self.label_files) == 1:
            self.label_map[0] = self.label_files[0]
            return
        for s in self.samples:
            stem = s["name"].lower()
            match = [lab for lab in self.label_files if stem and (stem in lab.stem.lower() or _strip_label(lab.stem) in stem)]
            if match:
                self.label_map[s["index"]] = min(match, key=lambda q: len(q.stem))
            elif len(self.label_files) == 1 and len(self.samples) == 1:
                self.label_map[s["index"]] = self.label_files[0]

    # ------------------------------------------------------------------ probing
    def _probe(self):
        if self.layout == "files" and len(self.samples) > 1:
            # a folder of single-band rasters of identical size = the bands of ONE image (B02.tif, B03.tif ...)
            shapes = []
            for s in self.samples:
                d, _, _, _ = read_raster(s["files"][0])
                shapes.append(d.shape)
                if d.shape[2] != 1:
                    break
            if len(shapes) == len(self.samples) and all(sh[2] == 1 for sh in shapes) and len({sh[:2] for sh in shapes}) == 1:
                self._log(f"{len(shapes)} single-band files of identical size {shapes[0][:2]}: stacked as the bands "
                          "of one image")
                self.layout = "band_stack"
                self.samples = [{"index": 0, "name": self.path.name, "files": [s["files"][0] for s in self.samples]}]
                if self.label_files and not self.label_map:
                    self.label_map = {0: self.label_files[0]} if len(self.label_files) == 1 else {}
        self.inspection = {"path": str(self.path), "layout": self.layout, "n_samples": len(self.samples),
                           "label_files": [str(q) for q in self.label_files], "samples": []}
        counts, names0, dtypes = {}, None, set()
        for s in self.samples:
            data, valid, names, info = self._read_sample(s)
            counts[s["name"]] = data.shape[2]
            dtypes.add(str(data.dtype))
            if names0 is None:
                names0 = names
            s["shape_hw"] = list(data.shape[:2])
            self.inspection["samples"].append(
                {"index": s["index"], "name": s["name"], "files": [str(f) for f in s["files"]],
                 "shape_hwc": list(data.shape), "dtype": str(data.dtype),
                 "valid_pct": float(100 * valid.mean()),
                 "value_range_per_channel": _ranges(data, valid), "readers": info})
            self._remember(s["index"], (data, valid))
        if len(set(counts.values())) > 1:
            raise DatasetError("Samples have different numbers of channels, so one model cannot be applied to all "
                               f"of them: {counts}. Put images with the same bands in separate folders.")
        self.n_channels = next(iter(counts.values()))
        self.channel_names = _unique_names(names0, self.n_channels)
        self.dtype = np.result_type(*[np.dtype(d) for d in dtypes])
        if self.n_channels < 2:
            raise DatasetError(f"The data have {self.n_channels} channel(s). Channel/band reduction needs at least "
                               "2 channels per pixel.")
        self.inspection.update({"n_channels": self.n_channels, "channel_names": self.channel_names,
                                "dtype": str(self.dtype)})

    def _remember(self, idx, value):
        size = value[0].nbytes + value[1].nbytes
        if self._cache_bytes + size <= gcfg.CACHE_LIMIT_BYTES:
            self._cache[idx] = value
            self._cache_bytes += size

    def _read_sample(self, s):
        infos = []
        if self.layout in ("single_file", "files"):
            data, names, mask, info = read_raster(s["files"][0])
            infos.append(info)
            valid = np.ones(data.shape[:2], bool) if mask is None else mask.copy()
        else:
            layers, names, masks = [], [], []
            for f in s["files"]:
                d, n, m, info = read_raster(f)
                infos.append(info)
                prefix = f.parent.name if self.layout == "products" else f.stem
                layers.append((d, np.ones(d.shape[:2], bool) if m is None else m))
                names += [f"{prefix}:{x}" for x in (n or [f"b{i + 1}" for i in range(d.shape[2])])] \
                    if d.shape[2] > 1 or self.layout == "products" else [prefix]
            shapes = {lay[0].shape[:2] for lay in layers}
            target = max(shapes, key=lambda hw: hw[0] * hw[1])
            if len(shapes) > 1:
                self._log(f"{s['name']}: files have different sizes {sorted(shapes)}; resampled to {target}", "warning")
            dt = np.result_type(*[lay[0].dtype for lay in layers])
            parts = []
            for d, m in layers:
                d2, m2 = _resize_stack(d, m, target)
                parts.append(d2.astype(dt))
                masks.append(m2)
            data = np.concatenate(parts, axis=-1)
            valid = np.logical_and.reduce(masks)
        if data.dtype.kind == "f":
            valid &= np.all(np.isfinite(data), axis=-1)
        if self.max_image_side and max(data.shape[:2]) > self.max_image_side:
            f = self.max_image_side / max(data.shape[:2])
            new = (max(1, int(round(data.shape[0] * f))), max(1, int(round(data.shape[1] * f))))
            self._log(f"{s['name']}: {data.shape[:2]} downsampled to {new} (MAX_IMAGE_SIDE)")
            data, valid = _resize_stack(data, valid, new)
        return data, valid, names, infos

    # ------------------------------------------------------------------ public API
    @property
    def n_samples(self):
        return len(self.samples)

    def load_native(self, index):
        """(data in native dtype (H, W, C), valid mask (H, W))."""
        if index in self._cache:
            return self._cache[index]
        data, valid, _, _ = self._read_sample(self.samples[index])
        self._remember(index, (data, valid))
        return data, valid

    def fit_normalizer(self, split, logger=None):
        """Normalisation statistics from the TRAINING region only (shared by all algorithms)."""
        blocks = []
        for entry in split["train"]:
            data, valid = self.load_native(entry["index"])
            v = restrict_rows(valid, entry["rows"], split.get("spatial_train_fraction", gcfg.SPATIAL_TRAIN_FRACTION))
            blocks.append(data[v])
        x = np.concatenate(blocks) if blocks else np.zeros((0, self.n_channels))
        if len(x) == 0:
            raise DatasetError("The training region contains no valid pixels.")
        self.normalizer = Normalizer(gcfg.NORMALIZATION).fit(x, self.dtype)
        if logger:
            logger.info(f"Normalisation '{self.normalizer.method_used}': per-channel low {np.round(self.normalizer.low, 4).tolist()[:12]}"
                        f"{' ...' if self.n_channels > 12 else ''} high {np.round(self.normalizer.high, 4).tolist()[:12]}"
                        f"{' ...' if self.n_channels > 12 else ''}")
        return self.normalizer

    def load_normalized(self, index):
        """(channels scaled to [0, 1] with the TRAINING statistics, valid mask)."""
        if self.normalizer is None:
            raise RuntimeError("fit_normalizer(split) must be called first")
        data, valid = self.load_native(index)
        return self.normalizer.transform(data), valid

    def labels_for(self, index):
        """Integer class raster on the sample's grid, or None if this sample has no label file."""
        f = self.label_map.get(index)
        if f is None:
            return None
        lab, _, _, _ = read_raster(f, want_labels=True)
        lab = np.squeeze(lab)
        if lab.ndim != 2:
            raise DatasetError(f"{f.name}: a label raster must be 2-D, got {lab.shape}")
        if lab.dtype.kind == "f":
            if not np.all(np.isfinite(lab)) or np.any(np.mod(lab, 1) != 0):
                raise DatasetError(f"{f.name}: label values must be integer class ids")
        shape = tuple(self.load_native(index)[1].shape)
        return resize_labels(lab.astype(np.int64), shape, name=f.name)

    def summary(self):
        return {k: v for k, v in self.inspection.items() if k != "samples"} | {
            "dataset_id": self.dataset_id, "label_map": {int(k): str(v) for k, v in self.label_map.items()}}


def _strip_label(stem):
    s = stem.lower()
    for w in gcfg.LABEL_KEYWORDS:
        s = re.sub(rf"[_\-. ]?{w}$", "", s)
    return s


def _common_tag(files):
    m = [re.search(r"\d{4}-\d{2}-\d{2}", f.name) for f in files]
    if all(m) and len({x.group(0) for x in m}) == 1:
        return m[0].group(0)
    return files[0].stem


def _unique_names(names, c):
    if not names or len(names) != c:
        width = max(3, len(str(c)))
        return [f"band_{i + 1:0{width}d}" for i in range(c)]
    out, seen = [], {}
    for n in names:
        n = str(n) if n else "band"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        out.append(n)
    return out


def _ranges(data, valid, max_channels=32):
    if not valid.any():
        return None
    v = data[valid]
    n = min(v.shape[1], max_channels)
    return [[float(np.nanmin(v[:, i])), float(np.nanmax(v[:, i]))] for i in range(n)]


# ============================================================================ split + matrices
def make_split(n_samples):
    """
    Which samples train and which test (rule of the original repository, see config.py).
    Returns {"mode": "temporal"|"spatial", "train": [...], "test": [...]} with entries {"index", "rows"},
    rows in {"all", "top", "bottom"}.
    """
    if n_samples == 0:
        raise DatasetError("No samples found")
    if gcfg.TRAIN_INDICES is not None and gcfg.TEST_INDICES is not None:
        overlap = set(gcfg.TRAIN_INDICES) & set(gcfg.TEST_INDICES)
        if overlap:
            raise DatasetError(f"Indices {overlap} are in both TRAIN_INDICES and TEST_INDICES")
        bad = [i for i in list(gcfg.TRAIN_INDICES) + list(gcfg.TEST_INDICES) if not 0 <= i < n_samples]
        if bad:
            raise DatasetError(f"Split indices {bad} do not exist (dataset has {n_samples} samples)")
        return {"mode": "temporal", "train": [{"index": i, "rows": "all"} for i in gcfg.TRAIN_INDICES],
                "test": [{"index": i, "rows": "all"} for i in gcfg.TEST_INDICES]}
    if n_samples == 1:
        return {"mode": "spatial", "spatial_train_fraction": gcfg.SPATIAL_TRAIN_FRACTION,
                "train": [{"index": 0, "rows": "top"}], "test": [{"index": 0, "rows": "bottom"}]}
    n_train = min(max(1, int(np.ceil(n_samples * gcfg.TRAIN_FRACTION))), n_samples - 1)
    return {"mode": "temporal", "train": [{"index": i, "rows": "all"} for i in range(n_train)],
            "test": [{"index": i, "rows": "all"} for i in range(n_train, n_samples)]}


def rows_slice(height, rows, fraction):
    """Row range for 'all', 'top' (training part) or 'bottom' (test part) of a spatial split."""
    cut = int(round(height * fraction))
    return {"all": slice(0, height), "top": slice(0, cut), "bottom": slice(cut, height)}[rows]


def restrict_rows(valid, rows, fraction):
    rs = rows_slice(valid.shape[0], rows, fraction)
    v = valid.copy()
    v[:rs.start] = False
    v[rs.stop:] = False
    return v


def load_split_pixels(dataset, entries, fraction, logger=None, rng=None):
    """
    Normalised feature matrix (valid pixels x channels) of the given split entries, plus an inspection
    report per sample. At most config.TRAIN_PIXEL_POOL_PER_SAMPLE pixels are taken from each sample
    (random, seeded) to bound memory on very large rasters.
    """
    rng = rng if rng is not None else np.random.default_rng(gcfg.RANDOM_SEED)
    blocks, inspection = [], {}
    for entry in entries:
        s = dataset.samples[entry["index"]]
        img, valid = dataset.load_normalized(entry["index"])
        rs = rows_slice(valid.shape[0], entry["rows"], fraction)
        v = restrict_rows(valid, entry["rows"], fraction)
        x = image_to_matrix(img, v)
        if len(x) > gcfg.TRAIN_PIXEL_POOL_PER_SAMPLE:
            x = x[np.sort(rng.choice(len(x), gcfg.TRAIN_PIXEL_POOL_PER_SAMPLE, replace=False))]
        blocks.append(x)
        inspection[s["name"]] = {"files": [str(f) for f in s["files"]], "grid_hw": list(valid.shape),
                                 "valid_pct_on_grid": float(100 * valid.mean()),
                                 "rows_used": (rs.start, rs.stop), "pixels_used": int(len(x))}
        if logger:
            logger.info(f"  loaded {s['name']}: grid {tuple(valid.shape)}, valid {100 * valid.mean():.3f}%, "
                        f"pixels used {len(x):,} (rows {rs.start}-{rs.stop})")
    if not blocks or sum(len(b) for b in blocks) == 0:
        raise DatasetError("No valid training pixels")
    return np.concatenate(blocks), dataset.channel_names, inspection


def channel_groups(names, max_groups=8):
    """
    Groups of channels for the per-product / per-region metrics.
      names with a "product:" prefix (multi-folder layout) -> one group per product (as in the original)
      otherwise, C <= max_groups                         -> one group per channel
      otherwise                                          -> max_groups contiguous blocks of bands
                                                             (spectral regions of a hyperspectral cube)
    """
    if any(":" in n for n in names):
        groups = {}
        for i, n in enumerate(names):
            groups.setdefault(n.split(":")[0], []).append(i)
        return groups
    c = len(names)
    if c <= max_groups:
        return {n: [i] for i, n in enumerate(names)}
    edges = np.linspace(0, c, max_groups + 1).round().astype(int)
    return {f"bands {a + 1}-{b}": list(range(a, b)) for a, b in zip(edges[:-1], edges[1:]) if b > a}


def sample_tag(dataset, entry):
    s = dataset.samples[entry["index"]]
    return f"{entry['index']:02d}_{safe_name(s['name'], 40)}" + ("" if entry["rows"] == "all" else f"_{entry['rows']}")
