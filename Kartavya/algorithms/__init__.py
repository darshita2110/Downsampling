"""
Registry of the channel-reduction algorithms in this repository.

Every sub-folder of algorithms/ that contains a pipeline.py implementing the common interface
(see common/pipeline.py) is discovered automatically. REGISTERED fixes the run order of the known ones:
PCA runs first because several methods compare themselves with PCA results of the same dataset,
and the search methods run in the order of their original cross-comparisons (GA -> PSO -> ACO -> SA -> UBS-CNN).
"""
import importlib
from pathlib import Path

REGISTERED = ["pca", "ica", "lda", "ga", "pso", "aco", "sa", "ubscnn"]
INTERFACE = ("NAME", "DISPLAY", "FAMILY", "output_name", "train", "test")


def discover(only=None, logger=None):
    """Import every algorithm pipeline. Returns [(name, module or None, error or None)]."""
    here = Path(__file__).resolve().parent
    found = [d.name for d in sorted(here.iterdir()) if d.is_dir() and (d / "pipeline.py").exists()]
    order = [n for n in REGISTERED if n in found] + [n for n in found if n not in REGISTERED]
    if only:
        wanted = [o.lower() for o in only]
        unknown = [w for w in wanted if w not in order]
        if unknown:
            raise ValueError(f"Unknown algorithm(s) {unknown}; available: {order}")
        order = [n for n in order if n in wanted]
    out = []
    for name in order:
        try:
            mod = importlib.import_module(f"algorithms.{name}.pipeline")
            missing = [a for a in INTERFACE if not hasattr(mod, a)]
            if missing:
                raise AttributeError(f"pipeline.py lacks {missing}")
            out.append((name, mod, None))
        except Exception as e:                      # an import problem must not stop the other algorithms
            if logger:
                logger.error(f"Could not load algorithm '{name}': {e}")
            out.append((name, None, f"{type(e).__name__}: {e}"))
    return out
