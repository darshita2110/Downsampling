"""
Diagnose why LLE on Indian Pines gave a degenerate embedding on one machine but a normal one on another.
Prints library versions, whether the 12-nearest-neighbour graph is split into pieces, and LLE results for a few
solver / regularisation settings. Run from the project root:
    python experiments/11_diagnose_lle.py
Takes a few minutes (the dense solver is slow at 8,000 pixels). Send me the full printout.
"""
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import LocallyLinearEmbedding, trustworthiness
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hsi_loader import load_hsi
from src.preprocessing import standardize

print("numpy", np.__version__, "| scipy", scipy.__version__, "| scikit-learn", sklearn.__version__)
scene = load_hsi("indian_pines", ROOT / "data" / "hsi", labelled_only=True)
n = scene["pixels"].shape[0]
idx = np.sort(np.random.default_rng(42).permutation(n)[:8000])
X, _, _ = standardize(scene["pixels"][idx])
print("pixels:", X.shape)

G = kneighbors_graph(X, n_neighbors=12, include_self=False)
n_comp, lab = connected_components(G, directed=False)
sizes = np.sort(np.bincount(lab))[::-1]
print(f"12-NN graph: {n_comp} connected component(s); largest sizes {sizes[:8].tolist()}")

for solver in ("auto", "dense"):
    for reg in (1e-3, 1e-1):
        try:
            m = LocallyLinearEmbedding(n_neighbors=12, n_components=3, reg=reg, eigen_solver=solver,
                                       random_state=42)
            E = m.fit_transform(X)
            t = trustworthiness(X[:2000], E[:2000], n_neighbors=12)
            print(f"solver={solver:5s} reg={reg:g}: reconstruction_error_={m.reconstruction_error_:.3e}, "
                  f"trustworthiness(first 2000 pts)={t:.4f}, embedding std per dim={E.std(0).round(5).tolist()}")
        except Exception as exc:
            print(f"solver={solver} reg={reg:g}: FAILED {type(exc).__name__}: {exc}")
