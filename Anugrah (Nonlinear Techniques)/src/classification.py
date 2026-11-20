"""Shared classification check for labelled hyperspectral scenes.

5-NN classifier on an embedding, scored with overall accuracy (OA), average
per-class accuracy (AA) and Cohen's kappa under two 5-fold schemes:
  - random stratified folds (optimistic: neighbouring pixels leak across folds)
  - spatial-block folds (whole 5x5-grid blocks of the scene held out together)
"""
import warnings

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

KNN_K = 5


def spatial_block_groups(row, col, grid_shape, n_blocks_per_side=5):
    """Assign each pixel to one of n_blocks_per_side**2 equal spatial blocks."""
    h, w = grid_shape
    return (row * n_blocks_per_side // h) * n_blocks_per_side + (col * n_blocks_per_side // w)


def classification_scores(X, y, groups, random_state=42):
    out = {}
    clf = KNeighborsClassifier(n_neighbors=KNN_K)
    schemes = {
        "random_5fold": (StratifiedKFold(5, shuffle=True, random_state=random_state), None),
        "spatial_block_5fold": (GroupKFold(5), groups),
    }
    for name, (cv, grp) in schemes.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # classes with < 5 members
            pred = cross_val_predict(clf, X, y, cv=cv, groups=grp)
        cm = confusion_matrix(y, pred, labels=np.unique(y))
        per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        out[name] = {
            "overall_accuracy": float((pred == y).mean()),
            "average_class_accuracy": float(per_class.mean()),
            "kappa": float(cohen_kappa_score(y, pred)),
        }
    return out
