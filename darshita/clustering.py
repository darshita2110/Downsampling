"""
clustering.py - the graph / clustering-based channel selection methods.

The idea in one line:
    put channels that look alike into the same group (cluster), make exactly K groups,
    and send ONE channel from each group.

Think of the channels as points (nodes of a graph) with a "distance" between every pair:
small distance = the two channels carry the same information. Because each group sends
only one channel, near-copies can never be sent together.

Methods (they differ in how "distance" is measured and how groups are made):
  1. WaLuDi - Ward clustering + KL divergence  (how different two channels' value patterns are)
  2. WaLuMI - Ward clustering + mutual information (how much one channel tells about another)
  3. Affinity Propagation - channels "vote" for representatives by passing messages
(WaLuDi / WaLuMI: Martinez-Uso et al., 2007.  Affinity Propagation: Frey & Dueck, 2007.)
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ================================================================ distances between channels
def kl_distance(pixels01):
    """Symmetric KL divergence between every pair of channels.
    Each channel is turned into a probability distribution over the pixels (values / sum).
        D(i, j) = sum over pixels of (p_i - p_j) * (log p_i - log p_j)
    0 = identical patterns, bigger = more different."""
    P = pixels01 + 1e-6
    P = P / P.sum(axis=0, keepdims=True)
    L = np.log(P)
    a = (P * L).sum(axis=0)                       # sum p_i log p_i for every channel
    cross = P.T @ L                               # sum p_i log p_j for every pair
    D = a[:, None] + a[None, :] - cross - cross.T
    return np.clip(D, 0, None)


def mi_distance(pixels01, levels=32):
    """Distance from mutual information ("variation of information"):
        D(i, j) = H(i) + H(j) - 2 * I(i; j)
    H = entropy of a channel, I = information the two channels share.
    0 = one channel tells you everything about the other."""
    Q = np.minimum((pixels01 * levels).astype(np.int64), levels - 1)   # values -> 0..levels-1
    n, C = Q.shape

    def entropy(counts):
        p = counts[counts > 0] / n
        return -(p * np.log2(p)).sum()

    H = np.array([entropy(np.bincount(Q[:, c], minlength=levels)) for c in range(C)])
    D = np.zeros((C, C))
    for i in range(C):
        for j in range(i + 1, C):
            joint = np.bincount(Q[:, i] * levels + Q[:, j], minlength=levels * levels)
            I = H[i] + H[j] - entropy(joint)                 # mutual information
            D[i, j] = D[j, i] = H[i] + H[j] - 2 * I
    return np.clip(D, 0, None)


# ================================================================ helpers
def representative(D, members):
    """The channel in a group that is closest (in total) to all the others = the group's centre."""
    sub = D[np.ix_(members, members)]
    return int(members[sub.sum(axis=1).argmin()])


def ward_groups(D, k):
    """Ward hierarchical clustering: start with every channel alone, keep merging the two most
    similar groups until only k groups are left. Returns the group number of every channel."""
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    tree = linkage(squareform(D, checks=False), method="ward")
    return fcluster(tree, t=k, criterion="maxclust"), tree


def exactly_k(selected, D, k):
    """Clustering can return one group too many or too few when distances tie. Fix the count."""
    selected = list(dict.fromkeys(selected))
    while len(selected) > k:                       # drop the channel closest to another selected one
        sub = D[np.ix_(selected, selected)] + np.eye(len(selected)) * 1e18
        selected.pop(int(sub.min(axis=1).argmin()))
    while len(selected) < k:                       # add the channel farthest from all selected ones
        rest = [c for c in range(D.shape[0]) if c not in selected]
        selected.append(rest[int(D[np.ix_(rest, selected)].min(axis=1).argmax())])
    return sorted(int(c) for c in selected)


# ================================================================ method 1 and 2
def ward_select(D, k):
    groups, tree = ward_groups(D, k)
    chosen = [representative(D, np.where(groups == g)[0]) for g in np.unique(groups)]
    return exactly_k(chosen, D, k), groups, tree


def waludi(pixels01, k):
    D = kl_distance(pixels01)
    chosen, groups, tree = ward_select(D, k)
    return chosen, groups, tree, D


def walumi(pixels01, k):
    D = mi_distance(pixels01)
    chosen, groups, tree = ward_select(D, k)
    return chosen, groups, tree, D


# ================================================================ method 3
def affinity_propagation(S, preference, damping=0.9, iterations=300):
    """
    S = similarity between channels (higher = more alike). Two kinds of messages are passed:
      responsibility r(i, k): how well channel k suits being channel i's representative
      availability   a(i, k): how much channel k is "available" to represent others
    After enough rounds, channels with r(k, k) + a(k, k) > 0 are the representatives (exemplars).
    `preference` (diagonal of S) controls how many representatives come out: higher -> more.
    """
    S = S.copy()
    np.fill_diagonal(S, preference)
    n = S.shape[0]
    R = np.zeros((n, n))
    A = np.zeros((n, n))
    rows = np.arange(n)
    for _ in range(iterations):
        # responsibilities
        AS = A + S
        top = AS.argmax(axis=1)
        first = AS[rows, top]
        AS[rows, top] = -np.inf
        second = AS.max(axis=1)
        R_new = S - first[:, None]
        R_new[rows, top] = S[rows, top] - second
        R = damping * R + (1 - damping) * R_new
        # availabilities
        Rp = np.maximum(R, 0)
        Rp[rows, rows] = R[rows, rows]
        A_new = Rp.sum(axis=0)[None, :] - Rp
        diag = A_new[rows, rows].copy()
        A_new = np.minimum(A_new, 0)
        A_new[rows, rows] = diag
        A = damping * A + (1 - damping) * A_new
    exemplars = np.where(np.diag(A + R) > 0)[0]
    labels = (S[:, exemplars].argmax(axis=1) if len(exemplars) else np.zeros(n, int))
    if len(exemplars):
        labels[exemplars] = np.arange(len(exemplars))
    return exemplars, labels


def ap_select(pixels01, k):
    """Affinity Propagation finds the number of groups by itself. To get exactly k, try different
    `preference` values (bisection) until it returns k representatives."""
    corr = np.abs(np.corrcoef(pixels01, rowvar=False))
    S = corr - 1.0                                  # 0 = identical channels, -1 = unrelated
    D = 1.0 - corr
    low, high = S.min() * 5, 0.0
    best, best_gap, best_labels = None, np.inf, None
    for _ in range(30):
        pref = (low + high) / 2
        ex, labels = affinity_propagation(S, pref)
        if len(ex) and abs(len(ex) - k) < best_gap:
            best, best_gap, best_labels = ex, abs(len(ex) - k), labels
        if len(ex) == k:
            break
        if len(ex) < k:
            low = pref                              # too few groups -> raise the preference
        else:
            high = pref
    if best is None:
        best, best_labels = [0], np.zeros(len(S), int)
    return exactly_k(list(map(int, best)), D, k), best_labels, D
