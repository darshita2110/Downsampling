"""
Linear Discriminant Analysis (Fisher LDA) written from scratch with NumPy (methodology unchanged).

PCA and ICA are UNSUPERVISED: they only look at the pixels. LDA is SUPERVISED: every training pixel
must come with a class label. LDA looks for the projection that pushes the class MEANS as far apart as
possible while keeping each class as TIGHT as possible - a reduction "for classification", not "for
reconstruction" (it has no inverse transform).

Maths (X: N x D features, y: class of each row, c classes)
  S_W = sum_k sum_{x in k} (x - mu_k)(x - mu_k)^T      within-class scatter
  S_B = sum_k N_k (mu_k - mu)(mu_k - mu)^T              between-class scatter
  maximise J(w) = (w^T S_B w) / (w^T S_W w)  ->  S_B w = lambda S_W w
  at most min(c - 1, D) useful directions.
  Shrinkage (optional): S_W <- (1 - a) S_W + a * (trace(S_W)/D) I keeps S_W invertible with duplicated channels.
Solving: whiten S_W, then an ordinary symmetric eigen-decomposition of P^T S_B P.
Classification: nearest class mean in the LDA space, corrected by log prior.
"""
import numpy as np


class LDA:
    def __init__(self, n_components=None, shrinkage=0.0, rank_tol=1e-10):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.rank_tol = rank_tol
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.xbar_ = None
        self.scalings_ = None
        self.eigenvalues_ = None
        self.explained_discriminant_ratio_ = None
        self.max_components_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
        c, d = len(self.classes_), x.shape[1]
        if c < 2:
            raise ValueError("LDA needs at least 2 classes")
        self.max_components_ = min(c - 1, d)
        k = self.max_components_ if self.n_components is None else int(self.n_components)
        if not 1 <= k <= self.max_components_:
            raise ValueError(f"n_components={k} not allowed: with {c} classes and {d} features LDA has at most "
                             f"min(c - 1, D) = {self.max_components_} components")
        self.priors_ = counts / counts.sum()
        self.xbar_ = x.mean(0)
        self.means_ = np.stack([x[y_idx == i].mean(0) for i in range(c)])
        xc = x - self.means_[y_idx]
        sw = xc.T @ xc
        dm = self.means_ - self.xbar_
        sb = (dm * counts[:, None]).T @ dm
        if self.shrinkage:
            a = float(self.shrinkage)
            sw = (1 - a) * sw + a * np.trace(sw) / d * np.eye(d)
        s, u = np.linalg.eigh(sw)
        keep = s > self.rank_tol * s.max()
        p = u[:, keep] / np.sqrt(s[keep])
        lam, v = np.linalg.eigh(p.T @ sb @ p)
        order = np.argsort(lam)[::-1]
        lam, v = np.clip(lam[order], 0, None), v[:, order]
        w = p @ v
        sgn = np.sign(w[np.abs(w).argmax(0), np.arange(w.shape[1])])
        sgn[sgn == 0] = 1
        w = w * sgn
        self.n_zero_variance_dirs_ = int((~keep).sum())
        useful = lam[:self.max_components_]
        self.all_eigenvalues_ = useful
        self.explained_discriminant_ratio_ = useful / useful.sum() if useful.sum() > 0 else useful
        self.scalings_ = w[:, :k]
        self.eigenvalues_ = lam[:k]
        return self

    def transform(self, x):
        """D channels -> K discriminant components per pixel."""
        return (np.asarray(x, dtype=np.float64) - self.xbar_) @ self.scalings_

    def decision_function(self, x):
        z = self.transform(x)
        n_minus_c = getattr(self, "n_samples_minus_classes_", 1.0)
        mz = (self.means_ - self.xbar_) @ self.scalings_
        d2 = ((z[:, None, :] - mz[None, :, :]) ** 2).sum(-1) * n_minus_c
        return -0.5 * d2 + np.log(self.priors_)

    def predict(self, x, batch=200_000):
        out = np.empty(len(x), dtype=self.classes_.dtype)
        for i in range(0, len(x), batch):
            out[i:i + batch] = self.classes_[self.decision_function(x[i:i + batch]).argmax(1)]
        return out

    def fit_with_counts(self, x, y):
        """fit() and remember N - c, which scales the whitened space to the unbiased pooled covariance."""
        self.fit(x, y)
        self.n_samples_minus_classes_ = float(len(x) - len(self.classes_))
        return self

    def n_model_values(self):
        d, k = self.scalings_.shape
        return d + d * k

    def state(self):
        return {"n_components": self.scalings_.shape[1], "shrinkage": self.shrinkage, "rank_tol": self.rank_tol,
                "classes": self.classes_, "priors": self.priors_, "means": self.means_, "xbar": self.xbar_,
                "scalings": self.scalings_, "eigenvalues": self.eigenvalues_,
                "all_eigenvalues": self.all_eigenvalues_,
                "explained_discriminant_ratio": self.explained_discriminant_ratio_,
                "max_components": self.max_components_,
                "n_samples_minus_classes": getattr(self, "n_samples_minus_classes_", 1.0)}

    @classmethod
    def from_state(cls, d):
        m = cls(int(d["n_components"]), float(d["shrinkage"]), float(d["rank_tol"]))
        for a in ["classes", "priors", "means", "xbar", "scalings", "eigenvalues", "all_eigenvalues",
                  "explained_discriminant_ratio"]:
            setattr(m, a + "_", np.asarray(d[a]))
        m.max_components_ = int(d["max_components"])
        m.n_samples_minus_classes_ = float(d["n_samples_minus_classes"])
        return m
