"""
Linear Discriminant Analysis (Fisher LDA) written from scratch with NumPy.

Idea in one paragraph
---------------------
PCA and ICA are UNSUPERVISED: they only look at the pixels. LDA is SUPERVISED: every training pixel
must come with a class label (e.g. water / vegetation / built-up / bare soil). LDA then looks for the
projection that pushes the class MEANS as far apart as possible while keeping each class as TIGHT as
possible. It keeps whatever separates the classes and throws away everything else, even high-variance
information, so it is a reduction "for classification", not "for reconstruction".

Maths (X: N x D features, y: class of each row, c classes)
---------------------------------------------------------
  mu       = overall mean,  mu_k = mean of class k,  N_k = pixels in class k
  S_W      = sum_k sum_{x in k} (x - mu_k)(x - mu_k)^T        within-class scatter (spread inside classes)
  S_B      = sum_k N_k (mu_k - mu)(mu_k - mu)^T                between-class scatter (spread of class means)
  Fisher's criterion: find w maximising  J(w) = (w^T S_B w) / (w^T S_W w)
  Solution: the generalised eigenproblem  S_B w = lambda S_W w ; keep eigenvectors with the largest lambda.

  Maximum number of components: S_B is a sum of c rank-1 terms whose weighted sum is zero, so its rank is
  at most c - 1. There are therefore at most min(c - 1, D) useful directions - with 2 classes LDA can only
  give ONE component, whatever D is.

  Shrinkage (optional): S_W <- (1 - a) S_W + a * (trace(S_W)/D) I keeps S_W invertible when channels are
  duplicated (as RGB:B04 / False_Color:B04 are here) or classes are small.

Solving: whiten S_W (S_W = U diag(s) U^T -> P = U diag(s)^-1/2), then ordinary eigen-decomposition of
P^T S_B P. This is numerically stable and needs only symmetric eigensolvers.

Classification: a pixel goes to the class whose mean is nearest in the LDA space, corrected by
log prior. With all components kept this is exactly the Gaussian shared-covariance LDA classifier.
"""
import numpy as np


class LDA:
    def __init__(self, n_components=None, shrinkage=0.0, rank_tol=1e-10):
        self.n_components = n_components   # None = maximum allowed, min(n_classes - 1, n_features)
        self.shrinkage = shrinkage         # 0 = classic LDA; 0 < a <= 1 blends S_W towards a scaled identity
        self.rank_tol = rank_tol           # relative eigenvalue threshold for "zero"
        self.classes_ = None
        self.priors_ = None
        self.means_ = None                 # (c, D) class means in input space
        self.xbar_ = None                  # (D,) overall mean
        self.scalings_ = None              # (D, K) projection matrix
        self.eigenvalues_ = None           # Fisher ratio of each kept direction
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

        xc = x - self.means_[y_idx]                                   # each pixel minus its class mean
        sw = xc.T @ xc                                                # within-class scatter
        dm = self.means_ - self.xbar_
        sb = (dm * counts[:, None]).T @ dm                            # between-class scatter
        if self.shrinkage:
            a = float(self.shrinkage)
            sw = (1 - a) * sw + a * np.trace(sw) / d * np.eye(d)

        # whiten S_W, dropping directions where S_W is ~0 (duplicated channels)
        s, u = np.linalg.eigh(sw)
        keep = s > self.rank_tol * s.max()
        p = u[:, keep] / np.sqrt(s[keep])                             # D x r, P^T S_W P = I
        lam, v = np.linalg.eigh(p.T @ sb @ p)                         # ordinary symmetric eigenproblem
        order = np.argsort(lam)[::-1]
        lam, v = np.clip(lam[order], 0, None), v[:, order]
        w = p @ v                                                      # D x r generalised eigenvectors
        # fixed sign: largest absolute weight of each direction positive (reproducible output)
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
        """Score per class: -0.5 * squared distance to the class mean in LDA space + log prior.
        (In the S_W-whitened LDA space the shared within-class covariance is the identity, up to the
        constant N - c, which is applied here.)"""
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
        """Floats needed to apply the reduction: overall mean (D) + projection (D x K)."""
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
