"""
Independent Component Analysis (FastICA) written from scratch with NumPy.

Idea in one paragraph
---------------------
ICA assumes every observed channel is a *mixture* of a few hidden source signals:
    x = A s + mean          x: C observed channels, s: K hidden sources, A: C x K "mixing" matrix
and tries to recover the sources by finding an "unmixing" matrix W with s = W (x - mean) such that
the recovered sources are as STATISTICALLY INDEPENDENT as possible. PCA only asks components to be
uncorrelated (no linear relation); ICA asks for independence (no relation of any kind), which is much
stronger. The Central Limit Theorem says mixtures look more Gaussian than their ingredients, so ICA
searches for the directions whose projections are the LEAST Gaussian.

Steps (X is N pixels x C channels, K = number of components)
------------------------------------------------------------
 1. centre:   Xc = X - mean
 2. whiten:   PCA on Xc, keep the top K axes, scale each to unit variance
                  Xw = Xc E_K D_K^(-1/2)       E_K: eigenvectors, D_K: eigenvalues (K of them)
              Now the K whitened channels are uncorrelated with variance 1. THIS IS WHERE THE
              DIMENSION DROPS from C to K - everything ICA does afterwards is a rotation inside
              this K-dimensional space.
 3. rotate:   FastICA finds an orthogonal K x K matrix R such that S = Xw R^T has maximally
              non-Gaussian (independent) columns. Fixed-point update for all rows at once
              ("parallel"/"symmetric" FastICA) with g(u) = tanh(alpha u)  (the "logcosh" contrast):
                  R_new = E[g(R xw) xw^T] - diag(E[g'(R xw)]) R
                  R_new = (R_new R_new^T)^(-1/2) R_new      (symmetric decorrelation keeps rows orthonormal)
              Stop when max_i |1 - |<r_i_new, r_i_old>|| < tol (every row stopped turning).
 4. reduce:   S = (X - mean) @ unmixing^T                  unmixing = R D_K^(-1/2) E_K^T   (K x C)
 5. rebuild:  X_hat = S @ mixing^T + mean                  mixing = E_K D_K^(1/2) R^T     (C x K)

Consequence worth knowing: because step 3 is only a rotation inside the PCA subspace of step 2, the
reconstruction X_hat is EXACTLY the K-component PCA reconstruction. ICA does not compress better than
PCA; what changes is what each stored component means (independent sources instead of variance-ordered
axes). test.py verifies this numerically.
"""
import numpy as np


class FastICA:
    def __init__(self, n_components, fun_alpha=1.0, max_iter=1000, tol=1e-4, random_state=42,
                 rank_tol=1e-10):
        self.n_components = int(n_components)
        self.fun_alpha = fun_alpha          # alpha in g(u) = tanh(alpha * u); 1.0 is the standard choice
        self.max_iter = max_iter            # safety limit on fixed-point iterations
        self.tol = tol                      # convergence threshold on the rotation change
        self.random_state = random_state    # seed for the random initial rotation
        self.rank_tol = rank_tol            # eigenvalues below rank_tol * largest count as zero
        # learnt attributes (all set by fit)
        self.mean_ = None                   # (C,)
        self.whitening_ = None              # (K, C)  x_centred -> whitened
        self.rotation_ = None               # (K, K)  orthogonal FastICA rotation R
        self.unmixing_ = None               # (K, C)  x_centred -> sources
        self.mixing_ = None                 # (C, K)  sources -> x_centred
        self.eigenvalues_ = None            # all C eigenvalues of the covariance, largest first
        self.n_iter_ = 0
        self.converged_ = False
        self.history_ = []                  # rotation change after each iteration (convergence curve)
        self.fit_seconds_ = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _sym_decorrelation(w):
        """W <- (W W^T)^(-1/2) W : the closest matrix with orthonormal rows."""
        s, u = np.linalg.eigh(w @ w.T)
        s = np.clip(s, 1e-30, None)
        return (u * (1.0 / np.sqrt(s))) @ u.T @ w

    @staticmethod
    def numerical_rank(eigvals, rank_tol=1e-10):
        """Number of covariance eigenvalues that are not zero up to round-off."""
        return int((eigvals > rank_tol * eigvals.max()).sum())

    # ------------------------------------------------------------------ fitting
    def fit(self, x, w_init=None):
        x = np.asarray(x, dtype=np.float64)
        n, c = x.shape
        k = self.n_components

        # 1. centre
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_

        # 2. whiten with PCA (eigen-decomposition of the C x C covariance)
        cov = xc.T @ xc / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = np.clip(eigvals[order], 0, None), eigvecs[:, order]
        # fixed sign per axis (largest weight positive) so the run is reproducible
        signs = np.sign(eigvecs[np.abs(eigvecs).argmax(axis=0), np.arange(c)])
        signs[signs == 0] = 1
        eigvecs = eigvecs * signs
        self.eigenvalues_ = eigvals
        rank = self.numerical_rank(eigvals, self.rank_tol)
        if k > rank:
            raise ValueError(f"n_components={k} exceeds the numerical rank of the data ({rank}). "
                             "Whitening would divide by ~0 and the extra components would be pure noise.")
        d_k, e_k = eigvals[:k], eigvecs[:, :k]
        self.whitening_ = (e_k / np.sqrt(d_k)).T                     # (K, C)
        xw = xc @ self.whitening_.T                                  # (N, K), identity covariance

        # 3. FastICA, parallel (symmetric) version with g = tanh
        rng = np.random.default_rng(self.random_state)
        w = rng.standard_normal((k, k)) if w_init is None else np.asarray(w_init, dtype=np.float64)
        w = self._sym_decorrelation(w)
        a = self.fun_alpha
        self.history_ = []
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            u = xw @ w.T                                             # (N, K) current source estimates
            g = np.tanh(a * u)
            g_prime = a * (1 - g ** 2)
            w_new = (g.T @ xw) / n - g_prime.mean(axis=0)[:, None] * w
            w_new = self._sym_decorrelation(w_new)
            change = float(np.max(np.abs(np.abs(np.einsum("ij,ij->i", w_new, w)) - 1)))
            self.history_.append(change)
            w = w_new
            if change < self.tol:
                self.converged_ = True
                break
        self.n_iter_ = it
        self.rotation_ = w

        # 4./5. combined matrices
        self.unmixing_ = w @ self.whitening_                         # (K, C)
        self.mixing_ = (e_k * np.sqrt(d_k)) @ w.T                    # (C, K) = pinv(unmixing)

        # ICA leaves order and sign undefined. Order sources by how much signal they carry back
        # (squared norm of their mixing column), and make the largest mixing weight positive.
        energy = (self.mixing_ ** 2).sum(axis=0)
        idx = np.argsort(energy)[::-1]
        sgn = np.sign(self.mixing_[np.abs(self.mixing_[:, idx]).argmax(axis=0), idx])
        sgn[sgn == 0] = 1
        self.rotation_ = self.rotation_[idx] * sgn[:, None]
        self.unmixing_ = self.unmixing_[idx] * sgn[:, None]
        self.mixing_ = self.mixing_[:, idx] * sgn
        self.component_energy_ = energy[idx]
        return self

    # ------------------------------------------------------------------ use
    def transform(self, x):
        """C channels -> K independent sources per pixel."""
        return (np.asarray(x, dtype=np.float64) - self.mean_) @ self.unmixing_.T

    def inverse_transform(self, s):
        """K sources -> C channels (the reconstruction)."""
        return np.asarray(s, dtype=np.float64) @ self.mixing_.T + self.mean_

    def project_onto_whitened_subspace(self, x):
        """Reconstruction using only the whitening (= K-component PCA). Used to show ICA's rotation
        changes nothing about reconstruction quality."""
        xc = np.asarray(x, dtype=np.float64) - self.mean_
        e_k_scaled = np.linalg.pinv(self.whitening_)                # (C, K)
        return (xc @ self.whitening_.T) @ e_k_scaled.T + self.mean_

    def n_model_values(self):
        """Floats a receiver needs to reconstruct: mean (C) + mixing matrix (C x K)."""
        c, k = self.mixing_.shape
        return c + c * k

    # ------------------------------------------------------------------ saving
    def state(self):
        return {"n_components": self.n_components, "fun_alpha": self.fun_alpha, "max_iter": self.max_iter,
                "tol": self.tol, "random_state": self.random_state, "rank_tol": self.rank_tol,
                "mean": self.mean_, "whitening": self.whitening_, "rotation": self.rotation_,
                "unmixing": self.unmixing_, "mixing": self.mixing_, "eigenvalues": self.eigenvalues_,
                "n_iter": self.n_iter_, "converged": self.converged_, "history": np.asarray(self.history_),
                "component_energy": self.component_energy_}

    @classmethod
    def from_state(cls, d):
        m = cls(int(d["n_components"]), float(d["fun_alpha"]), int(d["max_iter"]), float(d["tol"]),
                int(d["random_state"]), float(d["rank_tol"]))
        for attr in ["mean", "whitening", "rotation", "unmixing", "mixing", "eigenvalues", "component_energy"]:
            setattr(m, attr + "_", np.asarray(d[attr]))
        m.n_iter_ = int(d["n_iter"])
        m.converged_ = bool(d["converged"])
        m.history_ = list(np.asarray(d["history"]))
        return m
