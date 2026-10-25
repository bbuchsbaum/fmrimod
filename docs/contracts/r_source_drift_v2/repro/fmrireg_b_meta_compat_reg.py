"""REG-B: fmri_meta_fit (meta_compat) with covariates vs fmrireg meta_fit_cov_cpp.

R (src/meta_kernels.cpp at HEAD, pinned against metafor::rma by
tests/testthat/test-review-B-meta-estimators.R, B5): for a design X with
covariates, DL uses the regression Q and C = sum(w) - sum(w^2 x'Ax); PM solves
Q_reg(tau2) = m - p; Q_fe is the residual Q of the FE regression.

fmrimod HEAD: meta_compat._fit_feature always uses the intercept-only tau2 and
an intercept-only Q, whatever X is.
"""
import numpy as np

import fmrimod

rng = np.random.default_rng(11)
S, P = 15, 6
x = rng.normal(size=S)
g = np.array(["a", "b", "c"] * 5)
X = np.column_stack([np.ones(S), x, (g == "b").astype(float), (g == "c").astype(float)])
V = rng.uniform(0.02, 0.4, size=(S, P))
tau = np.linspace(0, 0.6, P)
Y = 0.3 + 1.5 * x[:, None] + rng.normal(size=(S, P)) * np.sqrt(V + tau**2)


def q_tau2(y, v, t2):
    w = 1 / (v + t2)
    A = np.linalg.inv((X * w[:, None]).T @ X)
    b = A @ X.T @ (w * y)
    r = y - X @ b
    return float(np.sum(w * r * r)), b, A


def tau2_dl(y, v):
    w = 1 / v
    q0, _, A = q_tau2(y, v, 0.0)
    df = S - X.shape[1]
    t = np.einsum("ij,jk,ik->i", X, A, X)
    den = w.sum() - np.sum(w * w * t)
    return max(0.0, (q0 - df) / den) if den > 0 else 0.0


def tau2_pm(y, v, tol=1e-6, max_iter=50):
    df = S - X.shape[1]
    f0 = q_tau2(y, v, 0.0)[0] - df
    if f0 <= 0:
        return 0.0
    hi = max(1.0, 1000 * np.median(v))
    for _ in range(20):
        if q_tau2(y, v, hi)[0] - df <= 0:
            break
        hi *= 2
    a, b = 0.0, hi
    mid = a
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = q_tau2(y, v, mid)[0] - df
        if abs(fm) < tol:
            break
        if fm > 0:
            a = mid
        else:
            b = mid
    return mid


for method, ref_fn in (("dl", tau2_dl), ("pm", tau2_pm)):
    out = fmrimod.fmri_meta_fit(Y, V, X, method=method)
    print(f"== method={method} ==")
    for j in range(P):
        t2 = ref_fn(Y[:, j], V[:, j])
        _, b, A = q_tau2(Y[:, j], V[:, j], t2)
        qfe = q_tau2(Y[:, j], V[:, j], 0.0)[0]
        print(f"vox{j}: tau2 fmrimod={out['tau2'][j]:.5f} R={t2:.5f} | "
              f"Q_fe fmrimod={out['Q_fe'][j]:.3f} R={qfe:.3f} | "
              f"se(x) fmrimod={out['se'][1, j]:.5f} R={np.sqrt(A[1, 1]):.5f}")

# cross-check the transcription against fmrimod's own R-oracle-tested DL reg reducer
from fmrimod.group import reducers as R

print("\nnative group meta_re_reg (R-oracle parity-tested) tau2, for the DL transcription check:")
try:
    import inspect
    print(inspect.signature(R.meta_re_reg))
except Exception as e:  # pragma: no cover
    print("signature unavailable", e)

# Transcription cross-check: fmrimod's own meta.py DL meta-regression
# (_dl_tau2_regression, parity port of fmrigds meta:re_reg) must equal tau2_dl above.
from fmrimod.stats.meta import _dl_tau2_regression

print("\ntranscription check (meta.py _dl_tau2_regression vs tau2_dl):",
      np.allclose([_dl_tau2_regression(Y[:, j], X, V[:, j]) for j in range(P)],
                  [tau2_dl(Y[:, j], V[:, j]) for j in range(P)]))
