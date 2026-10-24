"""REG-A: near-collinear design + large offset (R P-D4 a369e95;
test-review-D-solver.R 'solve_glm_core betas and RSS match lm on a
near-collinear design').  R solves from one pivoted QR (Q'Y) for betas and
exact RSS.  fmrimod fast_preproject uses Cholesky of X'X whenever
sqrt(cond(X'X)) <= 1/sqrt(eps) and RSS = y'y - b'X'y with a fallback."""
import numpy as np

from fmrimod.glm.solver import fast_lm_matrix, fast_preproject

rng = np.random.default_rng(11)
n = 200
x1 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1, x1 + 1e-7 * rng.normal(size=n), rng.normal(size=n)])
Y = 1000 + X @ np.column_stack([[0, 1, 2, -1], [0, -1, 0.5, 3]]) + rng.normal(size=(n, 2))
print("cond(X) =", f"{np.linalg.cond(X):.3g}")
ref_b, *_ = np.linalg.lstsq(X, Y, rcond=None)
Q, R = np.linalg.qr(X)
ref_b = np.linalg.solve(R, Q.T @ Y)
ref_rss = np.sum((Y - X @ ref_b) ** 2, axis=0)
proj = fast_preproject(X)
print("fmrimod path: ill_conditioned =", proj.ill_conditioned, " XtX cached =", proj.XtX is not None)
for rf in (False, True):
    res = fast_lm_matrix(X, Y, proj, return_fitted=rf)
    rel_b = np.max(np.abs(res.betas - ref_b)) / np.max(np.abs(ref_b))
    rel_rss = np.max(np.abs(res.rss - ref_rss) / ref_rss)
    print(f"return_fitted={rf}: max rel beta err={rel_b:.3e}  max rel RSS err={rel_rss:.3e}  (R test tol: 1e-6 / 1e-10)")
res = fast_lm_matrix(X, Y, proj, return_fitted=False)
print("ref RSS:", np.round(ref_rss, 3), " fmrimod RSS:", np.round(res.rss, 3))
print("ref intercepts:", np.round(ref_b[0], 4), " fmrimod:", np.round(res.betas[0], 4))
# the same through fmri_lm-level t for the well-determined column b (index 3)
se_ref = np.sqrt(ref_rss / (n - 4) * np.diag(np.linalg.inv(R.T @ R))[3])
se_fm = np.sqrt(res.sigma2 * proj.XtXinv[3, 3])
print("t for column b: ref", np.round(ref_b[3] / se_ref, 3), " fmrimod", np.round(res.betas[3] / se_fm, 3))
