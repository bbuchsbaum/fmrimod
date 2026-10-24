"""LSS-01: noise model from aggregate-trial residuals (fmrilss a9d816e).

Transcribes the R fixture from tests/testthat/test-lss-kernel.R
("aggregate noise residuals avoid the many-trial AR bias") and the R
aggregate path (.aggregate_trials + .noise_basis + fmriAR::fit_noise) in
NumPy, using fmrimod's own fit_noise/whiten_apply (the fmriAR port) as the
AR engine, then compares with fmrimod's prewhiten_matrices (full LSA design).
"""
import numpy as np
from scipy.stats import gamma

from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.whitening import whiten_apply
from fmrimod.single import estimate_single_trial
from fmrimod.single._prewhiten import PrewhitenConfig, prewhiten_matrices
from fmrimod.single.lss import lss_single_trial


def r_filter_sides1(x, h):
    # stats::filter(x, h, sides = 1); NA for i < len(h) -> 0
    out = np.convolve(x, h)[: len(x)]
    out[: len(h) - 1] = 0.0
    return out


def ar1_recursive(e, phi):
    out = np.empty_like(e)
    acc = 0.0
    for t, v in enumerate(e):
        acc = v + phi * acc
        out[t] = acc
    return out


def noise_basis(design, n):
    # R .noise_basis: add intercept unless already in span; rank-revealing Q
    ones = np.ones((n, 1))
    Q, R = np.linalg.qr(design)
    if np.linalg.norm(ones - Q @ (Q.T @ ones)) >= 1e-8:
        design = np.hstack([ones, design])
    U, s, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(s > max(design.shape) * np.finfo(float).eps * s[0]))
    return U[:, :rank]


def main(seed=21):
    rng = np.random.default_rng(seed)
    n, T, V = 240, 90, 200
    onsets = np.sort(rng.choice(np.arange(3, n - 18 + 1), T, replace=False))
    t = np.arange(16)
    h = gamma.pdf(t, 6, scale=1) - gamma.pdf(t, 16, scale=1) / 6
    X = np.zeros((n, T))
    for j, o in enumerate(onsets):
        s = np.zeros(n)
        s[o - 1] = 1.0
        X[:, j] = r_filter_sides1(s, h)
    noise = np.column_stack([ar1_recursive(rng.standard_normal(n), 0.4) for _ in range(V)])
    B = rng.normal(1.0, 0.3, size=(T, V))
    Y = X @ B + noise
    Z = np.ones((n, 1))  # lss() default Z = intercept

    # --- fmrimod HEAD: full design cbind(Z, X) residuals ---
    cfg = PrewhitenConfig(method="ar", p=1)
    pw_full = prewhiten_matrices(Y, X, None, cfg, baseline_regressors=Z)
    phi_full = float(np.asarray(pw_full.plan.phi[0])[0])

    # --- R HEAD default: residual_model = "aggregate" -> cbind(Z, rowSums(X)) ---
    Qn = noise_basis(np.hstack([Z, X.sum(axis=1, keepdims=True)]), n)
    resid_agg = Y - Qn @ (Qn.T @ Y)
    plan_agg = fit_noise(resid=resid_agg, method="ar", p=1, pooling="global", exact_first="ar1")
    phi_agg = float(np.asarray(plan_agg.phi[0])[0])

    # Betas under each plan (whiten [Z, X] and Y with the plan; LSS with whitened Z)
    def betas_for(plan):
        wh = whiten_apply(plan, np.hstack([Z, X]), Y)
        Zw, Xw = wh.X[:, :1], wh.X[:, 1:]
        return lss_single_trial(wh.Y, Xw, baseline_regressors=Zw).betas

    b_full = betas_for(pw_full.plan)
    b_agg = betas_for(plan_agg)
    rmse = lambda b: float(np.sqrt(np.mean((b - B) ** 2)))

    # Public dispatcher path (what users hit)
    res = estimate_single_trial(Y, X, method="lss", prewhiten=cfg, baseline_regressors=Z)
    b_pub = res.betas
    phi_pub = float(np.asarray(res.extra.whitening_plan.phi[0])[0])

    print("true phi = 0.40")
    print(f"fmrimod HEAD phi (full LSA residuals)      = {phi_full:+.4f}")
    print(f"fmrimod public dispatcher phi              = {phi_pub:+.4f}")
    print(f"R HEAD default phi (aggregate residuals)   = {phi_agg:+.4f}")
    print(f"R assertion |phi_agg-0.4|<0.1: {abs(phi_agg-0.4) < 0.1}; phi_full < phi_agg-0.15: {phi_full < phi_agg - 0.15}")
    print(f"max |beta_fmrimod - beta_R_agg| = {np.max(np.abs(b_pub - b_agg)):.4f}")
    print(f"beta RMSE vs truth: fmrimod={rmse(b_pub):.4f}  R-aggregate={rmse(b_agg):.4f}  (full via helper={rmse(b_full):.4f})")


if __name__ == "__main__":
    main()
