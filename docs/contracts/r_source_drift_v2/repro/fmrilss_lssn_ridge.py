"""LSS-05/06: LSS-N trial groups and fractional ridge (fmrilss 61b65b4, 28ba1bc).

Transcribes R .lss_weight_matrix / .lss_weight_matrix_grouped (R/lss_kernel.R)
and checks (a) the transcription against a naive per-trial GLM, (b) classic
LSS (no ridge, no groups) against fmrimod lss_single_trial, (c) fractional
ridge (no groups) against fmrimod OASIS ridge_mode='fractional' (R's own test
asserts lss(ridge=) == OASIS fractional), (d) how far LSS-N moves betas.
"""
import numpy as np
from scipy.stats import gamma

from fmrimod.single import OasisConfig
from fmrimod.single.lss import lss_single_trial
from fmrimod.single.oasis import oasis_single_trial


def resid_trials(C, Xc):
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    r = int(np.sum(s > max(Xc.shape) * np.finfo(float).eps * s[0]))
    Q = U[:, :r]
    return C - Q @ (Q.T @ C), Q


def weight_matrix(C, groups=None, eps=1e-12, ridge=(0.0, 0.0)):
    T = C.shape[1]
    CtC = (C ** 2).sum(0)
    lx = ridge[0] * CtC.mean()
    if T == 1:
        cc = CtC + lx
        return C * 0 if cc <= eps else C / cc
    if groups is None:
        tot = C.sum(1)
        CtT = C.T @ tot
        bt2 = tot @ tot - 2 * CtT + CtC
        ctb = CtT - CtC
        bt2 = bt2 + ridge[1] * bt2.mean()
        bt2 = np.where(bt2 < eps, np.inf, bt2)
        alpha = ctb / bt2
        den = np.maximum(CtC + lx - ctb ** 2 / bt2, eps)
        return C * ((1 + alpha) / den) - np.outer(tot, alpha / den)
    G = groups.max() + 1
    A = np.column_stack([C[:, groups == g].sum(1) for g in range(G)])
    AtA, AtC = A.T @ A, A.T @ C
    lb = 0.0
    if ridge[1] > 0:
        diag_all = np.repeat(np.diag(AtA)[:, None], T, 1)
        own = np.diag(AtA)[groups] - 2 * AtC[groups, np.arange(T)] + CtC
        diag_all[groups, np.arange(T)] = own
        lb = ridge[1] * diag_all[diag_all > eps].mean()
    s = np.zeros(T); U = np.zeros((G, T))
    for i in range(T):
        g = groups[i]
        a_c = AtC[:, i]
        M = AtA.copy(); M[g, :] -= a_c; M[:, g] -= a_c; M[g, g] += CtC[i]
        m = a_c.copy(); m[g] -= CtC[i]
        keep = np.diag(M) > eps
        if not keep.any():
            s[i] = 1 / (CtC[i] + lx) if CtC[i] + lx > eps else 0; continue
        Mk = M[np.ix_(keep, keep)] + lb * np.eye(keep.sum())
        h = np.linalg.solve(Mk, m[keep])
        den = max(CtC[i] + lx - m[keep] @ h, eps)
        hk = np.zeros(G); hk[keep] = h
        s[i] = (1 + hk[g]) / den; U[:, i] = -hk / den
    return C * s + A @ U


def naive(Y, C, Z, groups=None):
    Cr, Q = resid_trials(C, Z); Yr = Y - Q @ (Q.T @ Y)
    out = np.zeros((C.shape[1], Y.shape[1]))
    for i in range(C.shape[1]):
        if groups is None:
            D = np.column_stack([Cr[:, i], np.delete(Cr, i, 1).sum(1)])
        else:
            others = [Cr[:, (groups == g) & (np.arange(C.shape[1]) != i)].sum(1) for g in range(groups.max() + 1)]
            others = [o for o in others if o @ o > 0]
            D = np.column_stack([Cr[:, i]] + others)
        out[i] = np.linalg.pinv(D)[0] @ Yr
    return out


def main(seed=11):
    rng = np.random.default_rng(seed)
    n, T, V = 160, 24, 12
    onsets = np.sort(rng.choice(np.arange(3, n - 18 + 1), T, replace=False))
    t = np.arange(16); h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    C = np.zeros((n, T))
    for j, o in enumerate(onsets):
        s = np.zeros(n); s[o - 1] = 1; c = np.convolve(s, h)[:n]; c[:15] = 0; C[:, j] = c
    Z = np.column_stack([np.ones(n), np.arange(1, n + 1) / n])
    Nn = rng.standard_normal((n, 3))
    Y = C @ (rng.standard_normal((T, V)) + np.repeat([[3.0], [-3.0], [0.0]], T // 3, 0)) + rng.standard_normal((n, V))
    K = np.hstack([Z, Nn])
    groups = np.tile([0, 1, 2], T // 3)

    Cr, _ = resid_trials(C, K)
    b_cls = weight_matrix(Cr).T @ Y
    print("classic: |R kernel - naive|      =", float(np.abs(b_cls - naive(Y, C, K)).max()))
    print("classic: |R kernel - fmrimod lss| =",
          float(np.abs(b_cls - lss_single_trial(Y, C, confounds=K).betas).max()))
    b_grp = weight_matrix(Cr, groups).T @ Y
    print("LSS-N  : |R kernel - naive LSS-N| =", float(np.abs(b_grp - naive(Y, C, K, groups)).max()))
    print("LSS-N vs classic max |diff| (fmrimod has no LSS-N) =", float(np.abs(b_grp - b_cls).max()))
    for rg in [(0.3, 0.3), (0.1, 0.5), (0.5, 0.0)]:
        b_r = weight_matrix(Cr, ridge=rg).T @ Y
        o = oasis_single_trial(Y, C, confounds=K,
                               config=OasisConfig(ridge_mode="fractional", ridge_x=rg[0], ridge_b=rg[1])).betas
        print(f"ridge {rg}: |R lss(ridge) - fmrimod OASIS fractional| = {float(np.abs(b_r - o).max()):.2e}")


if __name__ == "__main__":
    main()
