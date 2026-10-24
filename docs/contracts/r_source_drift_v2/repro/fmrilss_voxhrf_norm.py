"""LSS-02: voxel-HRF shape normalization (fmrilss 0d97c8b / 6a2588b).

R .normalize_voxel_hrf_coefficients (HEAD) is transcribed below: waveforms of
the basis on a 0.05 s grid (baseline-subtracted) are oriented to correlate
positively with ref_hrf (default SPMG1), divided by their positive peak, and
flagged `degenerate` (scaled by max |waveform|) when the peak is < 5% of the
largest absolute deflection. fmrimod HEAD instead divides by the L2 norm of
the raw coefficients (fmrimod/single/voxel_hrf.py:114-117) with no
orientation. The LSS-with-HRF kernel itself is shared (fmrimod
lss_with_voxel_hrf), so differences come only from the normalization.
"""
import numpy as np
from scipy.stats import gamma

from fmrimod.single._types import VoxelHrfResult
from fmrimod.single.voxel_hrf import estimate_voxel_hrf, lss_with_voxel_hrf

SPAN = 24.0


def spmg1(t):
    # SPM canonical double gamma (shape only; normalization is irrelevant here)
    t = np.asarray(t, float)
    return gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0


def basis_fun(t):
    # 2-column basis: canonical + finite-difference temporal derivative
    d = (spmg1(t) - spmg1(t - 1.0))
    return np.column_stack([spmg1(t), d])


def r_normalize(coef, ref=spmg1, precision=0.05, peak_tol=0.05):
    grid = np.arange(0.0, SPAN + 1e-9, precision)
    H = basis_fun(grid)
    H = H - H[0, :]
    r = ref(grid) - ref(grid)[0]
    wf = H @ coef
    orient = np.sign(r @ wf)
    orient[~np.isfinite(orient) | (orient == 0)] = 1.0
    peak = (wf * orient).max(axis=0)
    amp = np.abs(wf).max(axis=0)
    degen = ~np.isfinite(peak) | ~(peak >= peak_tol * amp) | (amp <= 0)
    peak = np.where(degen, amp, peak)
    peak = np.where(~np.isfinite(peak) | (peak <= 0), 1.0, peak)
    scale = orient * peak
    return coef / scale, scale, degen


def main(seed=3):
    rng = np.random.default_rng(seed)
    n, TR = 300, 1.0
    onsets = np.arange(10, 280, 14.0)
    N, K = len(onsets), 2
    lag = np.arange(0, SPAN + 1e-9, TR)
    Bk = basis_fun(lag)
    X = np.zeros((n, N * K))
    for i, o in enumerate(onsets):
        idx = int(o)
        L = min(len(lag), n - idx)
        X[idx: idx + L, i * K: (i + 1) * K] = Bk[:L]
    amps = rng.normal(1.0, 0.2, N)
    shape_pos = np.array([1.0, 0.3])   # activating voxel
    shape_neg = -shape_pos             # deactivating voxel (same shape, sign flipped)
    shape_flat = np.array([0.0, 1.0])  # derivative-only: near-zero positive peak
    Y = np.column_stack([
        sum(amps[i] * X[:, i * K:(i + 1) * K] @ s for i in range(N))
        for s in (shape_pos, shape_neg, shape_flat)
    ]) + 0.05 * rng.standard_normal((n, 3))
    conf = np.ones((n, 1))

    fm = estimate_voxel_hrf(Y, X, basis=Bk, confounds=conf, K=K)
    fm_beta = lss_with_voxel_hrf(Y, X, fm, confounds=conf).betas

    # R: same pooled raw fit (aggregate per-basis regression), then R normalization
    A = np.column_stack([X[:, k::K].sum(axis=1) for k in range(K)])
    Q, _ = np.linalg.qr(conf)
    Ac = A - Q @ (Q.T @ A)
    Yc = Y - Q @ (Q.T @ Y)
    raw = np.linalg.lstsq(Ac, Yc, rcond=None)[0]
    r_coef, r_scale, r_degen = r_normalize(raw)
    r_beta = lss_with_voxel_hrf(Y, X, VoxelHrfResult(r_coef, Bk, []), confounds=conf).betas

    print("true mean amplitude per voxel:  +1 (act), -1 (deact), n/a (flat)")
    print("fmrimod coef (unit-L2):\n", np.round(fm.coefficients, 3))
    print("R coef (positive-peak, SPMG1-oriented):\n", np.round(r_coef, 3))
    print("R degenerate flags:", r_degen)
    print("mean trial beta  fmrimod:", np.round(fm_beta.mean(axis=0), 3))
    print("mean trial beta  R HEAD :", np.round(r_beta.mean(axis=0), 3))
    print("corr(beta, true amps) fmrimod:", [round(float(np.corrcoef(fm_beta[:, v], amps)[0, 1]), 3) for v in range(2)])
    print("corr(beta, true amps) R      :", [round(float(np.corrcoef(r_beta[:, v], amps)[0, 1]), 3) for v in range(2)])


if __name__ == "__main__":
    main()
