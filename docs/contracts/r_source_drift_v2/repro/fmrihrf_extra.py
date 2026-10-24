"""Extra checks: mexhat block pre-onset leak, SPMG2 column vs Nilearn, normalized block invariance."""
import math

import numpy as np
from nilearn.glm.first_level import spm_hrf, spm_time_derivative
from scipy import integrate

from fmrimod.hrf import HRF_MEXHAT, HRF_SPMG1, HRF_SPMG2, block_hrf, normalize


# mexhat: R hrf_mexhat at HEAD (causal, unit-L2 constant)
def r_mexhat(t, mean=6.0, sd=2.0):
    t = np.asarray(t, float); t0 = t - mean
    v = 2 / (math.sqrt(3 * sd) * math.pi ** 0.25) * (1 - (t0 / sd) ** 2) * np.exp(-t0 ** 2 / (2 * sd ** 2))
    return np.where(t < 0, 0.0, v)
fm = float(np.ravel(block_hrf(HRF_MEXHAT, 4.0, precision=0.001)(np.array([1.0])))[0])
ref, _ = integrate.quad(lambda u: r_mexhat(1.0 - u), 0, 4)
print(f"block_hrf(HRF_MEXHAT, 4)(1): fmrimod={fm:.5f}  R(HEAD, causal+new const)={ref:.5f}")

# SPMG2 column 2 vs Nilearn spm_time_derivative on Nilearn's own grid
tr, ov, tl = 1.0, 50, 32.0
dt = tr / ov
ts = np.linspace(0, tl, int(np.rint(tl / dt)))
nl_h = spm_hrf(tr, oversampling=ov, time_length=tl)
nl_d = spm_time_derivative(tr, oversampling=ov, time_length=tl)
fm_h = HRF_SPMG2(ts - dt)  # nilearn uses loc=dt
c1 = np.corrcoef(fm_h[:, 0], nl_h)[0, 1]
c2 = np.corrcoef(fm_h[:, 1], nl_d)[0, 1]
print(f"fmrimod SPMG2 vs nilearn on nilearn grid: col1 corr={c1:.6f}, col2 corr={c2:.4f}")

# normalized blocked HRF query invariance (R test_block_contract)
h = normalize(block_hrf(HRF_SPMG1, 4.0), "unit_peak_per_basis")
times = np.array([1, 2.3, 7, 12, 25.0])
whole = np.ravel(h(times))
scal = np.array([float(np.ravel(h(np.array([x])))[0]) for x in times])
print(f"normalize(block_hrf(SPMG1,4)) query-invariant: max|scalar-vector|={np.max(np.abs(scal - whole)):.2e}; span={h.span}")
