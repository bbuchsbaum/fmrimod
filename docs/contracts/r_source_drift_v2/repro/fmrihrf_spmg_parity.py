"""How close is fmrimod's default spmg1 to R HEAD HRF_SPMG1 (A1=1/120, span 24) at the kernel and regressor level?"""
import numpy as np
from scipy import stats

from fmrimod.hrf import HRF_SPMG1, as_hrf
from fmrimod.regressor import regressor

r_new = as_hrf(lambda t: np.where(np.asarray(t) < 0, 0.0, stats.gamma.pdf(np.asarray(t, float), 6) - stats.gamma.pdf(np.asarray(t, float), 16) / 6),
               span=24.0, name="R_HRF_SPMG1_HEAD")
t = np.linspace(0, 24, 100)
k = np.max(np.abs(HRF_SPMG1(t) - r_new(t)))
print(f"kernel on R test grid linspace(0,24,100): max|diff|={k:.2e} rel-to-peak={k / 0.1754:.2e}")
on = np.array([10, 30, 50.0]); grid = np.arange(0, 80, 0.5)
a = regressor(on, hrf=HRF_SPMG1).evaluate(grid)
b = regressor(on, hrf=r_new).evaluate(grid)
print(f"regressor (fmrimod span 32 vs R span 24): max|diff|={np.max(np.abs(a - b)):.2e} rel={np.max(np.abs(a - b)) / np.max(np.abs(b)):.2e}")
c = regressor(on, hrf=HRF_SPMG1, span=24.0).evaluate(grid)
print(f"regressor with span=24 override: max|diff|={np.max(np.abs(c - b)):.2e}")
