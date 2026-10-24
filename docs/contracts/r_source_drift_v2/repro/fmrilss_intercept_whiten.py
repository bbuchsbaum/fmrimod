"""Side check: include_intercept=True adds an UNWHITENED intercept after
prewhitening (fmrimod/single/__init__.py:141-154 then lss.py:362-367),
whereas R lss() whitens Z (default intercept) with the same plan. Compare
with baseline_regressors=ones, which fmrimod does whiten."""
import numpy as np

from fmrimod.single import estimate_single_trial
from fmrimod.single._prewhiten import PrewhitenConfig

rng = np.random.default_rng(1)
n, T, V = 200, 12, 4
X = np.zeros((n, T))
h = np.exp(-0.5 * ((np.arange(16) - 5) / 2.0) ** 2)
for j, o in enumerate(np.linspace(5, 180, T).astype(int)):
    X[o:o + 16, j] = h[: max(0, min(16, n - o))]
e = rng.standard_normal((n, V))
for t in range(1, n):
    e[t] += 0.5 * e[t - 1]
Y = 100 + X @ rng.normal(1, 0.3, (T, V)) + e
cfg = PrewhitenConfig(p=1, runs=np.repeat([1, 2], n // 2))
a = estimate_single_trial(Y, X, prewhiten=cfg, include_intercept=True).betas
b = estimate_single_trial(Y, X, prewhiten=cfg, baseline_regressors=np.ones((n, 1))).betas
print("max |include_intercept - whitened intercept| =", float(np.max(np.abs(a - b))))
