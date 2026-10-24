"""REG-A: hrf_smoothing_kernel normalisation (R review C6, 7c2a985;
test-review-C-kernel-instcor.R).  fmrimod/glm/compat.py:639-647 keeps the
pre-fix R code: K / diag(K) row scaling and column-wise 'cosine'."""
import warnings

import numpy as np

from fmrimod.glm import hrf_smoothing_kernel

warnings.simplefilter("ignore")
for method in ("gram", "cosine"):
    for buf in (0, 3):
        K = hrf_smoothing_kernel(30, tr=1.5, form="onset ~ trialwise(basis='gaussian')",
                                 method=method, buffer_scans=buf, normalize=True)
        print(f"{method:6s} buf={buf}: finite={np.all(np.isfinite(K))} "
              f"max|K-K'|={np.max(np.abs(K - K.T)):.3g} diag range=({np.min(np.diag(K)):.3g},{np.max(np.diag(K)):.3g}) "
              f"max|K|={np.max(np.abs(K)):.3g}")
