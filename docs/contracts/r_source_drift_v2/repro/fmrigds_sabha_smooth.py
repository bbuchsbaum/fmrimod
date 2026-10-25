"""GDS: fdr:sabha pi0 smoothing / min_pi0 placement (85c8d7c; src/spatial_fdr_sabha.cpp).

Fixed R: pi0_raw in [0, 1] (no min); smoothed = (pi0_g + lambda * mean_nb) / (1 + lambda);
then clamp to [min_pi0, 1].  fmrimod.stats.spatial_fdr: pi0_raw clamped to [min_pi0, 1]
first, then (1 - lam) * pi0_g + lam * mean_nb.  The weight-normalisation and NaN
divergences are covered by the fmrireg audit (fmrireg_b_spatial_fdr.py).
"""
import numpy as np

from fmrimod.stats.spatial_fdr import estimate_pi0_grouped, smooth_pi0


def r_smooth(pi0, nbrs, lam, min_pi0=0.05):
    out = pi0.copy()
    for g, nb in enumerate(nbrs):
        vals = [pi0[j] for j in nb if j != g and np.isfinite(pi0[j])]
        if vals:
            out[g] = (pi0[g] + lam * np.mean(vals)) / (1 + lam)
    return np.clip(out, min_pi0, 1.0)


nbrs = [[1], [0, 2], [1]]
pi0 = np.array([0.0, 0.6, 1.0])  # group 0 is all-signal -> raw pi0 = 0
for lam in (0.5, 1.0, 2.0):
    py = smooth_pi0(np.clip(pi0, 0.05, 1.0), nbrs, lam=lam)
    print(f"lambda={lam}: fmrimod {np.round(py, 4)}   R {np.round(r_smooth(pi0, nbrs, lam), 4)}")

p = np.r_[np.full(10, 1e-4), np.random.default_rng(0).uniform(size=20)]
gid = np.repeat([0, 1, 2], 10).astype(np.intp)
print("raw pi0 fmrimod (clamped before smoothing):", estimate_pi0_grouped(p, gid, 3).round(3))
