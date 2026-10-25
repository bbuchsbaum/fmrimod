"""AR-02: exact stationary start-up (Ansley) for AR(p)/ARMA (0.4.0, 617bd87).

fmrimod whitening.py:300 applies exact start only for p == 1, q == 0.
R oracle: dense-Cholesky GLS within each segment (gls_oracle).
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import gls_oracle, sim_arma  # noqa: E402

from fmrimod.ar import plan_from_phi, whiten_apply  # noqa: E402

rng = np.random.default_rng(9110)

for phi, theta in (([1.2, -0.5], []), ([0.6], [0.5]), ([0.5], [])):
    Y = sim_arma(30, 3000, phi, theta, rng=rng)
    plan = plan_from_phi(np.array(phi), np.array(theta), exact_first=True)
    W = whiten_apply(plan, np.ones((30, 1)), Y).Y
    v = W.var(axis=1)
    # oracle on one column with two segments
    y = rng.standard_normal(70)
    starts = [0, 30, 33]
    ref = np.concatenate([gls_oracle(y[a:b], phi, theta) for a, b in zip(starts, starts[1:] + [70])])
    from fmrimod.ar.whitening import arma_whiten_segments
    out = arma_whiten_segments(y, np.array(phi), np.array(theta), np.array(starts), exact_first_ar1=True)
    print(f"phi={phi} theta={theta}: fmrimod var(t=0..3)={np.round(v[:4], 2)}  "
          f"max|fmrimod - GLS oracle|={np.max(np.abs(out - ref)):.3g}")
