"""Shared fixture: 2-run event design for fmrireg_a drift checks."""
import numpy as np
import pandas as pd

import fmrimod as fm


def two_run(scale=1.0, seed=1, n=60, V=4, phi=None, degree=1, basis="poly", intercept="runwise"):
    rng = np.random.default_rng(seed)
    tr = 2.0
    on = np.arange(4.0, 100.0, 12.0)
    cond = np.array(["A", "B"] * 10)[: len(on)]
    ev = pd.DataFrame({
        "onset": np.concatenate([on, on]),
        "condition": np.concatenate([cond, cond]),
        "duration": 0.0,
        "run": np.repeat([1, 2], len(on)),
    })
    Y = rng.normal(size=(2 * n, V))
    if phi is not None:
        for r in range(2):
            for v in range(V):
                e = rng.normal(size=n)
                x = np.zeros(n)
                for t in range(n):
                    x[t] = e[t] + (phi * x[t - 1] if t else 0.0)
                Y[r * n:(r + 1) * n, v] = x
    Y = Y + 100.0
    Y = Y * scale
    ds = fm.fmri_dataset(Y, tr=tr, events=ev, run_length=[n, n])
    return ds, Y, ev
