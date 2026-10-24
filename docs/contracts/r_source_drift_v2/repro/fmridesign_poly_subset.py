"""DES: hrf(Poly(x, 2), subset=) - R subsets rows of the basis fitted on the full
data (sub_basis.Poly keeps coefs, 984f9bd/d3dd33d); check fmrimod amplitudes and
the single-event shape (R keeps a 1 x 2 matrix)."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.basis import Poly
from fmrimod.formula.base import Term

warnings.simplefilter("ignore")
n = 6
d = pd.DataFrame({"onset": np.arange(n) * 30.0 + 2, "x": np.array([0.1, 0.5, 0.9, 1.7, 2.0, 3.5])})

def amplitudes(X, onsets):
    """Per-event amplitude of each column (events spaced 30 s apart)."""
    base = []
    for o in onsets:
        e = event_model([Term("c", hrf="spmg1")], data=pd.DataFrame({"onset": [o], "c": ["a"]}), tr=1.0, n_scans=200)
        base.append(e.design_matrix[:, 0])
    B = np.column_stack(base)
    return np.linalg.lstsq(B, X, rcond=None)[0]

def r_poly(x, degree):
    xc = x - x.mean()
    Q, R = np.linalg.qr(np.vander(xc, degree + 1, increasing=True))
    return (Q * np.sign(np.diag(R)))[:, 1:]

full_basis = r_poly(d.x.to_numpy(), 2)          # R: Poly fitted on all events
keep = np.array([False, True, False, True, True, False])
t = Term("x", hrf="spmg1"); t.basis = Poly(degree=2, intercept=False); t._kwargs["subset"] = "keep"
em = event_model([t], data=d.assign(keep=keep), tr=1.0, n_scans=200); print("X nonzero cols:", np.abs(em.design_matrix).sum(0))
A = amplitudes(em.design_matrix, d.onset[keep].to_numpy())
print("fmrimod subset amplitudes:\n", np.round(A, 4))
print("R (rows of full-data Poly):\n", np.round(full_basis[keep], 4))
print("R if refit on subset:\n", np.round(r_poly(d.x[keep].to_numpy(), 2), 4))
