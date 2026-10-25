"""DES: Fcontrasts(EventModel) on multi-basis / continuous / single-level terms
(R 4415504 #25 kronecker(C, I_nbasis); fd6dbb2 skip continuous; f63135b single-level)."""
import warnings

import pandas as pd

from fmrimod import event_model
from fmrimod.contrast import Fcontrasts

warnings.simplefilter("ignore")
df = pd.DataFrame({"onset": [2., 20., 40., 60., 80., 100.],
                   "cond": ["A", "B", "C", "A", "B", "C"],
                   "task": ["x", "y", "x", "y", "x", "y"],
                   "go": "go",
                   "rt": [1., 2., 3., 4., 5., 6.]})
for f in ["onset ~ hrf(cond)",
          "onset ~ hrf(cond, basis='spmg3')",
          "onset ~ hrf(cond:task)",
          "onset ~ hrf(cond:go)",
          "onset ~ hrf(rt) + hrf(cond)",
          "onset ~ hrf(cond, basis='fir', nbasis=4)"]:
    em = event_model(f, data=df, tr=2.0, n_scans=70)
    _ = em.design_matrix
    try:
        F = Fcontrasts(em)
        print(f, "| ncol", em.design_matrix.shape[1], "| F:", {k: v.shape for k, v in F.items()})
    except Exception as e:
        print(f, "ERR", type(e).__name__, e)

# R fixed: hrf(cond, basis='spmg3') -> 'cond' F = kronecker(C_sum(3), I_3): 9 x 6
