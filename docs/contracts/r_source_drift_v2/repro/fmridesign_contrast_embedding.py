"""DES: positional embedding of term-local contrast weights in
fmrimod.contrast.contrast_weights(EventModel) vs R's name-matched embedding
(R a0081c7 de-duplicated term names; 2368564/4415504 empty cells & multi-basis)."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.contrast import contrast_set, contrast_weights, pair_contrast
from fmrimod.contrast.contrast_spec import Formula
from fmrimod.formula.base import Term

warnings.simplefilter("ignore")

def nz(names, w):
    w = np.asarray(w).ravel()
    return {n: round(float(v), 3) for n, v in zip(names, w) if abs(v) > 1e-12}

# 1) Interaction with an empty cell: (B, y) never occurs -> design drops it.
df = pd.DataFrame({"onset": [2., 20., 40., 60., 80.],
                   "cond": ["A", "B", "A", "C", "C"],
                   "task": ["x", "x", "y", "x", "y"]})
t = Term(["cond", "task"])
t.contrast_specs = contrast_set(
    pair_contrast(Formula("cond == 'C'"), Formula("cond == 'A'"), name="C_vs_A"))
em = event_model([t], data=df, tr=2.0, n_scans=60)
print("empty-cell columns:", em.column_names)
res = contrast_weights(em)
for k, con in next(iter(res.values())).items():
    print("  ", k, nz(em.column_names, con.offset_weights))
print("   (R fixed: +0.5 on cond.C_task.x & cond.C_task.y, -0.5 on cond.A_task.x & cond.A_task.y)")

# 2) Two terms sharing a name: second carries the contrast.
df2 = pd.DataFrame({"onset": [2., 20., 40., 60.], "cond": ["A", "B", "A", "B"]})
t1 = Term("cond", hrf="spmg1")
t2 = Term("cond", hrf="spmg2")
t2.contrast_specs = contrast_set(pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name="AvB"))
em2 = event_model([t1, t2], data=df2, tr=2.0, n_scans=60)
print("dup-name terms:", [t.name for t in em2.terms], em2.column_names)
res2 = contrast_weights(em2)
for tn, d in res2.items():
    for k, con in d.items():
        print("  ", tn, k, nz(em2.column_names, con.offset_weights))
