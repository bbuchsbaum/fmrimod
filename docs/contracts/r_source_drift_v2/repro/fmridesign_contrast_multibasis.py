"""DES: contrast weights on multi-basis terms (R 4415504/fd6dbb2 #23/#25).

R (fixed): unit/pair/formula contrasts on an hrf(cond, basis="spmg3") term are
expanded across basis functions and matched to design columns BY NAME
(condition-major term_cond_b##). fmrimod embeds term-local weights positionally.
"""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.contrast import (
    contrast_set,
    contrast_weights,
    pair_contrast,
    unit_contrast,
)
from fmrimod.contrast.contrast_spec import Formula
from fmrimod.formula.base import Term
from fmrimod.glm.fmri_lm import _FitColumnTerm

warnings.simplefilter("always")
df = pd.DataFrame({"onset": [2.0, 20.0, 40.0, 60.0], "cond": ["A", "B", "A", "B"]})

def show(title, names, w):
    w = np.asarray(w).ravel()
    print(title)
    for n, v in zip(names, w):
        if abs(v) > 1e-12:
            print(f"    {n:22s} {v:+.3f}")

# --- term-attached path: contrast_weights(EventModel) ---
term = Term("cond", hrf="spmg3")
term.contrast_specs = contrast_set(
    pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name="A_vs_B"),
    unit_contrast(Formula("cond == 'B'"), name="just_B"),
)
em = event_model([term], data=df, tr=2.0, n_scans=50)
print("columns:", em.column_names)
with warnings.catch_warnings(record=True) as rec:
    res = contrast_weights(em)
    for r in rec:
        print("  warning:", str(r.message)[:120])
for cname, con in res.get("cond", {}).items():
    show(f"[EventModel path] {cname}", em.column_names, con.offset_weights[:, 0])

# R expectation (fixed 4415504/fd6dbb2): A_vs_B -> +0.5? no: R pair_contrast on a
# multi-basis term without basis= gives a (6 x 1) vector with +1/-1 on EVERY
# basis column of A / B? -- see R test-multibasis-column-order.R; at minimum the
# weight must sit on cond.A_b0k / cond.B_b0k columns, never on A_b02 for "just_B".

# --- fit-level path: contrast_weights(spec, _FitColumnTerm(columns)) ---
fit_term = _FitColumnTerm(list(em.column_names))
for spec in [pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name="A_vs_B"),
             unit_contrast(Formula("cond == 'B'"), name="just_B")]:
    with warnings.catch_warnings(record=True) as rec:
        try:
            con = contrast_weights(spec, fit_term)
            show(f"[fit path] {spec.name}", em.column_names, con.weights[:, 0])
        except Exception as e:
            print(f"[fit path] {spec.name} ERR {type(e).__name__}: {e}")
        for r in rec:
            print("  warning:", str(r.message)[:120])
