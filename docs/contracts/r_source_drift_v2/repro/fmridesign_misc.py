"""DES misc checks. Each section prints a tag used in fmridesign.md."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.basis import Poly
from fmrimod.contrast import (
    apply_basis_filter,
    contrast_set,
    contrast_weights,
    one_against_all_contrast,
    pair_contrast,
    unit_contrast,
)
from fmrimod.contrast.contrast_spec import Formula
from fmrimod.formula.base import Term
from fmrimod.sampling import SamplingFrame
from fmrimod.spec import as_spec, compile_events, hrf
from fmrimod.utils.generics import condition_map
from fmrimod.validate import validate_contrasts

warnings.simplefilter("ignore")

print("[DES-03 where] ---------------------------------------------")
df = pd.DataFrame({"onset": np.arange(8) * 6. + 2, "cond": ["A", "B"] * 4, "task": ["x", "x", "y", "y"] * 2})
em = event_model("onset ~ hrf(cond:task)", data=df, tr=2., n_scans=40)
et = em._create_event_terms()[0]
c1 = contrast_weights(pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"),
                                    where=Formula("task == 'x'"), name="AvB_x"), et)
print("cells", et.conditions(drop_empty=False, expand_basis=False))
print("where=task=='x' weights", c1.weights.ravel(), "| R (dfe11ea semantics): [1, 0, -1, 0]")

print("[DES-06 hrf(a, b)] ----------------------------------------")
em = event_model("onset ~ hrf(cond, task)", data=df, tr=2., n_scans=40)
print("hrf(cond, task) columns:", em.column_names, "| R: 4 crossed cells")
dft = df.assign(Time=np.arange(8, dtype=float) + 1.0)
print("hrf(Time, cond) columns:", event_model("onset ~ hrf(Time, cond)", data=dft, tr=2., n_scans=40).column_names,
      "| hrf(Time:cond):", event_model("onset ~ hrf(Time:cond)", data=dft, tr=2., n_scans=40).column_names)

print("[DES-09 basis_weights] -------------------------------------")
names = ["c.A_b01", "c.A_b02", "c.A_b03", "c.B_b01", "c.B_b02", "c.B_b03"]
W = np.array([[1, 1, 1, -1, -1, -1]], float).T
out = apply_basis_filter(W, names, basis_spec=None, nbasis=3, contrast_name="x", basis_weights=np.array([1., 0, 0]))
ref = apply_basis_filter(W, names, basis_spec=[1], nbasis=3, contrast_name="x")
print("basis=None,bw=[1,0,0]:", out["weights"].ravel(), "| basis=[1]:", ref["weights"].ravel())
out2 = apply_basis_filter(W, names, basis_spec=None, nbasis=3, contrast_name="x", basis_weights=np.array([1., 0]))
print("length-2 bw on nbasis=3 (R: error):", out2["weights"].ravel())

print("[DES-13 condition_map] -------------------------------------")
d2 = pd.DataFrame({"onset": [2., 20., 40., 60.], "cond": ["A", "B", "A", "B"]})
em = event_model("onset ~ hrf(cond, basis='spmg3')", data=d2, tr=2., n_scans=50)
print(condition_map(em, expand_basis=True).to_string(index=False))

print("[DES-15 Poly single-event subset] ---------------------------")
d3 = pd.DataFrame({"onset": np.arange(0, 100, 10.), "x": np.linspace(-1, 1, 10),
                   "keep": [False] * 4 + [True] + [False] * 5})
t = Term("x", hrf="spmg1"); t.basis = Poly(degree=2, intercept=False); t._kwargs["subset"] = "keep"
em = event_model([t], data=d3, tr=2., n_scans=60)
print("design shape", em.design_matrix.shape, "names", em.column_names)

print("[DES-19 nested contrast_set] -------------------------------")
try:
    contrast_set(pair_contrast(Formula("A"), Formula("B"), name="AvB"), one_against_all_contrast(["A", "B", "C"], "cond"))
    print("nested accepted")
except Exception as e:
    print("ERR", type(e).__name__, e)

print("[DES-20 validate_contrasts] --------------------------------")
t = Term("cond"); t.contrast_specs = contrast_set(pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name="AvB"))
em = event_model([t], data=d2, tr=2., n_scans=50)
print("attached contrasts:", {k: list(v) for k, v in contrast_weights(em).items()})
print("validate_contrasts(em) rows:", len(validate_contrasts(em)))
try:
    validate_contrasts(np.eye(3), weights={})
    print("empty dict -> frame")
except Exception as e:
    print("empty dict ERR", type(e).__name__, e)

print("[DES-21 basis names] ---------------------------------------")
for b in ["bs", "tent"]:
    try:
        m = event_model(f"onset ~ hrf(cond, basis='{b}')", data=d2, tr=2., n_scans=50)
        print(b, "ok", m.design_matrix.shape)
    except Exception as e:
        print(b, "ERR", type(e).__name__, e)

print("[DES-28 summate / negative onset] --------------------------")
sf = SamplingFrame(blocklens=[60], tr=2.0)
ev = pd.DataFrame({"onset": [2., 40.], "cond": ["A", "A"], "run": 1, "duration": [10., 10.]})
X1 = compile_events(as_spec(hrf("cond", durations="duration")), ev, sf, block="run", durations=None, precision=None).design_matrix
X0 = compile_events(as_spec(hrf("cond", durations="duration", summate=False)), ev, sf, block="run", durations=None, precision=None).design_matrix
print("summate=False differs from True:", not np.allclose(X0, X1))
try:
    compile_events(as_spec(hrf("cond")), ev.assign(onset=[-1., 40.]), sf, block="run", durations=None, precision=None).design_matrix
    print("negative onset accepted")
except Exception as e:
    print("negative onset ERR", type(e).__name__, e)

print("[DES-29 term-local factor x modulator / unit selector] ------")
d4 = pd.DataFrame({"onset": [2., 20., 40., 60., 80., 100.], "cond": ["A", "B", "C"] * 2, "rt": [1., 2., 3., 4., 5., 6.]})
et = event_model("onset ~ hrf(cond:rt)", data=d4, tr=2., n_scans=70)._create_event_terms()[0]
print("cond x rt conditions", et.conditions(drop_empty=False, expand_basis=False))
print("pair A-B", contrast_weights(pair_contrast(Formula("cond == 'A'"), Formula("cond == 'B'"), name="p"), et).weights.ravel())
et2 = event_model("onset ~ hrf(cond)", data=d4, tr=2., n_scans=70)._create_event_terms()[0]
print("unit cond=='A'", contrast_weights(unit_contrast(Formula("cond == 'A'"), name="u"), et2).weights.ravel(),
      "| unit ~cond", contrast_weights(unit_contrast(Formula("cond"), name="u2"), et2).weights.ravel())
