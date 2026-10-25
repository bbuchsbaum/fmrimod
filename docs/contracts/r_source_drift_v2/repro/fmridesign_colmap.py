"""DES: design_colmap basis_name/basis_label (R 0d53fb2/453c62a) and baseline
is_block_diagonal / role / run (R 1ecaf91, fa40860)."""
import warnings

import numpy as np
import pandas as pd

from fmrimod.baseline import baseline_model
from fmrimod.design_colmap import design_colmap
from fmrimod.sampling import SamplingFrame
from fmrimod.spec import as_spec, compile_events, hrf

warnings.simplefilter("ignore")
sf = SamplingFrame(blocklens=[40, 40], tr=2.0)
ev = pd.DataFrame({"onset": [2., 20., 4., 30.], "cond": ["A", "B", "A", "B"], "run": [1, 1, 2, 2]})

def my_fun(t):
    return np.exp(-((np.asarray(t) - 5.0) ** 2) / 4.0)

for label, term in [("spmg2", hrf("cond", basis="spmg2")),
                    ("spmg3", hrf("cond", basis="spmg3")),
                    ("spmg3 lag=2", hrf("cond", basis="spmg3", lag=2.0)),
                    ("fir nbasis=3", hrf("cond", basis="fir", nbasis=3)),
                    ("fourier nbasis=3", hrf("cond", basis="fourier", nbasis=3)),
                    ("spmg1", hrf("cond"))]:
    try:
        em = compile_events(as_spec(term), ev, sf, block="run", durations=None, precision=None)
        cm = design_colmap(em)
        print(f"{label:18s} basis_name={sorted(set(map(str, cm.basis_name)))} labels={list(cm.basis_label[:3])}")
    except Exception as e:
        print(label, "ERR", type(e).__name__, e)

# baseline: global constant drift + nuisance
nuis = [pd.DataFrame({"trans_x": np.random.randn(40), "rot_y": np.random.randn(40)}) for _ in range(2)]
for kw in [dict(basis="constant", degree=1, intercept="global"),
           dict(basis="poly", degree=1, intercept="global"),
           dict(basis="poly", degree=1, intercept="runwise", nuisance_list=nuis)]:
    try:
        bm = baseline_model(sframe=sf, **kw) if "sframe" in baseline_model.__code__.co_varnames else baseline_model(sampling_frame=sf, **kw)
        cm = design_colmap(bm)
        print(kw.get("basis"), kw.get("intercept"), "nuis" if "nuisance_list" in kw else "")
        print(cm[["name", "role", "run", "is_block_diagonal", "basis_label"]].to_string(index=False))
    except Exception as e:
        print(kw, "ERR", type(e).__name__, e)
