"""DES: hrf(cond, basis=<plain function of t>) - R wraps it with gen_hrf()
(453c62a names it after the argument); fmrimod silently substitutes SPM canonical."""
import warnings

import numpy as np
import pandas as pd

from fmrimod.design_colmap import design_colmap
from fmrimod.sampling import SamplingFrame
from fmrimod.spec import as_spec, compile_events, hrf

warnings.simplefilter("always")
sf = SamplingFrame(blocklens=[60], tr=1.0)
ev = pd.DataFrame({"onset": [5.0], "cond": ["A"], "run": 1})

def my_box(t):
    t = np.asarray(t, float)
    return ((t >= 0) & (t < 4)).astype(float)

with warnings.catch_warnings(record=True) as rec:
    em_fun = compile_events(as_spec(hrf("cond", basis=my_box)), ev, sf, block="run", durations=None, precision=None)
    X_fun = em_fun.design_matrix[:, 0]
em_spm = compile_events(as_spec(hrf("cond")), ev, sf, block="run", durations=None, precision=None)
X_spm = em_spm.design_matrix[:, 0]
print("plain-function basis == SPM canonical design:", np.allclose(X_fun, X_spm))
print("nonzero samples (boxcar would be ~4 s after onset):", np.nonzero(np.abs(X_fun) > 1e-6)[0][:12])
print("warnings:", [str(r.message)[:100] for r in rec])
print("colmap basis_name:", design_colmap(em_fun).basis_name.tolist())
