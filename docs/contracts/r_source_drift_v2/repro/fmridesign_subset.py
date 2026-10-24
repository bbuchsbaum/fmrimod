"""DES: hrf(subset=) edge cases vs R dbc2ad1 (#30) and v1 'zero-event subset' row,
plus contrast rows for subset-dropped levels (R 1edac0b #17)."""
import warnings

import numpy as np
import pandas as pd

from fmrimod.sampling import SamplingFrame
from fmrimod.spec import as_spec, compile_events, hrf

sf = SamplingFrame(blocklens=[30], tr=2.0)

def build(df, **kw):
    return compile_events(as_spec(hrf("condition", **kw)), df, sf, block="run", durations=None, precision=None)

ev = pd.DataFrame({"onset": [0., 10., 20.], "condition": ["A", "B", "C"],
                   "keep": [False, True, False], "run": 1})
for label, df, kw in [
    ("single-event subset (#30)", ev, {"subset": "keep"}),
    ("multi-level subset", ev.assign(keep=[True, True, False]), {"subset": "keep"}),
    ("zero-event subset (v1 S2 doc)", ev.assign(keep=False), {"subset": "keep"}),
]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            em = build(df, **kw)
            X = em.design_matrix
            print(f"{label}: cols={em.column_names} nonzero={[bool(np.any(X[:, j])) for j in range(X.shape[1])]}")
        except Exception as e:
            print(f"{label}: ERR {type(e).__name__}: {e}")
        for r in rec:
            print("   warning:", type(r.message).__name__, str(r.message)[:100])
