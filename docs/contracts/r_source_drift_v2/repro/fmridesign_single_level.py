"""DES: single-level factor crossed with other variables (R d37175b)."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.sampling import SamplingFrame
from fmrimod.spec import compile_events, hrf

warnings.simplefilter("ignore")
sf = SamplingFrame(blocklens=[40], tr=1.0)

des = pd.DataFrame({"onset": [1.0, 10.0, 20.0, 30.0], "run": 1,
                    "cond": ["A", "B", "C", "A"], "task": "go"})
for label, build in [
    ("string hrf(cond:task)", lambda: event_model("onset ~ hrf(cond:task)", data=des, sampling_frame=sf, block="run")),
    ("spec hrf('cond','task')", lambda: compile_events(hrf("cond", "task").__add__(hrf("cond")) if False else __import__('fmrimod.spec', fromlist=['as_spec']).as_spec(hrf("cond", "task")), des, sf, block="run", durations=None, precision=None)),
]:
    try:
        m = build()
        print(label, "->", m.design_matrix.shape[1], "cols", m.column_names)
    except Exception as e:
        print(label, "ERR", type(e).__name__, e)

des2 = pd.DataFrame({"onset": [1.0, 10.0, 20.0], "run": 1, "cond": ["A", "A", "A"], "rt": [0.5, 1.0, 2.0]})
for label, build in [
    ("string hrf(cond:rt)", lambda: event_model("onset ~ hrf(cond:rt)", data=des2, sampling_frame=sf, block="run")),
    ("spec hrf('cond','rt')", lambda: compile_events(__import__('fmrimod.spec', fromlist=['as_spec']).as_spec(hrf("cond", "rt")), des2, sf, block="run", durations=None, precision=None)),
]:
    try:
        m = build()
        X = m.design_matrix
        # R (fixed): column is rt-modulated -> ratio of peak amplitudes after each onset reflects rt (0.5,1,2)
        # compare with an all-ones (unmodulated) regressor
        m1 = event_model("onset ~ hrf(cond)", data=des2, sampling_frame=sf, block="run")
        print(label, "->", X.shape[1], "cols", m.column_names, "equals unmodulated:", np.allclose(X[:, 0], m1.design_matrix[:, 0]))
    except Exception as e:
        print(label, "ERR", type(e).__name__, e)
