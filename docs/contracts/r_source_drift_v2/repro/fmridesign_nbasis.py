"""DES: FIR nbasis, default nbasis, trialwise nbasis/lag handling."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model

warnings.simplefilter("ignore")
df = pd.DataFrame({"onset": [10.0, 40.0, 70.0, 100.0], "cond": ["a", "b", "a", "b"], "duration": 0.0})

for f in ["onset ~ hrf(cond, basis='fir', nbasis=6)",
          "onset ~ hrf(cond, basis='fir')",
          "onset ~ hrf(cond, basis='bspline')",
          "onset ~ hrf(cond, basis='fourier')",
          "onset ~ hrf(cond, basis='tent')",
          "onset ~ trialwise(basis='fir', nbasis=6)",
          "onset ~ trialwise(basis='fir')",
          ]:
    try:
        m = event_model(f, data=df, tr=2.0, n_scans=80)
        X = m.design_matrix
        print(f, "->", X.shape, m.column_names[:3])
    except Exception as e:
        print(f, "ERR", type(e).__name__, e)

# FIR nbasis=6 bin indicator check via design: impulse at onset=10, TR=2 -> sample t=20 is 10s post onset
m = event_model("onset ~ hrf(cond, basis='fir', nbasis=6)", data=df.iloc[:1].assign(cond="a"), tr=2.0, n_scans=80)
X = m.design_matrix
print("FIR6 row at t=20s (10s post-onset):", np.round(X[10], 3))

# trialwise lag
m0 = event_model("onset ~ trialwise()", data=df, tr=2.0, n_scans=80)
m1 = event_model("onset ~ trialwise(lag=4)", data=df, tr=2.0, n_scans=80)
print("trialwise lag=4 changes design?", not np.allclose(m0.design_matrix, m1.design_matrix))
