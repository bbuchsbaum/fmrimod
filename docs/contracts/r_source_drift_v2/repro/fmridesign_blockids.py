"""Cross-audit lead: block ids canonicalised by first appearance; a run with no
events shifts later runs. R canonicalize_blockids() does the same
(match(x, unique(x)), R/event_model_helpers.R:60-73) and additionally rejects
decreasing block ids."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.sampling import SamplingFrame

warnings.simplefilter("ignore")
sf = SamplingFrame(blocklens=[20, 20, 20], tr=1.0)
df = pd.DataFrame({"onset": [2., 2.], "cond": ["A", "A"], "run": [1, 3]})
em = event_model("onset ~ hrf(cond)", data=df, sampling_frame=sf, block="run")
X = em.design_matrix[:, 0]
live = [bool(np.any(np.abs(X[i * 20:(i + 1) * 20]) > 1e-6)) for i in range(3)]
print("run labels [1, 3] -> blockids", em.blockids.tolist(), "| runs with signal:", live)
df2 = pd.DataFrame({"onset": [2., 2., 5.], "cond": ["A"] * 3, "run": [2, 1, 2]})
em2 = event_model("onset ~ hrf(cond)", data=df2, sampling_frame=sf, block="run")
print("run labels [2, 1, 2] accepted -> blockids", em2.blockids.tolist(), "| R: error 'blockids must be non-decreasing'")
