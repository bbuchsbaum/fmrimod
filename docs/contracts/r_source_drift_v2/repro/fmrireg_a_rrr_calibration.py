"""REG-A: rrr engine null calibration (R E1 f98a71a; test-review-E-rrr.R)."""
import inspect
import warnings

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.glm import ReducedRankEngineOptions

warnings.simplefilter("ignore")
print(inspect.signature(ReducedRankEngineOptions))
rng = np.random.default_rng(11)
ons = np.arange(10, 381, 15.0)
ev = pd.DataFrame({"onset": ons, "condition": np.array(["a", "b", "c"] * 30)[: len(ons)],
                   "duration": 0.0, "run": 1})
Y = rng.normal(size=(200, 2000))
ds = fm.fmri_dataset(Y, tr=2.0, events=ev, run_length=[200])
ols = fm.fmri_lm("hrf(condition)", ds)
for r in (1, 2, None):
    f = fm.fmri_lm("hrf(condition)", ds, engine=ReducedRankEngineOptions(rank=r))
    t = f.tstat()[:3]
    print(f"rank={r}: |t|>1.96 rate={np.mean(np.abs(t) > 1.96):.3f} median|t|={np.median(np.abs(t)):.3f}",
          "| max|se-se_ols| full rank:" if r is None else "",
          float(np.max(np.abs(f.se()[:3] - ols.se()[:3]))) if r is None else "")
