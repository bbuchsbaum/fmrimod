"""REG-A: censor semantics vs R (12344ba; test-censor-frame-integration.R).

R: censoring feeds AR estimation/whitening only and never drops volumes from
the regression; under iid it is inert and warns; AROptions censor vector
overrides the frame column; censor='none' opts out.
fmrimod: dataset censor drops rows from OLS (glm/preprocess.py apply_censoring)
and AROptions.censor is never read by fmri_lm.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm
from fmrimod.model.config import AROptions, FmriLmConfig

ds, Y, ev = two_run(seed=42, n=60)
cen = np.zeros(120, bool); cen[[9, 10, 11, 54, 55]] = True
ds_c = fm.fmri_dataset(Y, tr=2.0, events=ev, run_length=[60, 60],
                       censor=[cen[:60], cen[60:]])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    f_iid_c = fm.fmri_lm("hrf(condition)", ds_c)
    print("iid + censor warnings:", [str(x.message)[:60] for x in w if "censor" in str(x.message).lower()])
f_iid = fm.fmri_lm("hrf(condition)", ds)
print("iid: censored vs plain max|dbeta| =", float(np.max(np.abs(f_iid_c.betas - f_iid.betas))),
      "(R: identical + warning)  dfres", f_iid_c.residual_df, "vs", f_iid.residual_df)
a = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1")))
b = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1", censor=[19, 20, 69])))
print("AROptions(censor=[...]) effect on ar1 fit max|dbeta| =", float(np.max(np.abs(a.betas - b.betas))), "(R: differs)")
try:
    c = fm.fmri_lm("hrf(condition)", ds_c, config=FmriLmConfig(ar=AROptions(struct="ar1")))
    print("ar1 + dataset censor: ok, ar_params", np.ravel(c.ar_params), "dfres", c.residual_df)
except Exception as e:
    print("ar1 + dataset censor: FAILS:", type(e).__name__, str(e)[:120])
