"""DES: multi-variable covariate() must yield one unconvolved column per
variable (R covariate(x, y) -> cov_x, cov_y; test-covariate.R ncol(dm) == 2).
fmrimod builds a single elementwise-product column."""
import warnings

import numpy as np
import pandas as pd

from fmrimod import event_model
from fmrimod.covariate import CovariateTerm
from fmrimod.sampling import SamplingFrame
from fmrimod.spec import compile_events, covariate, hrf

warnings.simplefilter("ignore")
n = 40
rng = np.random.default_rng(0)
cov = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "z": rng.normal(size=n)})

# typed spec path
sf = SamplingFrame(blocklens=[n], tr=2.0)
ev = pd.DataFrame({"onset": [2., 20.], "cond": ["A", "B"], "run": 1})
for kw in [{}, {"prefix": "motion"}, {"id": "alignment"}]:
    em = compile_events(hrf("cond") + covariate("x", "y", "z", source=cov, **kw), ev, sf,
                        block="run", durations=None, precision=None)
    X = em.design_matrix
    cov_cols = X[:, 2:]
    print(f"typed {kw}: names={em.column_names[2:]} ncols={cov_cols.shape[1]} "
          f"== x*y*z: {np.allclose(cov_cols[:, 0], cov.x * cov.y * cov.z)}")

# legacy CovariateTerm path
data = cov.copy()
data["onset"] = 2.0
data["cond"] = "A"
em2 = event_model([CovariateTerm(["x", "y"], prefix="motion")], data=data, tr=2.0, n_scans=n)
X2 = em2.design_matrix
print(f"legacy: names={em2.column_names} ncols={X2.shape[1]} == x*y: {np.allclose(X2[:, 0], cov.x * cov.y)}")
print("R (6907d0b): names cov_x/cov_y/cov_z (motion_x.., alignment_x..), 3 columns equal to x, y, z")
