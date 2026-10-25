"""GDS: lmm:ri `converged` semantics (7c7ecae; test-review-numeric.R:223-235).

Fixed R: converged = 0 when the variance ratio is pinned at the upper search bound
(no within-subject noise), 1 otherwise.  Fixture mirrors the R test.
"""
import warnings

import numpy as np

from fmrimod.group import SampleLabelSpace, group_dataset, lmm_ri

warnings.simplefilter("ignore")
rng = np.random.default_rng(82)
subjects = [f"s{i:02d}" for i in range(10)]
u = rng.normal(scale=5, size=10)
beta = np.empty((2, 10, 3))
beta[0] = u[:, None] + rng.normal(scale=1e-9, size=(10, 3))
beta[1] = u[:, None] + rng.normal(scale=1.0, size=(10, 3))
ds = group_dataset({"beta": beta}, space=SampleLabelSpace(["a", "b"]), subjects=subjects, contrasts=["c1", "c2", "c3"])
try:
    out = lmm_ri(ds, formula="~ 1", theta_mode="voxelwise")
    print("fmrimod converged:", out.assay("converged").ravel(), " lambda:", out.assay("lambda").ravel(),
          " failed:", out.metadata.get("fit_failed_feature_indices"))
except Exception as exc:  # noqa: BLE001
    print(type(exc).__name__, exc)
print("R expected converged: [0, 1]")
