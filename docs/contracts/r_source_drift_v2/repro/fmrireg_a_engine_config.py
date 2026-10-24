"""REG-A: engines that ignore FmriLmConfig (R caab78d/43028b0 'reject unsupported
AR and robust plugin fits'; 505a32c rrr_gls resolves censoring; 60f8430
'unsupported weighted backends fail instead of fitting unweighted')."""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm
from fmrimod.glm import ReducedRankEngineOptions, SketchEngineOptions
from fmrimod.model.config import (
    AROptions,
    FmriLmConfig,
    RobustOptions,
    VolumeWeightOptions,
)

warnings.simplefilter("ignore")
ds, Y, ev = two_run(seed=5, phi=0.4, n=100)
w = np.ones(200); w[[3, 4, 5, 150]] = 0.0
cfgs = {
    "iid": FmriLmConfig(),
    "ar1": FmriLmConfig(ar=AROptions(struct="ar1")),
    "huber": FmriLmConfig(robust=RobustOptions(type="huber")),
    "vol-weights": FmriLmConfig(volume_weights=VolumeWeightOptions(enabled=True, weights=w)),
}
for ename, eng in [("rrr rank=1", ReducedRankEngineOptions(rank=1)),
                   ("sketch m=60", SketchEngineOptions(sketch_size=60, seed=1))]:
    base = None
    for cname, cfg in cfgs.items():
        try:
            f = fm.fmri_lm("hrf(condition)", ds, config=cfg, engine=eng)
            if base is None:
                base = f.betas
            print(f"{ename:12s} {cname:12s}: ok  max|dbeta vs iid|={np.max(np.abs(f.betas - base)):.3g}")
        except Exception as e:
            print(f"{ename:12s} {cname:12s}: {type(e).__name__}: {str(e)[:90]}")
