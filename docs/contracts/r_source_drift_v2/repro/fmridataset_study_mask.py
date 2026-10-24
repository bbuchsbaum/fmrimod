"""DS check: StudyDataset lowering vs R e998bd4 (study backend columns routed through each subject's mask).

R (last legacy revision 3ae565e): under strict='intersect' every subject's columns are translated
from combined-mask positions to that subject's own mask; masks over different grids are refused;
under 'identical' non-identical masks are refused. fmrimod's StudyDataset.to_group_dataset stacks
per-subject estimates by position and only checks that lengths agree; mask_strategy is a label.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

import fmrimod.dataset as fds

grid = 6
m1 = np.array([1, 1, 1, 1, 0, 0], bool)   # subject 1 voxels 0..3
m2 = np.array([0, 0, 1, 1, 1, 1], bool)   # subject 2 voxels 2..5 (same count, 50% overlap)
rng = np.random.default_rng(0)
d1 = fds.matrix_dataset(rng.normal(size=(10, grid)), tr=2.0, mask=m1)
d2 = fds.matrix_dataset(rng.normal(size=(10, grid)), tr=2.0, mask=m2)
# Per-subject estimate encodes the voxel id it belongs to.
est1 = np.flatnonzero(m1).astype(float)
est2 = np.flatnonzero(m2).astype(float) + 1000
for strategy in ("subject_specific", "intersect"):
    study = fds.study_dataset(pd.DataFrame({"subject_id": ["s1", "s2"], "dataset": [d1, d2]}),
                              mask_strategy=strategy)
    g = study.to_group_dataset({"s1": SimpleNamespace(estimate=est1, name="c"),
                                "s2": SimpleNamespace(estimate=est2, name="c")})
    beta = g.assay("beta")[:, :, 0]
    print(strategy, "-> group shape", g.shape, "space", type(g.space).__name__)
    print("  sample row 0 holds voxel", beta[0, 0], "(s1) beside voxel", beta[0, 1] - 1000, "(s2)")
# R 'intersect' would keep only voxels {2,3} and route s1 cols [2,3], s2 cols [0,1].
common = np.flatnonzero(m1 & m2)
print("R-intersect expected common voxels:", common.tolist(),
      "s1 positions", [int(np.flatnonzero(np.flatnonzero(m1) == v)[0]) for v in common],
      "s2 positions", [int(np.flatnonzero(np.flatnonzero(m2) == v)[0]) for v in common])
