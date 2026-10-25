"""DS check: R ec6efd3 selection algebra (ADR-010) vs fmrimod get_data(rows=, cols=).

R HEAD: logical selectors must match the axis length (no NA) and act as masks; numeric selectors
must be whole and in bounds, may be negative (exclusion) but not mixed, and repeat at most once.
fmrimod: FmriDataset.get_data / MatrixBackend.get_data cast selectors with np.asarray(..., dtype=intp).
"""
import numpy as np

import fmrimod.dataset as fds

mat = np.arange(40, dtype=float).reshape(8, 5)       # row r holds 5r..5r+4
d = fds.matrix_dataset(mat, tr=2.0)
keep = np.zeros(8, bool); keep[[5, 7]] = True
print("rows=bool mask (rows 5,7) ->", d.get_data(rows=keep)[:, 0].tolist(), "(expected [25.0, 35.0])")
print("cols=[-1]                 ->", d.get_data(cols=np.array([-1]))[0].tolist(), "(numpy wrap to last column; R: exclusion)")
print("rows=[1, 1]               ->", d.get_data(rows=np.array([1, 1]))[:, 0].tolist(), "(R: duplicate refused)")
try:
    print("rows=[2.7]                ->", d.get_data(rows=np.array([2.7]))[:, 0].tolist(), "(R: non-whole refused)")
except Exception as e:  # noqa: BLE001
    print("rows=[2.7] refused:", type(e).__name__)
