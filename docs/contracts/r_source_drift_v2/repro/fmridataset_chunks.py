"""DS check: data_chunks voxel indices on a 3-D-mask dataset, and empty-feature chunking.

R legacy data_chunks: voxel_ind = which(mask) (linear index into the volume). R 0.10 bug fix:
block_apply() on a zero-feature frame returns an empty list instead of failing.
"""
import warnings

import neuroim as ni
import numpy as np

import fmrimod.dataset as fds

shape = (3, 2, 2)
mask = np.zeros(shape, bool); mask[1, 1, 1] = True; mask[2, 0, 1] = True
space = ni.NeuroSpace(dim=shape + (5,), spacing=(1, 1, 1, 1), origin=(0, 0, 0, 0))
vec = ni.DenseNeuroVec(np.random.default_rng(0).normal(size=shape + (5,)), space)
d = fds.fmri_dataset(vec, mask=mask, tr=2.0)
ch = fds.data_chunks(d, nchunks=1).collect()[0]
print("chunk.voxel_ind:", ch.voxel_ind.tolist(), "| C-order linear index:", np.flatnonzero(mask).tolist(),
      "| x-coordinates:", np.where(mask)[0].tolist())

empty = fds.matrix_dataset(np.zeros((6, 3)), tr=2.0, mask=np.zeros(3, bool))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        print("empty-mask chunks:", len(fds.data_chunks(empty, nchunks=2).collect()))
    except Exception as e:  # noqa: BLE001
        print("empty-mask data_chunks:", type(e).__name__, e)
