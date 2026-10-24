"""DS check: spatial identity of mask and runs (R 7cb121e / read_bids_bold 'explicit spatial ambiguity failures').

R HEAD refuses a volume_space that differs from the NIfTI source's native volume (other affine
or reordered support) and read_bids_bold() fails on spatially ambiguous runs. fmrimod's
NeuroVecAdapter compares only the 3-D shape.
"""
import neuroim as ni
import numpy as np

import fmrimod.dataset as fds

shape = (4, 4, 3)
aff_a = np.diag([2.0, 2.0, 2.0, 1.0])
aff_b = np.array([[-2.0, 0, 0, 8.0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]])  # x flipped
rng = np.random.default_rng(3)

def vec(aff, n=8):
    return ni.DenseNeuroVec(rng.normal(size=shape + (n,)) + 10,
                            ni.NeuroSpace.from_affine(aff, shape + (n,)))

mask_b = ni.LogicalNeuroVol(np.ones(shape, bool), ni.NeuroSpace.from_affine(aff_b, shape))
for label, runs, mask in (
    ("mask affine != run affine", [vec(aff_a)], mask_b),
    ("run 2 affine != run 1 affine", [vec(aff_a), vec(aff_b)], None),
):
    try:
        ds = fds.fmri_dataset(runs, mask=mask, tr=2.0)
        print(f"{label}: ACCEPTED; dataset affine =\n{ds._source.get_affine()[:3]}")
    except Exception as e:  # noqa: BLE001
        print(f"{label}: refused ({type(e).__name__}: {e})")
