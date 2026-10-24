"""DS check: does the image path keep NIfTI axis flips (R 7cb121e analogue)?"""
import pathlib
import tempfile

import neuroim as ni
import nibabel as nib
import numpy as np

import fmrimod as fm

aff = np.array([[-2.0, 0, 0, 90.0], [0, 2.0, 0, -126.0], [0, 0, 2.0, -72.0], [0, 0, 0, 1]])
rng = np.random.default_rng(0)
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "bold.nii.gz"
    m = pathlib.Path(d) / "mask.nii.gz"
    nib.save(nib.Nifti1Image(rng.normal(size=(4, 5, 3, 6)).astype("float32"), aff), p)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 3), dtype="uint8"), aff), m)
    vec = ni.read_vec(str(p))
    print("neuroim read_vec affine:\n", np.asarray(vec.space.affine)[:4, :4])
    ds = fm.fmri_dataset(str(p), mask=str(m), tr=2.0)
    print("fmri_dataset get_affine:\n", ds._source.get_affine())
    print("flip kept:", np.allclose(ds._source.get_affine()[:4, :4], aff))
