"""DS check: unsigned NIfTI dtypes and scl_slope through fmrimod image paths (R 1b58be9 analogue)."""
import pathlib
import tempfile

import nibabel as nib
import numpy as np

import fmrimod as fm
from fmrimod.dataset import nifti_backend

aff = np.diag([2.0, 2.0, 2.0, 1.0])
with tempfile.TemporaryDirectory() as d:
    for dt, hi in (("uint8", 250), ("uint16", 65000), ("uint32", 4_000_000_000)):
        p = pathlib.Path(d) / f"bold_{dt}.nii"
        m = pathlib.Path(d) / "mask.nii"
        arr = np.full((2, 2, 2, 3), hi, dtype=dt)
        img = nib.Nifti1Image(arr, aff)
        img.header.set_data_dtype(dt)
        nib.save(img, p)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype="uint8"), aff), m)
        ds = fm.fmri_dataset(str(p), mask=str(m), tr=2.0)
        v1 = ds.get_run_data(0)[0, 0]
        be = nifti_backend(str(p), str(m)); be.open()
        v2 = be.get_data()[0, 0]
        print(dt, "stored", hi, "fmri_dataset->", v1, "nifti_backend->", v2)
