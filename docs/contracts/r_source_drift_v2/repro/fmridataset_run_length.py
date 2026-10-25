"""DS check: R 631cf0d (#81) validates run_length elementwise against per-file volume counts.

fmrimod: image inputs to fmri_dataset derive one run per image (no run_length allowed), but the
backend paths (BackendAdapter over NiftiBackend, latent_dataset(source=[...])) check only the total.
"""
import pathlib
import tempfile

import nibabel as nib
import numpy as np

from fmrimod.dataset import FmriDataset, nifti_backend
from fmrimod.dataset.adapters import BackendAdapter
from fmrimod.sampling import SamplingFrame

aff = np.eye(4)
with tempfile.TemporaryDirectory() as d:
    paths = []
    for i, n in enumerate((6, 7)):
        arr = np.full((2, 2, 2, n), float(i + 1))          # file 1 -> value 1.0, file 2 -> value 2.0
        p = pathlib.Path(d) / f"run{i + 1}.nii"
        nib.save(nib.Nifti1Image(arr, aff), p)
        paths.append(str(p))
    m = pathlib.Path(d) / "mask.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), "uint8"), aff), m)
    be = nifti_backend(paths, str(m)); be.open()
    print("per-file volumes:", be._file_time_dims)
    try:
        ds = FmriDataset(BackendAdapter(be, SamplingFrame(blocklens=[7, 6], tr=2.0)))
        r1 = ds.get_run_data(0)[:, 0]
        print("blocklens [7, 6] ACCEPTED; run 1 values:", r1.tolist(), "(last row comes from file 2)")
    except Exception as e:  # noqa: BLE001
        print("refused:", type(e).__name__, e)
