"""DS check: R 7dad6c0 (get_mask on a nifti-backed dataset keeps the backend's NeuroSpace).

fmrimod: NiftiBackend records the affine in get_metadata(), but a dataset built on it
(BackendAdapter) exposes only a flat mask and no get_affine(), so SpatialContext is None.
"""
import pathlib
import tempfile

import nibabel as nib
import numpy as np

from fmrimod.dataset import FmriDataset, nifti_backend
from fmrimod.dataset.adapters import BackendAdapter
from fmrimod.glm.spatial import SpatialContext
from fmrimod.sampling import SamplingFrame

aff = np.array([[-2.0, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])
with tempfile.TemporaryDirectory() as d:
    p, m = pathlib.Path(d) / "b.nii", pathlib.Path(d) / "m.nii"
    nib.save(nib.Nifti1Image(np.random.default_rng(0).normal(size=(3, 3, 2, 5)), aff), p)
    nib.save(nib.Nifti1Image(np.ones((3, 3, 2), "uint8"), aff), m)
    be = nifti_backend(str(p), str(m)); be.open()
    ds = FmriDataset(BackendAdapter(be, SamplingFrame(blocklens=[5], tr=2.0)))
    print("backend metadata affine kept:", np.allclose(be.get_metadata()["affine"], aff))
    print("dataset mask shape:", ds.get_mask().shape, "| has get_affine:", hasattr(ds._source, "get_affine"))
    print("SpatialContext.from_dataset:", SpatialContext.from_dataset(ds))
