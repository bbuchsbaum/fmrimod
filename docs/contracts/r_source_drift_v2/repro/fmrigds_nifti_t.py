"""GDS: NIfTI t/df source -> GroupDataset (e71c9b3, test-nifti-t-df.R)."""
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

from fmrimod.dataset import group_data_from_nifti
from fmrimod.group import group_dataset_from_group_data

warnings.simplefilter("ignore")
rng = np.random.default_rng(5)
with tempfile.TemporaryDirectory() as d:
    files, vals = [], []
    for i in range(4):
        v = rng.normal(size=(3, 3, 2)).astype(np.float32)
        f = Path(d) / f"sub-{i + 1:02d}_stat-t.nii.gz"
        nib.save(nib.Nifti1Image(v, np.diag([2, 2, 2, 1.0])), f)
        files.append(str(f))
        vals.append(v)
    gd = group_data_from_nifti(t_paths=files, df=[20, 22, 25, 30])
    ds = group_dataset_from_group_data(gd)
    print("assays:", ds.assay_names(), "shape:", ds.shape)
    print("t matches:", all(np.allclose(ds.assay("t")[:, i, 0], vals[i].ravel()) for i in range(4)))
    print("df per subject:", ds.assay("df")[0, :, 0])
    for bad in (0, -3, 18.5):
        try:
            group_data_from_nifti(t_paths=files, df=bad)
            print(f"df={bad!r}: accepted")
        except Exception as exc:  # noqa: BLE001
            print(f"df={bad!r}: {type(exc).__name__}: {exc}")

# 8b145f9: images on different grids (same dims, different affine) must be rejected.
with tempfile.TemporaryDirectory() as d:
    fa, fb, sa, sb = (str(Path(d) / n) for n in ("a.nii.gz", "b.nii.gz", "sa.nii.gz", "sb.nii.gz"))
    v = np.ones((3, 3, 2), np.float32)
    nib.save(nib.Nifti1Image(v, np.diag([2, 2, 2, 1.0])), fa)
    nib.save(nib.Nifti1Image(v, np.diag([3, 3, 3, 1.0])), fb)
    nib.save(nib.Nifti1Image(v, np.diag([2, 2, 2, 1.0])), sa)
    nib.save(nib.Nifti1Image(v, np.diag([3, 3, 3, 1.0])), sb)
    try:
        group_data_from_nifti(beta_paths=[fa, fb], se_paths=[sa, sb], subjects=["s1", "s2"])
        print("mismatched affines (2mm vs 3mm, same dims): accepted")
    except Exception as exc:  # noqa: BLE001
        print("mismatched affines rejected:", exc)
