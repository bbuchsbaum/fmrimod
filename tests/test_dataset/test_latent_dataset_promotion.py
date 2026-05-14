"""Canonical-home contract for `LatentDataset` / `latent_dataset`.

These typed values were promoted out of `fmrimod/dataset/compat.py`
into `fmrimod/dataset/latent.py`. The promotion preserves every prior
import path; this test pins the canonical home and the back-compat
re-exports so the move cannot regress silently.
"""

from __future__ import annotations

import numpy as np

import fmrimod
from fmrimod import dataset as ds_pkg
from fmrimod.dataset import compat as ds_compat
from fmrimod.dataset import latent as ds_latent


def test_latent_dataset_canonical_module_is_dataset_latent():
    assert ds_latent.LatentDataset.__module__ == "fmrimod.dataset.latent"
    assert ds_latent.latent_dataset.__module__ == "fmrimod.dataset.latent"


def test_back_compat_reexports_resolve_to_canonical_class():
    """Class identity routes everywhere through `dataset/latent.py`.

    Constructor identity is preserved at the dataset and top-level
    surfaces (`fmrimod.latent_dataset` is the canonical function).
    The `dataset/compat` constructor is a deprecation shim, not the
    canonical function — see :func:`test_compat_latent_dataset_warns`."""
    assert ds_compat.LatentDataset is ds_latent.LatentDataset
    assert ds_pkg.LatentDataset is ds_latent.LatentDataset
    assert ds_pkg.latent_dataset is ds_latent.latent_dataset
    assert fmrimod.latent_dataset is ds_latent.latent_dataset


def test_compat_latent_dataset_warns_and_delegates():
    """`fmrimod.dataset.compat.latent_dataset` is a deprecation shim
    that calls through to the canonical constructor."""
    import warnings

    rng = np.random.default_rng(11)
    scores = rng.standard_normal((12, 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        latent = ds_compat.latent_dataset(scores, tr=1.5)

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "compat.latent_dataset must emit DeprecationWarning"
    assert "compat.latent_dataset" in str(deprecation[0].message)
    assert isinstance(latent, ds_latent.LatentDataset)
    assert latent.sampling_frame.TR == 1.5


def test_latent_dataset_constructor_smoke():
    rng = np.random.default_rng(7)
    scores = rng.standard_normal((20, 3))
    loadings = rng.standard_normal((3, 4))
    latent = fmrimod.latent_dataset(scores, loadings=loadings, tr=2.0)
    assert latent.n_runs == 1
    assert latent.n_voxels == 4
    np.testing.assert_allclose(latent.get_data(0), scores @ loadings)


def test_latent_dataset_validation_messages():
    """Validation errors at construction stay tight after the move."""
    import pytest

    with pytest.raises(ValueError, match="2-D"):
        fmrimod.latent_dataset(np.zeros(10))
    with pytest.raises(ValueError, match="run_length must divide"):
        fmrimod.latent_dataset(np.zeros((10, 2)), run_length=3)
