"""Acceptance tests for FitProvenance — Slice A (bd-01KRGWNV5WGD71QZ1E6YY3ACRG).

Operationalizes VISION.md:99-103. Slice A populates three trivially-derivable
fields (fmrimod_version, solver_path, hrf_norm_modes); fields that cannot be
truthfully populated yet carry explicit status companions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.glm import SketchEngineOptions
from fmrimod.glm.fmri_lm import FitProvenance, FmriLm
from fmrimod.model.config import AROptions, FmriLmConfig
from fmrimod.spec import hrf as hrf_term


def _build_minimal_fit() -> FmriLm:
    """Build the smallest fmri_lm() call that exercises the typed spec path."""
    rng = np.random.default_rng(0)
    n_time, n_voxels = 60, 4
    tr = 2.0
    bold = rng.standard_normal((n_time, n_voxels)).astype(np.float64)

    events = pd.DataFrame(
        {
            "onset": [10.0, 30.0],
            "duration": [2.0, 2.0],
            "trial_type": ["A", "B"],
            "run": [1, 1],
        }
    )

    ds = fm.fmri_dataset(bold, tr=tr, events=events)
    return fm.fmri_lm(hrf_term("trial_type", norm="spm"), ds)


def test_fit_provenance_is_frozen_dataclass() -> None:
    """Provenance shape is closed and immutable."""
    assert dataclasses.is_dataclass(FitProvenance)
    params = dataclasses.fields(FitProvenance)
    field_names = {f.name for f in params}
    # The six VISION-enumerated value slots plus explicit status carriers.
    expected = {
        "fmrimod_version",
        "solver_path",
        "hrf_norm_modes",
        "seed",
        "seed_status",
        "ar_config",
        "ar_status",
        "mask_mode",
        "mask_status",
    }
    assert field_names == expected, (
        f"FitProvenance shape drifted: got {field_names}, expected {expected}"
    )
    # Frozen: attempting to mutate raises.
    prov = FitProvenance(
        fmrimod_version="x", solver_path="y", hrf_norm_modes=()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        prov.fmrimod_version = "z"  # type: ignore[misc]


def test_fmri_lm_has_provenance_field() -> None:
    """FmriLm exposes provenance as a top-level attribute."""
    fit = _build_minimal_fit()
    assert hasattr(fit, "provenance")
    assert fit.provenance is not None
    assert isinstance(fit.provenance, FitProvenance)


def test_fit_provenance_public_glm_import_boundary() -> None:
    """FitProvenance is public through the glm package boundary."""
    from fmrimod.glm import FitProvenance as PublicFitProvenance

    assert PublicFitProvenance is FitProvenance


def test_slice_a_populates_three_fields() -> None:
    """Slice A populates fmrimod_version, solver_path, hrf_norm_modes."""
    fit = _build_minimal_fit()
    prov = fit.provenance
    assert prov is not None

    # 1. fmrimod_version — must equal the live module version.
    assert prov.fmrimod_version == fm.__version__
    assert isinstance(prov.fmrimod_version, str) and prov.fmrimod_version

    # 2. solver_path — non-empty string naming the resolved engine.
    assert isinstance(prov.solver_path, str) and prov.solver_path
    # The trial_type term has an HRF; the default engine is runwise OLS.
    # Don't lock in the exact name, just that it's a non-empty class-like token.
    assert prov.solver_path.replace("_", "").isalnum()

    # 3. hrf_norm_modes — tuple with one entry per term, declared "spm" present.
    assert isinstance(prov.hrf_norm_modes, tuple)
    assert "spm" in prov.hrf_norm_modes, (
        f"declared norm='spm' should survive to provenance; got {prov.hrf_norm_modes}"
    )


def test_slice_a_other_fields_carry_status() -> None:
    """Unwired fields explicitly carry status companions.

    The pushback in work-requests/post-01KRGWDM1A4HE5FWW6P6E7QWBE required
    status-bearing on every field that cannot truthfully populate today;
    a None default would be a typed-looking escape hatch.
    """
    fit = _build_minimal_fit()
    prov = fit.provenance
    assert prov is not None

    assert prov.seed is None
    assert prov.seed_status == "not_randomized"
    assert prov.ar_config is not None
    assert prov.ar_config.struct == "iid"
    assert prov.ar_status == "carried"
    assert prov.mask_mode is None
    assert prov.mask_status == "not_yet_carried"


def test_fit_provenance_records_typed_sketch_seed() -> None:
    """Randomized typed engine options carry their seed explicitly."""
    rng = np.random.default_rng(1)
    events = pd.DataFrame(
        {
            "onset": [10.0, 30.0],
            "duration": [2.0, 2.0],
            "trial_type": ["A", "B"],
            "run": [1, 1],
        }
    )
    ds = fm.fmri_dataset(
        rng.standard_normal((60, 4)).astype(np.float64),
        tr=2.0,
        events=events,
    )

    fit = fm.fmri_lm(
        hrf_term("trial_type"),
        ds,
        engine=SketchEngineOptions(sketch_ratio=1.0, seed=123),
    )

    assert fit.provenance is not None
    assert fit.provenance.solver_path == "SketchEngine"
    assert fit.provenance.seed == 123
    assert fit.provenance.seed_status == "randomized"


def test_fit_provenance_marks_unseeded_sketch_engine_unknown() -> None:
    """Randomized engine without a seed is explicit, not mislabeled deterministic."""
    rng = np.random.default_rng(2)
    events = pd.DataFrame(
        {
            "onset": [10.0, 30.0],
            "duration": [2.0, 2.0],
            "trial_type": ["A", "B"],
            "run": [1, 1],
        }
    )
    ds = fm.fmri_dataset(
        rng.standard_normal((60, 4)).astype(np.float64),
        tr=2.0,
        events=events,
    )

    fit = fm.fmri_lm(
        hrf_term("trial_type"),
        ds,
        engine=SketchEngineOptions(sketch_ratio=1.0),
    )

    assert fit.provenance is not None
    assert fit.provenance.solver_path == "SketchEngine"
    assert fit.provenance.seed is None
    assert fit.provenance.seed_status == "unknown"


def test_fit_provenance_carries_json_safe_ar_config_snapshot() -> None:
    """Fit provenance carries config.ar without ndarray leaves or parcel loss."""
    rng = np.random.default_rng(3)
    events = pd.DataFrame(
        {
            "onset": [10.0, 30.0],
            "duration": [2.0, 2.0],
            "trial_type": ["A", "B"],
            "run": [1, 1],
        }
    )
    ds = fm.fmri_dataset(
        rng.standard_normal((60, 4)).astype(np.float64),
        tr=2.0,
        events=events,
    )
    config = FmriLmConfig(
        ar=AROptions(
            struct="iid",
            censor=np.array([1, 2]),
            parcels=np.array([0, 1, 1, 0]),
        )
    )

    fit = fm.fmri_lm(hrf_term("trial_type"), ds, config=config)

    assert fit.provenance is not None
    assert fit.provenance.ar_status == "carried"
    assert fit.provenance.ar_config is not None
    assert fit.provenance.ar_config is not config.ar
    assert fit.provenance.ar_config.struct == "iid"
    assert fit.provenance.ar_config.censor == [1, 2]
    assert fit.provenance.ar_config.parcels == [0, 1, 1, 0]

    payload = fit.provenance.to_dict()
    assert payload["ar_config"]["censor"] == [1, 2]
    assert payload["ar_config"]["parcels"] == [0, 1, 1, 0]
    assert FitProvenance.from_json(fit.provenance.to_json()) == fit.provenance


def test_provenance_constructor_defaults_match_slice_a() -> None:
    """Direct construction with the three required fields produces Slice A shape."""
    prov = FitProvenance(
        fmrimod_version="0.1.0",
        solver_path="RunwiseEngine",
        hrf_norm_modes=("spm", None),
    )
    assert prov.seed is None
    assert prov.seed_status == "not_randomized"
    assert prov.ar_config is None
    assert prov.ar_status == "not_yet_carried"
    assert prov.mask_mode is None
    assert prov.mask_status == "not_yet_carried"


def test_fit_provenance_json_round_trips_status_fields() -> None:
    """Serialization preserves value slots and explicit status companions."""
    prov = FitProvenance(
        fmrimod_version="0.1.0",
        solver_path="RunwiseEngine",
        hrf_norm_modes=("spm", None),
        seed=None,
        seed_status="not_randomized",
        ar_config=None,
        ar_status="not_yet_carried",
        mask_mode=None,
        mask_status="not_yet_carried",
    )

    payload = prov.to_dict()
    assert payload["schema_version"] == "FitProvenance/v1"
    assert payload["hrf_norm_modes"] == ["spm", None]
    assert payload["seed"] is None
    assert payload["seed_status"] == "not_randomized"
    assert payload["ar_config"] is None
    assert payload["ar_status"] == "not_yet_carried"
    assert payload["mask_mode"] is None
    assert payload["mask_status"] == "not_yet_carried"

    assert FitProvenance.from_dict(payload) == prov
    assert FitProvenance.from_json(prov.to_json()) == prov


def test_fit_provenance_normalizes_direct_array_ar_options() -> None:
    """Direct construction with ndarray AR leaves is still JSON-roundtrippable."""
    prov = FitProvenance(
        fmrimod_version="0.1.0",
        solver_path="RunwiseEngine",
        hrf_norm_modes=("spm", None),
        ar_config=AROptions(
            struct="ar1",
            censor=np.array([1, 2]),
            parcels=np.array([0, 1, 1]),
        ),
        ar_status="carried",
    )

    assert prov.ar_config is not None
    assert prov.ar_config.censor == [1, 2]
    assert prov.ar_config.parcels == [0, 1, 1]

    payload = prov.to_dict()
    assert payload["ar_config"]["censor"] == [1, 2]
    assert payload["ar_config"]["parcels"] == [0, 1, 1]
    assert FitProvenance.from_json(prov.to_json()) == prov
