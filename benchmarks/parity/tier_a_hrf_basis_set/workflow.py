"""HRF basis-set parity: SPMG3 + informed-basis F-tests against Nilearn.

This case stresses an angle absent from the other Tier A workflows:
each condition is convolved with a **three-column HRF basis set**
(SPM canonical + temporal derivative + dispersion derivative), and
the headline analytic outputs are *informed-basis* F-tests — joint
F-statistics over the three basis columns per condition, which are
robust to HRF shape mis-specification and are the SPM default for
exploratory event-related analyses.

Why this is a stress test
-------------------------
Nilearn's ``hrf_model="spm + derivative + dispersion"`` produces six
task columns (``A, A_derivative, A_dispersion, B, B_derivative,
B_dispersion``) but exposes them only by string name. To build a
condition-specific informed-basis F-test the user enumerates
``["A", "A_derivative", "A_dispersion"]`` by hand, builds the 3-row
contrast matrix, and prays the column order didn't change underneath
them.

fmrimod's typed design surface tags every basis-set column with
``(term, level, basis_ix, basis_total, basis_name)``. The typed
contrast builder reads::

    canon = cols.where(term="trial_type", level="A", basis_ix=1)
    f_A = stack(
        cols.where(term="trial_type", level="A", basis_ix=k).one()
        for k in (1, 2, 3)
    )

— no string parsing, no positional column counting.

A second finding surfaced while wiring this case: fmrimod's
``SPMG3`` basis defines the third column as the **second time
derivative** of the canonical HRF, but Nilearn's
``spm_dispersion_derivative`` defines it as the **derivative with
respect to the dispersion parameter**. Same name in both libraries,
mathematically different basis functions (correlation 0.01 on the
realised columns). The two engines also differ on the temporal
derivative numerically — different oversampling grids, different dt.
That makes column-by-column realised-design parity impossible.

Pattern B parity claim
----------------------
fmrimod realises the SPMG3 concatenated design via the typed spec;
both engines solve OLS on that **same** ``X``
(``fm.fmri_lm(spec, ds, engine="concat")`` on fmrimod;
``run_glm(data, X)`` on Nilearn). Compared outputs cover the
single-column canonical effect, the canonical contrast between
conditions, both informed-basis 3-DF F-tests, the 3-DF F-test for
the HRF-shape *difference* between conditions, and the 6-DF omnibus
F over the whole task block.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from numpy.typing import NDArray

import fmrimod as fm
from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.design.columns import DesignColumns
from fmrimod.spec import drift, hrf

Array = NDArray[np.float64]

TR = 2.0
N_SCANS = 180
MAX_VOXELS = 1024
TRIAL_TYPES: tuple[str, ...] = ("A", "B")
BASIS_TOTAL = 3
BASIS_NAMES = ("canonical", "temporal_deriv", "dispersion_deriv")


@dataclass(frozen=True)
class BasisSetInputs:
    """Shared inputs for the SPMG3 basis-set parity case."""

    events: pd.DataFrame
    data: Array
    design: Array
    design_columns: DesignColumns
    cell_indices: dict[tuple[str, int], int]
    c_t_A_canonical: Array
    c_t_AB_canonical_diff: Array
    c_f_A_informed: Array
    c_f_B_informed: Array
    c_f_AB_diff_informed: Array
    c_f_task_omnibus: Array


def _make_events(seed: int) -> pd.DataFrame:
    """Twelve well-spaced trials per condition, jittered onsets."""
    rng = np.random.default_rng(seed)
    n_per_cond = 12
    rows: list[dict[str, Any]] = []
    for cond_idx, trial_type in enumerate(TRIAL_TYPES):
        grid = np.linspace(
            12.0 + 2.0 * cond_idx,
            N_SCANS * TR - 30.0,
            n_per_cond,
            dtype=np.float64,
        )
        jitter = rng.uniform(-1.2, 1.2, n_per_cond)
        for onset in grid + jitter:
            rows.append(
                {
                    "onset": float(onset),
                    "duration": 0.0,
                    "trial_type": trial_type,
                    "run": 1,
                }
            )
    return (
        pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    )


def _realize_design(
    events: pd.DataFrame,
) -> tuple[Array, DesignColumns]:
    """Build the SPMG3 design via the typed spec."""
    spec = (
        hrf("trial_type", basis="spmg3", norm="spm")
        + drift("cosine", cutoff=128.0)
    )
    dummy = fm.fmri_dataset(
        np.zeros((N_SCANS, 1), dtype=np.float64),
        tr=TR,
        events=events,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, dummy, engine="concat")
    design = np.asarray(
        fit.model.design_matrix_array(run=None), dtype=np.float64
    )
    return design, fit.design_columns()


def _cell_indices(
    columns: DesignColumns,
) -> dict[tuple[str, int], int]:
    """Look up every (trial_type, basis_ix) task column by typed lookup.

    This is the headline ergonomic claim: each basis-set column is
    addressable by its semantic identity (condition × basis component
    index) without referring to its position or parsing its name.
    """
    cells: dict[tuple[str, int], int] = {}
    for trial_type in TRIAL_TYPES:
        for basis_ix in range(1, BASIS_TOTAL + 1):
            matches = columns.where(
                term="trial_type", level=trial_type, basis_ix=basis_ix
            )
            cells[(trial_type, basis_ix)] = matches.one().index
    return cells


def _build_contrasts(
    cells: dict[tuple[str, int], int], n_total: int
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Assemble the six contrast vectors / matrices.

    - ``c_t_A_canonical``: t for A's canonical column alone.
    - ``c_t_AB_canonical_diff``: t for ``A_canonical - B_canonical``.
    - ``c_f_A_informed``: 3-DF F over A's three basis columns.
    - ``c_f_B_informed``: 3-DF F over B's three basis columns.
    - ``c_f_AB_diff_informed``: 3-DF F over the ``A-B`` difference in
      each basis component (tests whether the *shape* of the HRF
      differs between conditions, not just the canonical amplitude).
    - ``c_f_task_omnibus``: 6-DF F over the whole task block.
    """
    c_t_A = np.zeros(n_total, dtype=np.float64)
    c_t_A[cells[("A", 1)]] = 1.0

    c_t_AB = np.zeros(n_total, dtype=np.float64)
    c_t_AB[cells[("A", 1)]] = +1.0
    c_t_AB[cells[("B", 1)]] = -1.0

    c_f_A = np.zeros((BASIS_TOTAL, n_total), dtype=np.float64)
    c_f_B = np.zeros((BASIS_TOTAL, n_total), dtype=np.float64)
    c_f_AB_diff = np.zeros((BASIS_TOTAL, n_total), dtype=np.float64)
    for row, basis_ix in enumerate(range(1, BASIS_TOTAL + 1)):
        c_f_A[row, cells[("A", basis_ix)]] = 1.0
        c_f_B[row, cells[("B", basis_ix)]] = 1.0
        c_f_AB_diff[row, cells[("A", basis_ix)]] = +1.0
        c_f_AB_diff[row, cells[("B", basis_ix)]] = -1.0

    c_f_omni = np.zeros(
        (BASIS_TOTAL * len(TRIAL_TYPES), n_total), dtype=np.float64
    )
    for row, idx in enumerate(
        sorted(cells[(t, b)] for t in TRIAL_TYPES for b in range(1, BASIS_TOTAL + 1))
    ):
        c_f_omni[row, idx] = 1.0

    return c_t_A, c_t_AB, c_f_A, c_f_B, c_f_AB_diff, c_f_omni


def load_inputs(
    max_voxels: int = MAX_VOXELS, seed: int = 20260514
) -> BasisSetInputs:
    """Synthesize a single-run BOLD time series with known HRF-shape effects.

    The true generating model places different amplitudes on the
    canonical and temporal-derivative columns for A vs B (B's HRF is
    shifted slightly earlier in time via a non-zero temporal-derivative
    component). That makes both the canonical contrast and the
    informed-basis F-tests detectable above noise without making either
    trivial.
    """
    rng = np.random.default_rng(seed)
    events = _make_events(seed)
    design, columns = _realize_design(events)
    cells = _cell_indices(columns)
    n_total = design.shape[1]

    n_voxels = min(int(max_voxels), MAX_VOXELS)
    betas = np.zeros((n_total, n_voxels), dtype=np.float64)
    cell_amplitudes = {
        ("A", 1): 1.10,   # A canonical
        ("A", 2): 0.10,   # A temporal-deriv (small)
        ("A", 3): 0.00,   # A dispersion (none)
        ("B", 1): 0.55,   # B canonical (different amplitude)
        ("B", 2): 0.35,   # B temporal-deriv (larger -> earlier-peaking B)
        ("B", 3): -0.20,  # B dispersion (non-zero -> width differs)
    }
    voxel_ramp = np.linspace(0.5, 1.5, n_voxels, dtype=np.float64)
    for cell, amp in cell_amplitudes.items():
        betas[cells[cell]] = amp * voxel_ramp

    # Baseline + drift coefficients.
    for c in columns.columns:
        if c.role == "baseline" and "constant" in (c.name or ""):
            betas[c.index] = 100.0 + rng.normal(scale=0.4, size=n_voxels)
        elif c.role == "baseline":
            betas[c.index] = rng.normal(scale=0.15, size=n_voxels)

    data = design @ betas + rng.normal(scale=0.35, size=(N_SCANS, n_voxels))

    c_t_A, c_t_AB, c_f_A, c_f_B, c_f_AB_diff, c_f_omni = _build_contrasts(
        cells, n_total
    )
    return BasisSetInputs(
        events=events,
        data=data.astype(np.float64),
        design=design,
        design_columns=columns,
        cell_indices=cells,
        c_t_A_canonical=c_t_A,
        c_t_AB_canonical_diff=c_t_AB,
        c_f_A_informed=c_f_A,
        c_f_B_informed=c_f_B,
        c_f_AB_diff_informed=c_f_AB_diff,
        c_f_task_omnibus=c_f_omni,
    )


def nilearn_pipeline(inputs: BasisSetInputs) -> PipelineOutput:
    """Reference: ``run_glm`` on the fmrimod-realised SPMG3 design."""
    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_A = compute_contrast(
        labels, estimates, inputs.c_t_A_canonical, stat_type="t"
    )
    t_AB = compute_contrast(
        labels, estimates, inputs.c_t_AB_canonical_diff, stat_type="t"
    )
    f_A = compute_contrast(
        labels, estimates, inputs.c_f_A_informed, stat_type="F"
    )
    f_B = compute_contrast(
        labels, estimates, inputs.c_f_B_informed, stat_type="F"
    )
    f_AB = compute_contrast(
        labels, estimates, inputs.c_f_AB_diff_informed, stat_type="F"
    )
    f_omni = compute_contrast(
        labels, estimates, inputs.c_f_task_omnibus, stat_type="F"
    )
    rank_observed = int(np.linalg.matrix_rank(inputs.design))
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_A_canonical": np.asarray(t_A.effect_size(), np.float64),
            "t_A_canonical": np.asarray(t_A.stat(), np.float64),
            "effect_AB_canonical_diff": np.asarray(
                t_AB.effect_size(), np.float64
            ),
            "t_AB_canonical_diff": np.asarray(t_AB.stat(), np.float64),
            "f_A_informed_3df": np.asarray(f_A.stat(), np.float64),
            "f_B_informed_3df": np.asarray(f_B.stat(), np.float64),
            "f_AB_diff_informed_3df": np.asarray(f_AB.stat(), np.float64),
            "f_task_omnibus_6df": np.asarray(f_omni.stat(), np.float64),
            "rank": np.array([rank_observed], dtype=np.float64),
        }
    )


def fmrimod_pipeline(inputs: BasisSetInputs) -> PipelineOutput:
    """Typed fmrimod path: ``fmri_lm(spec, ds, engine="concat")``."""
    spec = (
        hrf("trial_type", basis="spmg3", norm="spm")
        + drift("cosine", cutoff=128.0)
    )
    ds = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fm.fmri_lm(spec, ds, engine="concat")

    t_A = fit.contrast(inputs.c_t_A_canonical, name="A_canonical")
    t_AB = fit.contrast(inputs.c_t_AB_canonical_diff, name="AB_canonical_diff")
    f_A = fit.contrast(inputs.c_f_A_informed, name="A_informed")
    f_B = fit.contrast(inputs.c_f_B_informed, name="B_informed")
    f_AB = fit.contrast(inputs.c_f_AB_diff_informed, name="AB_diff_informed")
    f_omni = fit.contrast(inputs.c_f_task_omnibus, name="task_omnibus")

    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=None),
            "effect_A_canonical": np.asarray(t_A.estimate, np.float64),
            "t_A_canonical": np.asarray(t_A.stat, np.float64),
            "effect_AB_canonical_diff": np.asarray(t_AB.estimate, np.float64),
            "t_AB_canonical_diff": np.asarray(t_AB.stat, np.float64),
            "f_A_informed_3df": np.asarray(f_A.stat, np.float64),
            "f_B_informed_3df": np.asarray(f_B.stat, np.float64),
            "f_AB_diff_informed_3df": np.asarray(f_AB.stat, np.float64),
            "f_task_omnibus_6df": np.asarray(f_omni.stat, np.float64),
            "rank": np.array(
                [int(fit.condition_report().runs[0].rank)], dtype=np.float64
            ),
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the SPMG3 + informed-basis F-test parity case."""
    return ParityCase(
        name="tier_a_hrf_basis_set",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_A_canonical": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_A_canonical": ParityTolerance(rtol=1e-7, atol=1e-8),
            "effect_AB_canonical_diff": ParityTolerance(rtol=1e-8, atol=1e-9),
            "t_AB_canonical_diff": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_A_informed_3df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_B_informed_3df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_AB_diff_informed_3df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "f_task_omnibus_6df": ParityTolerance(rtol=1e-7, atol=1e-8),
            "rank": ParityTolerance(rtol=0.0, atol=0.0),
        },
    )


def main() -> None:
    result = run(make_case())
    out_dir = Path(__file__).resolve().parent / "reports"
    render(result, out_dir)
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
