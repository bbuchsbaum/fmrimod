"""Typed multi-run demo: dataset -> fit -> typed F contrast -> typed accessors.

Shows the multi-run `matrix_dataset -> fmri_lm -> contrast` seam with the
Literal-typed accessor surface. Every dispatch is typed at construction:
no string scopes leak past `fmrimod.accessors`' Literal validators.

This is a companion to ``first_trust_contrast.py``. That demo shows
single-run F-contrast; this one extends to multi-run and exercises the
typed accessor surface (``stats``, ``standard_error``, ``tidy``,
``p_values``) with explicit ``Literal`` arguments.

Friction findings surfaced while writing this demo (see
``examples/README.md`` if/when filed as beads):

1. ``fm.fmri_dataset`` does not expose ``run_length``; multi-run from a
   stacked numpy array requires ``fmrimod.dataset.matrix_dataset``
   directly. The canonical typed seam entry point should accept
   multi-run raw matrices.

2. ``fit.contrast()`` accepts ``OmnibusContrast`` but rejects the rest
   of the typed ``*ContrastSpec`` hierarchy returned by
   ``fm.pair_contrast`` / ``fm.unit_contrast`` / etc. — the typed-spec
   classes are declarative but not wired into the fit surface. Demo
   uses ``OmnibusContrast`` plus a raw-weights t-contrast as a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.contrast import OmnibusContrast
from fmrimod.dataset import matrix_dataset
from fmrimod.spec import hrf

rng = np.random.default_rng(7)

# Two 60-TR runs, 8 voxels.
bold = rng.standard_normal((120, 8))
events = pd.DataFrame(
    {
        "onset": [6.0, 18.0, 30.0, 42.0, 54.0, 66.0, 78.0, 90.0, 102.0, 114.0],
        "duration": [2.0] * 10,
        "trial_type": ["face", "scene", "face", "scene", "face"] * 2,
        "run": [1] * 5 + [2] * 5,
    }
)

dataset = matrix_dataset(bold, tr=2.0, run_length=[60, 60], event_table=events)
fit = fm.fmri_lm(hrf("trial_type", norm="spm"), dataset)

# Typed F contrast across the two levels.
omnibus = OmnibusContrast(term="trial_type", levels=("face", "scene"))
fit.contrast(omnibus)

# Typed accessor surface — every kwarg is a Literal, validated at the boundary.
per_coef_t: np.ndarray = fm.stats(fit, type="estimates")
per_coef_se: np.ndarray = fm.standard_error(fit, type="estimates")
per_coef_p: np.ndarray = fm.p_values(fit, type="estimates")
contrast_fs: dict = fm.stats(fit, type="F")
tidy_estimates: pd.DataFrame = fm.tidy(fit, type="estimates")

print(f"design: {len(fit.model.event_model.column_names)} columns over 2 runs")
print(f"per-coefficient t shape:  {per_coef_t.shape}  (coefficients × voxels)")
print(f"per-coefficient se shape: {per_coef_se.shape}")
print(f"per-coefficient p shape:  {per_coef_p.shape}")
print(f"F-contrasts available:    {sorted(contrast_fs)}")
print(f"tidy rows: {len(tidy_estimates)}  columns: {list(tidy_estimates.columns)}")
print(
    f"provenance: solver={fit.provenance.solver_path}  "
    f"ar={fit.provenance.ar_config.struct}  hrf_norm={fit.provenance.hrf_norm_modes}"
)
