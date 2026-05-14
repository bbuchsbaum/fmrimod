"""SBHM (Shared-Basis HRF Matching) single-trial estimation.

This sub-package implements the SBHM pipeline from R's fmrilss package.
SBHM uses a library of candidate HRF shapes to estimate per-voxel HRF
variability and then computes single-trial amplitudes with the matched
HRF shape.

Pipeline
--------
1. **Library construction**: SVD of HRF library → shared basis
2. **Prepass**: Aggregate regression to get basis coefficients
3. **Matching**: Find best-matching library HRF per voxel
4. **Amplitude**: Single-trial estimation with matched HRF

Example
-------
>>> from fmrimod.single.sbhm import sbhm_single_trial
>>> result = sbhm_single_trial(Y, X, confounds=confounds)
>>> result.betas  # (n_trials, n_voxels)
"""

from .amplitude import sbhm_amplitude
from .library import SbhmLibrary, build_sbhm_library
from .match import SbhmMatchResult, sbhm_match
from .pipeline import sbhm_single_trial
from .prepass import sbhm_prepass

__all__ = [
    "SbhmLibrary",
    "build_sbhm_library",
    "sbhm_prepass",
    "SbhmMatchResult",
    "sbhm_match",
    "sbhm_amplitude",
    "sbhm_single_trial",
]
