"""Experiment plugin from the 2026-10-25 R parity run (not part of fmrimod).

Gives fmrimod's SPMG1/2/3 classes R fmrihrf's undershoot ratio (1/6) and
span (24 s) to show which R-parity failures are steward decision D2. Use:
``PYTHONPATH=docs/contracts/r_parity_receipts pytest -p spm_r_conventions_plugin ...``
"""

import gc

import fmrimod.hrf.spm_hrf as spm

_CLASSES = (spm.SPMG1_HRF, spm.SPMG2_HRF, spm.SPMG3_HRF)
for cls in _CLASSES:
    cls.ratio = 1 / 6
    cls.span = 24.0
for obj in gc.get_objects():
    if type(obj) in _CLASSES:
        obj.ratio = 1 / 6
        obj.span = 24.0
