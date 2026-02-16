.. _api_hrf_functions:

HRF Mathematical Functions
==========================

Low-level functions that compute HRF values at arbitrary time points.
Each function accepts a 1-D array ``t`` (seconds) and returns an array of
the same length (single-basis) or shape ``(len(t), nbasis)`` (multi-basis).

These are the building blocks used by :func:`~fmrimod.gen_hrf` and the
pre-defined HRF objects in :mod:`fmrimod.hrf.library`.

SPM family
----------

.. autofunction:: fmrimod.hrf.functions.spm_canonical

Statistical distributions
--------------------------

.. autofunction:: fmrimod.hrf.functions.gamma_hrf

.. autofunction:: fmrimod.hrf.functions.gaussian_hrf

Basis sets
----------

.. autofunction:: fmrimod.hrf.functions.bspline_hrf

.. autofunction:: fmrimod.hrf.functions.fir_basis

.. autofunction:: fmrimod.hrf.functions.fourier_hrf

Specialised shapes
-------------------

.. autofunction:: fmrimod.hrf.functions.mexhat_hrf

.. autofunction:: fmrimod.hrf.functions.sine_hrf

.. autofunction:: fmrimod.hrf.functions.half_cosine_hrf

.. autofunction:: fmrimod.hrf.functions.inv_logit_hrf

.. autofunction:: fmrimod.hrf.functions.lwu_hrf

Simple / utility shapes
------------------------

.. autofunction:: fmrimod.hrf.functions.boxcar_hrf

.. autofunction:: fmrimod.hrf.functions.weighted_hrf

.. autofunction:: fmrimod.hrf.functions.hrf_time

.. autofunction:: fmrimod.hrf.functions.hrf_ident
