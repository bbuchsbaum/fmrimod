.. _api_hrf_advanced:

Advanced HRF Utilities
======================

Penalty matrices
-----------------

Penalty (roughness) matrices for regularised estimation with multi-basis
HRFs.  The matrix :math:`\mathbf{P}` is used in penalised least squares
as :math:`(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{P})^{-1}\mathbf{X}^\top\mathbf{y}`.

.. autofunction:: fmrimod.hrf.penalty.penalty_matrix

Empirical HRFs
---------------

Construct an HRF object from observed data (e.g. an FIR-estimated
response), with optional smoothing.

.. autofunction:: fmrimod.hrf.empirical.empirical_hrf

Reconstruction matrices
------------------------

Convert a vector of basis-function coefficients back to a dense HRF
time course at arbitrary temporal resolution.

.. autofunction:: fmrimod.hrf.reconstruction.reconstruction_matrix

SPM-specific derivatives
-------------------------

.. automodule:: fmrimod.hrf.derivatives
   :members:
