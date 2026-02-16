.. _api_hrf_library:

Pre-defined HRF Objects
=======================

Ready-to-use HRF instances available as module-level constants.
These are the objects returned by :func:`~fmrimod.get_hrf` when
you pass the corresponding name string.

SPM family
----------

.. autodata:: fmrimod.hrf.library.SPM_CANONICAL
   :annotation:

.. autodata:: fmrimod.hrf.library.SPM_WITH_DERIVATIVE
   :annotation:

.. autodata:: fmrimod.hrf.library.SPM_WITH_DISPERSION
   :annotation:

R-compatibility aliases: ``HRF_SPMG1``, ``HRF_SPMG2``, ``HRF_SPMG3``.

Statistical
-----------

.. autodata:: fmrimod.hrf.library.GAMMA_HRF
   :annotation:

.. autodata:: fmrimod.hrf.library.GAUSSIAN_HRF
   :annotation:

Basis sets
----------

.. autodata:: fmrimod.hrf.library.BSPLINE_HRF
   :annotation:

.. autodata:: fmrimod.hrf.library.FIR_HRF
   :annotation:

HRF library utilities
---------------------

.. autofunction:: fmrimod.hrf.hrf_library.hrf_library
