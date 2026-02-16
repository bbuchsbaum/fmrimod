.. _api_hrf_core:

HRF Core
========

The abstract base class :class:`~fmrimod.hrf.core.HRF` defines the
interface shared by every hemodynamic response function in fmrimod.
Concrete HRFs are instances of :class:`~fmrimod.hrf.core.FunctionHRF`,
which wraps any callable as an HRF object.

Base class
----------

.. autoclass:: fmrimod.hrf.core.HRF
   :members:
   :no-index:
   :special-members: __call__

.. autoclass:: fmrimod.hrf.core.FunctionHRF
   :members:
   :no-index:
   :show-inheritance:

Factory helpers
---------------

.. autofunction:: fmrimod.hrf.core.as_hrf

.. autofunction:: fmrimod.hrf.core.bind_basis

.. autofunction:: fmrimod.hrf.core.hrf_from_coefficients
