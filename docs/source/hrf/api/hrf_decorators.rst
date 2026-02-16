.. _api_hrf_decorators:

HRF Decorators
==============

Decorators transform an existing HRF into a new one by applying temporal
lag, block-duration convolution, or amplitude normalisation.  They return
a new :class:`~fmrimod.hrf.core.HRF` object -- the original is never
mutated.

Single-HRF transforms
----------------------

.. autofunction:: fmrimod.hrf.decorators.lag_hrf

.. autofunction:: fmrimod.hrf.decorators.block_hrf

.. autofunction:: fmrimod.hrf.decorators.normalize_hrf

Multi-HRF generators
---------------------

These produce a multi-basis HRF by applying a decorator across a
sequence of parameter values.

.. autofunction:: fmrimod.hrf.decorators.gen_hrf_lagged

.. autofunction:: fmrimod.hrf.decorators.gen_hrf_blocked
