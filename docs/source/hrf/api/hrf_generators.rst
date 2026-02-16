.. _api_hrf_generators:

HRF Generators
==============

Generator functions create :class:`~fmrimod.hrf.core.HRF` objects with
caller-specified parameters (number of basis functions, temporal span, etc.).

General-purpose
----------------

.. autofunction:: fmrimod.hrf.generators.gen_hrf

.. autofunction:: fmrimod.hrf.generators.gen_hrf_set

.. autofunction:: fmrimod.hrf.generators.make_hrf

Basis-set factories
--------------------

.. autofunction:: fmrimod.hrf.generators.bspline_generator

.. autofunction:: fmrimod.hrf.generators.fir_generator

.. autofunction:: fmrimod.hrf.generators.fourier_generator

.. autofunction:: fmrimod.hrf.generators.daguerre_generator

.. autofunction:: fmrimod.hrf.generators.tent_generator

Simple-shape factories
-----------------------

.. autofunction:: fmrimod.hrf.generators.boxcar_generator

.. autofunction:: fmrimod.hrf.generators.weighted_generator

.. autofunction:: fmrimod.hrf.generators.gamma_generator
