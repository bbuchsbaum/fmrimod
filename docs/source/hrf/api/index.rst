.. _api_reference:

API Reference
=============

This section documents every public class and function in fmrimod.
Objects are grouped by module; each entry links to a detail page with
full parameter descriptions, return types, and examples.

.. contents:: Modules
   :local:
   :depth: 1

----

HRF -- Hemodynamic Response Functions
--------------------------------------

Core classes, pre-defined HRFs, mathematical functions, decorators,
generators, and the pluggable registry.

.. toctree::
   :maxdepth: 1

   hrf_core
   hrf_functions
   hrf_library
   hrf_registry
   hrf_decorators
   hrf_generators
   hrf_advanced

Regressor -- Design-matrix construction
----------------------------------------

Event-related regressors, multi-condition regressor sets, neural-input
generation, convolution, and design-matrix utilities.

.. toctree::
   :maxdepth: 1

   regressor

Sampling -- Acquisition timing
-------------------------------

Multi-block, multi-TR sampling-frame specification.

.. toctree::
   :maxdepth: 1

   sampling

Plotting -- Visualisation
--------------------------

Convenience functions for plotting HRFs, basis sets, regressors, and
design matrices.

.. toctree::
   :maxdepth: 1

   plotting

Utilities
---------

Helper functions for single-trial regressors, Toeplitz matrices, and
caching.

.. toctree::
   :maxdepth: 1

   utils
