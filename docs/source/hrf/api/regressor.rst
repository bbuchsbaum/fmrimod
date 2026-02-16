.. _api_regressor:

Regressor
=========

Classes and functions for constructing predicted BOLD time series from
event timings and HRFs.

Core classes
------------

.. autoclass:: fmrimod.regressor.core.Regressor
   :members:
   :no-index:
   :special-members: __call__

.. autoclass:: fmrimod.regressor.core.RegressorSet
   :members:
   :no-index:

Factory functions
-----------------

.. autofunction:: fmrimod.regressor.regressor

.. autofunction:: fmrimod.regressor.regressor_set

.. autofunction:: fmrimod.regressor.null_regressor

Design-matrix construction
---------------------------

.. autofunction:: fmrimod.regressor.design.regressor_design

Neural input
------------

.. automodule:: fmrimod.regressor.neural_input
   :members:

Convolution
-----------

.. automodule:: fmrimod.regressor.convolution
   :members:
