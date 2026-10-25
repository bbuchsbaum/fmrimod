.. _api_plotting:

Plotting
========

HRF and regressor figures, ported from R fmrihrf's ``plot_hrfs`` /
``plot_regressors``. Every function returns a matplotlib
:class:`~matplotlib.figure.Figure` drawn under fmrimod's theme
(:mod:`fmrimod.visualization.style`): curves on a shared time axis, overlaid
or stacked one panel per curve, with a zero rule, events as bars under the
curves, an optional dashed grey reference curve, and direct labels at the
peaks of a single basis set. Single-panel views accept ``ax=`` to draw into
an existing axes.

.. code-block:: python

   from fmrimod.hrf.library import HRF_SPMG1, HRF_SPMG3, HRF_GAUSSIAN
   from fmrimod.plotting import plot_hrfs, plot_regressors
   from fmrimod.regressor import regressor

   plot_hrfs(HRF_SPMG1, HRF_GAUSSIAN, normalize=True)   # compare shapes
   plot_hrfs(HRF_SPMG3)                                 # one curve per basis
   plot_hrfs(HRF_SPMG3, layout="stack")                 # one panel per basis

   reg = regressor([10, 30, 50], HRF_SPMG1, duration=[0, 4, 0])
   plot_regressors({"A": reg, "B": reg.shift(5)}, show_onsets=True)

The data each figure draws are available as a long data frame (``time``,
``series``, ``response``) from :func:`~fmrimod.plotting.hrf_curves` and
:func:`~fmrimod.plotting.regressor_curves`.

.. autofunction:: fmrimod.plotting.plot_hrfs

.. autofunction:: fmrimod.plotting.plot_hrf

.. autofunction:: fmrimod.plotting.plot_regressors

.. autofunction:: fmrimod.plotting.plot_regressor

.. autofunction:: fmrimod.plotting.hrf_curves

.. autofunction:: fmrimod.plotting.regressor_curves
