:html_theme.sidebar_secondary.remove:

.. _fmrimod:

fmrimod
=========

**fMRI Hemodynamic Response Function and Regressor Tools for Python**

fmrimod provides a complete toolkit for specifying, manipulating, and
visualising hemodynamic response functions (HRFs) and building fMRI design
regressors. It is a faithful Python port of the R
`fmrihrf <https://github.com/bbuchsbaum/fmrihrf>`_ package, with Pythonic
enhancements including trial-varying HRFs, a pluggable HRF registry, and
NumPy-native array semantics throughout.

.. code-block:: python

   from fmrimod import get_hrf, regressor, SamplingFrame

   hrf = get_hrf("spmg1")                        # SPM canonical HRF
   reg = regressor([10, 30, 50], hrf=hrf, duration=2.0)
   sf  = SamplingFrame(blocklens=100, tr=2.0)
   bold = reg.evaluate(sf.samples)                # predicted BOLD signal

----

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: 20+ HRF types
      :link: vignettes/01_hemodynamic_response
      :link-type: doc

      SPM canonical, Gamma, Gaussian, B-spline, FIR, Fourier, Laguerre,
      Mexican hat, and more -- with temporal lag, block duration, and
      normalisation modifiers.

   .. grid-item-card:: Flexible regressors
      :link: vignettes/02_building_regressors
      :link-type: doc

      Event-related and block designs with per-trial varying HRFs,
      parametric amplitude modulation, and multi-condition regressor sets.

   .. grid-item-card:: Basis function generators
      :link: vignettes/03_hrf_generators
      :link-type: doc

      Configurable basis-set factories (B-spline, FIR, Fourier, tent,
      Laguerre) that produce multi-column HRF objects for data-driven
      estimation.

   .. grid-item-card:: HRF libraries
      :link: vignettes/04_advanced_modeling
      :link-type: doc

      Systematic parameter-space exploration, reconstruction matrices,
      penalty matrices for regularisation, and empirical HRF construction.

   .. grid-item-card:: Sampling frames
      :link: api/sampling
      :link-type: doc

      Multi-block, multi-TR acquisition timing with block-relative onset
      conversion and concatenation support.

   .. grid-item-card:: Publication-quality plots
      :link: api/plotting
      :link-type: doc

      Built-in plotting for HRFs, basis sets, regressors, and design
      matrices via Matplotlib.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   vignettes/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   changelog
   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
