Vignettes
=========

These vignettes provide comprehensive tutorials for using fmrimod, 
replicating the functionality of the R fmrihrf package vignettes in 
a Pythonic way.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   01_hemodynamic_response
   02_building_regressors
   03_hrf_generators
   04_advanced_modeling
   performance_benchmark

Getting Started
---------------

If you're new to fmrimod, start with the 
:doc:`Hemodynamic Response Functions <01_hemodynamic_response>` vignette 
to understand the basics of HRF modeling.

For Jupyter Users
-----------------

Interactive Jupyter notebook versions of these vignettes are available 
in the ``examples/`` directory of the package repository.

Comparison with R
-----------------

These Python vignettes provide equivalent functionality to the R vignettes:

.. list-table:: R to Python Vignette Mapping
   :widths: 40 40 20
   :header-rows: 1

   * - R Vignette
     - Python Vignette
     - Key Topics
   * - ``a_01_hemodynamic_response.Rmd``
     - :doc:`01_hemodynamic_response`
     - HRF basics, modifications
   * - ``a_02_regressor.Rmd``
     - :doc:`02_building_regressors`
     - Event modeling, design matrices
   * - ``a_03_hrf_generators.Rmd``
     - :doc:`03_hrf_generators`
     - Basis functions, B-splines
   * - ``a_04_advanced_modeling.Rmd``
     - :doc:`04_advanced_modeling`
     - HRF libraries, regularization