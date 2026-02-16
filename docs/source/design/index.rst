.. fmrimod documentation master file

Welcome to fmrimod's documentation!
=========================================

**fmrimod** is a Python implementation of the R fmridesign package for creating and manipulating 
design matrices for fMRI data analysis. It provides a comprehensive set of tools for experimental 
design specification, HRF convolution, and statistical contrast definition.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Interactive Notebooks

   notebooks/01_hemodynamic_response
   notebooks/02_building_regressors
   notebooks/03_baseline_models
   notebooks/04_event_models

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/event_models
   user_guide/contrasts
   user_guide/baseline
   user_guide/covariates
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Migration & Development

   migration_guide
   contributing
   changelog

Features
--------

* **Formula Interface**: Intuitive formula-based specification of design matrices
* **Event Models**: Flexible event specification with various basis functions
* **HRF Library**: Built-in hemodynamic response functions (SPM, AFNI, custom)
* **Contrasts**: Comprehensive contrast specification and weight computation
* **Baseline Modeling**: Polynomial and spline-based drift modeling
* **Visualization**: Design matrix and contrast visualization tools
* **R Compatibility**: Designed to match R fmridesign functionality

Quick Example
-------------

.. code-block:: python

    from fmrimod import event_model
    
    # Create a simple event model
    model = event_model(
        "condition",
        data=my_data,
        tr=2.0,
        n_scans=200
    )
    
    # Access the design matrix
    X = model.design_matrix

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`