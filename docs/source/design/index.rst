.. fmrimod documentation master file

Design Matrix Documentation
===========================

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
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Migration

   migration_guide
   fmrireg_migration

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
