Installation
============

Requirements
------------

fmrimod requires Python 3.8 or later and the following dependencies:

- NumPy >= 1.21.0
- SciPy >= 1.7.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0 (for plotting)

Installing from PyPI
--------------------

The easiest way to install fmrimod is using pip:

.. code-block:: bash

    pip install fmrimod

Installing from Source
----------------------

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/bbuchsbaum/fmrimod.git
    cd fmrimod
    pip install -e .

For development with all optional dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This includes testing tools (pytest), code formatting (black), and type checking (mypy).

Verifying Installation
----------------------

You can verify your installation by running:

.. code-block:: python

    import fmrimod
    print(fmrimod.__version__)
    
    # List available HRFs
    from fmrimod import list_available_hrfs
    print(list_available_hrfs())

Migration from R
----------------

If you're migrating from the R fmrihrf package, fmrimod provides equivalent 
functionality with a Pythonic API. See the :doc:`vignettes/index` for 
detailed tutorials that mirror the R package vignettes.