Installation
============

Requirements
------------

* Python 3.8 or higher
* NumPy
* Pandas
* SciPy
* Matplotlib
* fmrimod

Installing from PyPI
--------------------

The easiest way to install fmrimod is via pip:

.. code-block:: bash

    pip install fmrimod

Installing from Source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

    git clone https://github.com/yourusername/fmrimod.git
    cd fmrimod
    pip install -e .

This will install the package in "editable" mode, which is useful for development.

Dependencies
------------

The package requires the following dependencies:

* **numpy** >= 1.20.0 - Array operations and numerical computing
* **pandas** >= 1.3.0 - Data manipulation and event handling
* **scipy** >= 1.7.0 - Scientific computing and spline functions
* **matplotlib** >= 3.4.0 - Plotting and visualization
* **fmrimod** >= 0.1.0 - Hemodynamic response functions

Optional Dependencies
---------------------

For building documentation:

.. code-block:: bash

    pip install -e ".[docs]"

For running tests:

.. code-block:: bash

    pip install -e ".[test]"

For development (includes all optional dependencies):

.. code-block:: bash

    pip install -e ".[dev]"

Verifying Installation
----------------------

To verify that fmrimod is installed correctly:

.. code-block:: python

    import fmrimod
    print(fmrimod.__version__)
    
    # Test basic functionality
    from fmrimod import event_model
    help(event_model)