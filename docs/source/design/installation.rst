Installation
============

Requirements
------------

* Python 3.10 or higher
* NumPy
* Pandas
* SciPy
* Matplotlib
* fmrimod

Installing the alpha from GitHub
--------------------------------

fmrimod is not yet published on PyPI. Install the current source directly:

.. code-block:: bash

    python -m pip install "fmrimod @ git+https://github.com/bbuchsbaum/fmrimod.git"

Installing from Source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

    git clone https://github.com/bbuchsbaum/fmrimod.git
    cd fmrimod
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -e ".[dev,test]"

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

    uv pip install --python .venv/bin/python -e ".[docs]"

For running tests:

.. code-block:: bash

    uv pip install --python .venv/bin/python -e ".[test]"

For development (includes all optional dependencies):

.. code-block:: bash

    uv pip install --python .venv/bin/python -e ".[dev,test]"

Verifying Installation
----------------------

To verify that fmrimod is installed correctly:

.. code-block:: python

    import fmrimod
    print(fmrimod.__version__)
    
    # Test basic functionality
    from fmrimod import event_model
    help(event_model)
