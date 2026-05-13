.. fmrimod documentation master file

Legacy Design Source Tree
=========================

This Sphinx source tree is retained for migration provenance from the older
design-package documentation. It is not the canonical rendered site. The
canonical user-facing docs live in the Quarto files at ``docs/*.qmd`` and
``docs/tutorials/*.qmd``.

The current project shape is defined by ``MISSION.md`` and ``VISION.md``:
``fmrimod`` is a typed, composable Python library for fMRI experimental design
and statistical modeling. The R ``fmridesign`` behavior is an important
statistical specification, but the Python API is not a mechanical port. Design
objects are expected to feed the load-bearing public seam:

``fmri_dataset -> fmri_lm -> contrast -> group_fit``

Older pages below may still contain migration examples or R-compatibility
notes. Treat them as historical reference unless they agree with the current
Quarto site and contracts.

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
   fmrilss_migration

Features
--------

* **Typed design values**: Event, baseline, HRF, and contrast objects should be
  inspectable and serializable.
* **Formula convenience**: Formula strings are authoring sugar for typed design
  specifications, not the only public shape.
* **Parity discipline**: R behavior is tested as a reference, and divergences
  belong in ``docs/contracts/CAVEATS.md``.
* **Workflow integration**: Design matrices exist to feed the public
  first-level and group-analysis seam.

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
