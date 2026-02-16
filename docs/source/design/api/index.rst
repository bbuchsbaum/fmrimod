API Reference
=============

This section contains the complete API reference for fmrimod.

.. toctree::
   :maxdepth: 2
   
   core
   events
   contrasts
   baseline
   visualization
   utilities

Core Functions
--------------

.. autosummary::
   :toctree: generated
   :template: function.rst

   fmrimod.event_model
   fmrimod.baseline_model
   fmrimod.contrast
   fmrimod.contrast_weights

Event Types
-----------

.. autosummary::
   :toctree: generated
   :template: class.rst

   fmrimod.events.EventFactor
   fmrimod.events.EventVariable
   fmrimod.events.EventMatrix
   fmrimod.events.EventBasis

Contrast Functions
------------------

.. autosummary::
   :toctree: generated
   :template: function.rst

   fmrimod.contrast.pair_contrast
   fmrimod.contrast.unit_contrast
   fmrimod.contrast.column_contrast
   fmrimod.contrast.poly_contrast
   fmrimod.contrast.oneway_contrast
   fmrimod.contrast.interaction_contrast
   fmrimod.contrast.pairwise_contrasts
   fmrimod.contrast.one_against_all_contrast
   fmrimod.contrast.contrast_set

Special Functions
-----------------

.. autosummary::
   :toctree: generated
   :template: function.rst

   fmrimod.covariate
   fmrimod.trialwise
   fmrimod.cells
   fmrimod.conditions