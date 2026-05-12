Migrating from fmrilss
======================

This guide maps the main R ``fmrilss`` workflows to the current Python
``fmrimod`` surface. It is a migration aid, not a line-for-line port of the R
API. New Python code should prefer explicit matrices, dataclass configs, and
``SingleTrialResult`` objects over R-style argument names and bare matrices.

LSS
---

R ``fmrilss::lss(Y, X, Z, Nuisance)`` maps to ``fmrimod.single``:

.. code-block:: python

   import numpy as np
   import fmrimod as fm

   result = fm.lss_single_trial(
       Y,
       trial_regressors,
       baseline_regressors=np.column_stack([intercept, drift]),
       confounds=motion,
   )

   betas = result.betas

The R ``Z`` matrix is called ``baseline_regressors`` in Python. R
``Nuisance`` maps to ``confounds``. If you want the R default
``Z = intercept`` behavior, pass ``include_intercept=True``:

.. code-block:: python

   result = fm.lss_single_trial(Y, trial_regressors, include_intercept=True)

For repeated fits with the same adjustment design, precompute the projection:

.. code-block:: python

   projector = fm.build_nuisance_projector(
       np.column_stack([np.ones(Y.shape[0]), drift, motion])
   )
   result = fm.lss_single_trial(Y, trial_regressors, nuisance_projector=projector)

``lss_single_trial`` uses a vectorized closed-form LSS solve and returns
``SingleTrialResult`` with ``betas``, optional ``se``, residual degrees of
freedom, trial labels, and method metadata.

Compatibility Beta Dispatch
---------------------------

Older fmrireg/fmrilss-style single-trial calls can use the compatibility
dispatcher:

.. code-block:: python

   result = fm.estimate_betas(
       trial_regressors,
       Y,
       method="lss",
       baseline_regressors=baseline,
       confounds=motion,
   )

``fm.glm_lss`` remains available as a thin alias. New code should use
``fm.lss_single_trial`` or ``fm.estimate_single_trial`` when it needs standard
errors, prewhitening, OASIS, SBHM, or mixed-model alternatives.

Other fmrilss Methods
---------------------

The main ``fmrilss`` method families have Python equivalents:

.. list-table::
   :header-rows: 1

   * - R workflow
     - Python surface
   * - ``lss()``, ``lss_optimized_fit()``, ``lss_naive_fit()``
     - ``fm.lss_single_trial`` or ``fm.estimate_single_trial(..., method="lss")``
   * - ``lsa()``
     - ``fmrimod.single.lsa_single_trial`` or ``method="lsa"``
   * - ``method = "oasis"``
     - ``fmrimod.single.oasis_single_trial`` or ``method="oasis"``
   * - ``lss_with_hrf()``, voxel-specific HRF helpers
     - ``fmrimod.single.estimate_voxel_hrf`` and ``lss_with_voxel_hrf``
   * - SBHM helpers
     - ``fmrimod.single.sbhm`` and ``method="sbhm"``
   * - ITEM helpers
     - ``fmrimod.single.item_*`` functions

Intentional Differences
-----------------------

``fmrimod`` does not expose every R helper as a top-level function. Internal R
helpers, C++ entry points, and script-oriented wrappers map to Python modules or
are treated as implementation details. Python result objects carry metadata
instead of encoding row and column names on bare matrices.

The lower-level Python LSS default does not silently add an intercept. Use
``include_intercept=True`` when reproducing the default ``fmrilss::lss``
contract exactly.
