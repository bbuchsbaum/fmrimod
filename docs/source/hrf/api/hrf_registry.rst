.. _api_hrf_registry:

HRF Registry
=============

The registry is a name-to-HRF lookup table.  Use :func:`get_hrf` to
retrieve an HRF by name, :func:`list_available_hrfs` to discover what is
registered, and :func:`register_hrf` / :func:`remove_hrf` to extend or
modify the registry at runtime.

.. autofunction:: fmrimod.hrf.registry.get_hrf

.. autofunction:: fmrimod.hrf.registry.list_available_hrfs

.. autofunction:: fmrimod.hrf.registry.register_hrf

.. autofunction:: fmrimod.hrf.registry.remove_hrf
