"""Dataset adapters for various neuroimaging formats.

Adapters wrap external data sources into the
:class:`fmrimod.dataset.protocols.DatasetProtocol` shape so that the GLM
pipeline can consume them uniformly.

- :class:`NumpyAdapter` — raw 2-D matrices.
- :class:`BackendAdapter` — wrap a :class:`StorageBackend`.
- :class:`NeuroVecAdapter` — canonical NIfTI path via ``neuroim``.
- :class:`NibabelAdapter` — compatibility shim for ``nibabel.Nifti1Image``
  callers; new code should prefer :class:`NeuroVecAdapter`.
"""

from .backend_adapter import BackendAdapter
from .neuroim_adapter import NeuroVecAdapter
from .nibabel_adapter import NibabelAdapter
from .numpy_adapter import NumpyAdapter

__all__ = ["BackendAdapter", "NeuroVecAdapter", "NibabelAdapter", "NumpyAdapter"]
