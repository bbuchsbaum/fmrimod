"""Dataset adapters for various neuroimaging formats."""

from .backend_adapter import BackendAdapter
from .numpy_adapter import NumpyAdapter

__all__ = ["BackendAdapter", "NumpyAdapter"]
