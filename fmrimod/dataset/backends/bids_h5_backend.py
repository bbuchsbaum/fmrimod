"""BIDS-HDF5 scan backend."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import BackendIOError, ConfigError

CompressionMode = Literal["parcellated", "latent"]


def _decode_h5_value(value: object) -> object:
    """Convert common HDF5 scalar/string values into Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_h5_value(value.item())
        if value.dtype.kind in {"S", "O", "U"}:
            return [_decode_h5_value(v) for v in value.tolist()]
    return value


def _as_index_array(
    value: NDArray[np.intp] | int | list[int] | tuple[int, ...] | None,
    upper: int,
    name: str,
) -> NDArray[np.intp] | None:
    if value is None:
        return None
    arr = np.atleast_1d(np.asarray(value))
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} indices must be integers")
    out: NDArray[np.intp] = arr.astype(np.intp, copy=False)
    if np.any(out < 0) or np.any(out >= upper):
        raise ValueError(f"{name} indices must be within [0, {upper - 1}]")
    return out


class SharedH5Connection:
    """Reference-counted HDF5 file connection shared by scan backends."""

    def __init__(self, file: str | Path) -> None:
        self.file = Path(file)
        self.ref_count = 0
        self._handle: object | None = None
        self._open_handle()

    def _open_handle(self) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ConfigError(
                "h5py is required for BIDS-HDF5 support. "
                "Install with: pip install fmrimod[hdf5]",
                parameter="h5py",
            ) from exc

        if not self.file.exists():
            raise BackendIOError(
                f"BIDS-HDF5 file not found: {self.file}",
                file=str(self.file),
                operation="open",
            )

        try:
            self._handle = h5py.File(str(self.file), "r")
        except OSError as exc:
            raise BackendIOError(
                f"Failed to open BIDS-HDF5 file '{self.file}': {exc}",
                file=str(self.file),
                operation="open",
            ) from exc

    @property
    def handle(self) -> object:
        if self._handle is None or not bool(self._handle.id.valid):
            raise BackendIOError(
                f"HDF5 file handle for '{self.file}' is not open",
                file=str(self.file),
                operation="read",
            )
        return self._handle

    @property
    def is_valid(self) -> bool:
        return self._handle is not None and bool(self._handle.id.valid)

    def acquire(self) -> None:
        if not self.is_valid:
            self._open_handle()
        self.ref_count += 1

    def release(self) -> None:
        self.ref_count = max(0, self.ref_count - 1)
        if self.ref_count == 0 and self._handle is not None and self.is_valid:
            self._handle.close()

    def close(self) -> None:
        self.ref_count = 0
        if self._handle is not None and self.is_valid:
            self._handle.close()

    def __repr__(self) -> str:
        return (
            f"<SharedH5Connection file={str(self.file)!r} "
            f"ref_count={self.ref_count} is_valid={self.is_valid}>"
        )


class BidsH5ScanBackend(StorageBackend):
    """Backend for one scan stored inside a BIDS-HDF5 archive."""

    def __init__(
        self,
        h5_connection: SharedH5Connection,
        scan_group_path: str,
        n_features: int,
        n_time: int,
        metadata: dict[str, object] | None = None,
        compression_mode: CompressionMode = "parcellated",
    ) -> None:
        if not isinstance(h5_connection, SharedH5Connection):
            raise ConfigError(
                "h5_connection must be a SharedH5Connection",
                parameter="h5_connection",
            )
        if compression_mode not in ("parcellated", "latent"):
            raise ConfigError(
                "compression_mode must be 'parcellated' or 'latent'",
                parameter="compression_mode",
                value=compression_mode,
            )
        if n_features < 1:
            raise ConfigError(
                "n_features must be a positive integer",
                parameter="n_features",
                value=n_features,
            )
        if n_time < 1:
            raise ConfigError(
                "n_time must be a positive integer",
                parameter="n_time",
                value=n_time,
            )

        self.h5_connection = h5_connection
        self.scan_group_path = scan_group_path.rstrip("/")
        self.n_features = int(n_features)
        self.n_time = int(n_time)
        self.metadata = metadata or {}
        self.compression_mode: CompressionMode = compression_mode
        self.is_open = False

    def open(self) -> None:
        if not self.is_open:
            self.h5_connection.acquire()
            self.is_open = True

    def close(self) -> None:
        if self.is_open:
            self.h5_connection.release()
            self.is_open = False

    def get_dims(self) -> BackendDims:
        return BackendDims(spatial=(self.n_features, 1, 1), time=self.n_time)

    def get_mask(self) -> NDArray[np.bool_]:
        return np.ones(self.n_features, dtype=np.bool_)

    @property
    def _data_path(self) -> str:
        dataset = "basis" if self.compression_mode == "latent" else "summary_data"
        return f"{self.scan_group_path}/data/{dataset}"

    def get_data(
        self,
        rows: NDArray[np.intp] | int | list[int] | tuple[int, ...] | None = None,
        cols: NDArray[np.intp] | int | list[int] | tuple[int, ...] | None = None,
    ) -> NDArray[np.float64]:
        if not self.is_open:
            self.open()

        handle = self.h5_connection.handle
        path = self._data_path
        if path not in handle:
            raise BackendIOError(
                f"Data dataset '{path}' not found in {self.h5_connection.file}",
                file=str(self.h5_connection.file),
                operation="read",
            )

        try:
            data = np.asarray(handle[path], dtype=np.float64)
        except Exception as exc:
            raise BackendIOError(
                f"Failed to read data from '{path}': {exc}",
                file=str(self.h5_connection.file),
                operation="read",
            ) from exc

        if data.ndim != 2:
            data = data.reshape(self.n_time, self.n_features)

        row_idx = _as_index_array(rows, data.shape[0], "rows")
        col_idx = _as_index_array(cols, data.shape[1], "cols")
        if row_idx is not None:
            data = data[row_idx, :]
        if col_idx is not None:
            data = data[:, col_idx]
        return data

    def get_metadata(self) -> dict[str, object]:
        meta = dict(self.metadata)
        meta.update(
            {
                "compression_mode": self.compression_mode,
                "n_features": self.n_features,
                "scan_group_path": self.scan_group_path,
                "format": "bids_h5",
            }
        )
        return meta


def h5_shared_connection(file: str | Path) -> SharedH5Connection:
    """Create a shared BIDS-HDF5 file connection."""
    return SharedH5Connection(file)


def bids_h5_scan_backend(
    h5_connection: SharedH5Connection,
    scan_group_path: str,
    n_features: int,
    n_time: int,
    metadata: dict[str, object] | None = None,
    compression_mode: CompressionMode = "parcellated",
) -> BidsH5ScanBackend:
    """Construct a :class:`BidsH5ScanBackend`."""
    return BidsH5ScanBackend(
        h5_connection=h5_connection,
        scan_group_path=scan_group_path,
        n_features=n_features,
        n_time=n_time,
        metadata=metadata,
        compression_mode=compression_mode,
    )


__all__ = [
    "BidsH5ScanBackend",
    "CompressionMode",
    "SharedH5Connection",
    "bids_h5_scan_backend",
    "h5_shared_connection",
]
