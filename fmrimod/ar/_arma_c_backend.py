"""Optional native C backend for segment-aware ARMA whitening."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from typing import Optional

import numpy as np
from numpy.typing import NDArray

_LIB: Optional[ctypes.CDLL] = None
_FUNC = None
_FAILED = False


def _source_path() -> Path:
    return Path(__file__).with_name("arma_filter.c")


def _cache_dir() -> Path:
    root = os.environ.get("FMRIMOD_C_BACKEND_CACHE")
    if root:
        p = Path(root)
    else:
        p = Path.home() / ".cache" / "fmrimod"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _artifact_path() -> Path:
    src = _source_path()
    key = hashlib.sha256(
        (
            src.read_text(encoding="utf-8")
            + f"|py={sys.version_info[:3]}|plat={sys.platform}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return _cache_dir() / f"arma_filter_{key}{suffix}"


def _packaged_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    suffixes = []
    if ext_suffix:
        suffixes.append(ext_suffix)
    suffixes.extend([".so", ".pyd", ".dylib", ".dll"])
    seen = set()
    out: list[Path] = []
    for suf in suffixes:
        if suf in seen:
            continue
        seen.add(suf)
        out.append(here / f"_arma_filter{suf}")
    return out


def _compile_shared(target: Path) -> bool:
    src = _source_path()
    if not src.exists():
        return False

    cc_raw = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    cc = shlex.split(cc_raw)
    if not cc:
        cc = ["cc"]

    cmd = [*cc, "-O3", "-fPIC", "-shared", str(src), "-o", str(target)]

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        log_path = Path(tf.name)
    try:
        with log_path.open("wb") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=log, check=False)
        return proc.returncode == 0 and target.exists()
    except Exception:
        return False
    finally:
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass


def _load_func():
    global _LIB, _FUNC, _FAILED
    if _FUNC is not None:
        return _FUNC
    if _FAILED:
        return None

    targets = [p for p in _packaged_candidates() if p.exists()]
    if not targets:
        target = _artifact_path()
        if not target.exists():
            ok = _compile_shared(target)
            if not ok:
                _FAILED = True
                return None
        targets = [target]

    for target in targets:
        try:
            _LIB = ctypes.CDLL(str(target))
            break
        except Exception:
            _LIB = None
    if _LIB is None:
        _FAILED = True
        _FUNC = None
        return None

    try:
        fn = _LIB.arma_whiten_segments_c
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # y
            ctypes.c_int,  # n
            ctypes.c_int,  # v
            ctypes.POINTER(ctypes.c_double),  # phi
            ctypes.c_int,  # p
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.c_int,  # q
            ctypes.POINTER(ctypes.c_int),  # seg_starts
            ctypes.c_int,  # n_seg
            ctypes.c_int,  # do_exact
            ctypes.POINTER(ctypes.c_double),  # out
        ]
        fn.restype = ctypes.c_int
        _FUNC = fn
        return _FUNC
    except Exception:
        _FAILED = True
        _FUNC = None
        _LIB = None
        return None


def arma_whiten_segments_c(
    y: NDArray,
    phi: NDArray,
    theta: NDArray,
    seg_starts: NDArray,
    do_exact: bool,
) -> Optional[NDArray]:
    """Run C backend whitening. Returns None when backend is unavailable."""
    fn = _load_func()
    if fn is None:
        return None

    y_c = np.ascontiguousarray(y, dtype=np.float64)
    phi_c = np.ascontiguousarray(phi, dtype=np.float64).ravel()
    theta_c = np.ascontiguousarray(theta, dtype=np.float64).ravel()
    starts_c = np.ascontiguousarray(seg_starts, dtype=np.int32).ravel()

    if y_c.ndim == 1:
        y_c = y_c[:, np.newaxis]

    n, v = y_c.shape
    out = np.empty_like(y_c)

    rc = fn(
        y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(n),
        int(v),
        phi_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(phi_c.shape[0]),
        theta_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(theta_c.shape[0]),
        starts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        int(starts_c.shape[0]),
        int(bool(do_exact)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        return None

    return out
