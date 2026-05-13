"""Canonical HRF names and the alias normalizer.

The closed set of HRF kinds available via :func:`fmrimod.hrf.get_hrf` is
expressed as a ``Literal`` so type-checkers can pin call sites. A
single normalizer collapses the legacy aliases (``"spm"`` ->
``"spmg1"``, ``"spm_canonical"`` -> ``"spmg1"``, etc.) into the
canonical names. Mirrors the precedent in
``fmrimod/stats/normalize.py``.

See bead ``bd-01KRGCZD8QC6W4XE73JW0DJK5A``.
"""

from __future__ import annotations

from typing import Literal

HRFName = Literal[
    "simple",
    "spmg1",
    "spmg2",
    "spmg3",
    "gamma",
    "gaussian",
    "bspline",
    "fir",
    "fourier",
    "time",
    "mexhat",
    "inv_logit",
    "half_cosine",
    "sine",
    "lwu",
    "lwu_basis",
    "daguerre",
    "tent",
    "boxcar",
    "weighted",
]


# Maps every accepted public name (case-insensitive) to its canonical form.
# Generator-only registry entries with the ``hrf_`` prefix (e.g.
# ``hrf_bspline``) are routed through this table too — they alias to the
# same canonical kind as their non-prefixed counterpart.
_ALIASES: dict[str, HRFName] = {
    # Identity entries
    "simple": "simple",
    "spmg1": "spmg1",
    "spmg2": "spmg2",
    "spmg3": "spmg3",
    "gamma": "gamma",
    "gaussian": "gaussian",
    "bspline": "bspline",
    "fir": "fir",
    "fourier": "fourier",
    "time": "time",
    "mexhat": "mexhat",
    "inv_logit": "inv_logit",
    "half_cosine": "half_cosine",
    "sine": "sine",
    "lwu": "lwu",
    "lwu_basis": "lwu_basis",
    # SPM canonical synonyms
    "spm": "spmg1",
    "spm_canonical": "spmg1",
    # Generator-prefixed names
    "hrf_gamma": "gamma",
    "hrf_bspline": "bspline",
    "hrf_fir": "fir",
    "hrf_fourier": "fourier",
    "hrf_daguerre": "daguerre",
    "daguerre": "daguerre",
    "hrf_tent": "tent",
    "tent": "tent",
    "hrf_boxcar": "boxcar",
    "boxcar": "boxcar",
    "hrf_weighted": "weighted",
    "weighted": "weighted",
}


def _normalize_hrf_name(name: str) -> HRFName:
    """Collapse an alias to its canonical HRF name.

    Parameters
    ----------
    name
        Free-form HRF name supplied by a user. Whitespace is stripped
        and matching is case-insensitive.

    Returns
    -------
    str
        The canonical name; this string is the key under which the
        target HRF lives in :data:`fmrimod.hrf.registry._HRF_REGISTRY`.

    Raises
    ------
    ValueError
        If ``name`` is not in :data:`_ALIASES`.
    """
    key = name.strip().lower()
    out = _ALIASES.get(key)
    if out is None:
        raise ValueError(
            f"HRF name {name!r} not found in registry or alias table. "
            f"Available: {sorted(set(_ALIASES.values()))}."
        )
    return out
