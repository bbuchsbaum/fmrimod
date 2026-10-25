"""Design matrix column metadata."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd


# Patterns for parsing the user-visible name out of a baseline-term column.
# - Nuisance columns are emitted as ``nuis_run{K}_{user_name}``.
# - Drift columns are emitted as ``base_{basis}{degree}_block_{K}``.
# - Block (intercept) columns are emitted as ``base_constant{N}_block_{K}``
#   or ``constant_{K}`` / ``constant_global``.
_NUISANCE_NAME_PATTERN = re.compile(r"^nuis_run(?P<run>\d+)_(?P<name>.+)$")


def _split_baseline_column_name(
    name, *, term_role: str
) -> tuple[str | None, int | None]:
    """Recover (user_visible_name, run_id) from a realised baseline column name.

    For nuisance columns the realised name has the form
    ``nuis_run{K}_{user_name}``; we strip the prefix so typed lookups
    via ``cols.where(term=user_name)`` resolve directly without
    requiring suffix matching. For drift and block terms we return
    ``None`` for the user-visible name (they don't have one in the
    typed-spec sense — only ``term_role`` is meaningful).
    """
    if name is None:
        return None, None
    if term_role == "nuisance":
        match = _NUISANCE_NAME_PATTERN.match(str(name))
        if match is not None:
            return match.group("name"), int(match.group("run"))
        return str(name), None
    return None, None


_RUN_SUFFIX_PATTERN = re.compile(r"_\d+$")


def _baseline_column_runs(
    term: Any, term_role: str, n_cols: int
) -> list[int | None] | None:
    """Run (1-based) of each column of a baseline term, from its ``colind``.

    A column listed under exactly one block belongs to that run; a column
    listed under several blocks (a global intercept or constant drift) is
    shared and has no run (R fmridesign 1ecaf91/fa40860). With a single run
    the block bookkeeping cannot tell the two apart, so a drift or intercept
    column whose name carries no run suffix (``constant_global``,
    ``base_constant``) is treated as shared. ``None`` when ``colind`` does
    not cover the term's columns.
    """
    colind = getattr(term, "colind", None)
    if not colind:
        return None
    blocks_of: dict[int, list[int]] = {}
    for block, cols in enumerate(colind, start=1):
        for col in cols:
            blocks_of.setdefault(int(col), []).append(block)
    if set(blocks_of) != set(range(n_cols)):
        return None
    names = [str(c) for c in term.design_matrix.columns]
    runs: list[int | None] = []
    for col in range(n_cols):
        blocks = blocks_of[col]
        shared = len(blocks) > 1 or (
            len(colind) == 1
            and term_role in ("drift", "block")
            and _RUN_SUFFIX_PATTERN.search(names[col]) is None
        )
        runs.append(None if shared else blocks[0])
    return runs


def design_colmap(x: object) -> pd.DataFrame:
    """Get structured column metadata for a design matrix.

    Returns a DataFrame with one row per regressor column, providing
    rich metadata about each column's origin, role, basis function
    membership, and display properties.

    Parameters
    ----------
    x : EventModel or BaselineModel
        Model containing a design matrix.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the following columns:

        - ``col`` : int -- 1-based column index.
        - ``name`` : str -- Column name in the design matrix.
        - ``term_tag`` : str or None -- Term tag this column belongs to.
        - ``term_index`` : int or None -- 1-based term index.
        - ``condition`` : str -- Condition label (term tag stripped).
        - ``run`` : int or None -- Run of a per-run (block-diagonal)
          baseline column; ``None`` for columns shared by all runs.
        - ``role`` : str -- ``'task'``, ``'intercept'``, ``'drift'``,
          ``'confound'`` (nuisance regressors), or ``'baseline'``.
        - ``model_source`` : str -- ``'event'`` or ``'baseline'``.
        - ``basis_name`` : str or None -- For event columns, the HRF basis
          family as spelled in ``hrf(basis=...)`` (``'spmg3'``, ``'fir'``,
          ...), also when the HRF is lagged; other HRFs keep their name.
        - ``basis_ix`` : int or None -- 1-based basis index.
        - ``basis_total`` : int or None -- Total basis functions.
        - ``basis_label`` : str or None -- Human-readable basis label
          (e.g., ``'canonical'``, ``'derivative'``).
        - ``pretty_name`` : str -- Display-friendly column name.
        - ``is_block_diagonal`` : bool -- Whether the column belongs to a
          single run (``False`` for event columns and for baseline columns
          shared by all runs, such as a global intercept).
        - ``modulation_type`` : str or None -- Modulation type.
        - ``modulation_id`` : str or None -- Modulation identifier.

    Raises
    ------
    TypeError
        If ``x`` is not an ``EventModel`` or ``BaselineModel``.

    Examples
    --------
    >>> cmap = design_colmap(model)
    >>> cmap[['name', 'term_tag', 'role']].head()
       name     term_tag  role
    0  cond.A   cond      task
    1  cond.B   cond      task
    """
    from .design.event_model import EventModel

    if isinstance(x, EventModel):
        return _colmap_event_model(x)

    # Try baseline model
    try:
        from .baseline.baseline_model import BaselineModel
        if isinstance(x, BaselineModel):
            return _colmap_baseline_model(x)
    except ImportError:
        pass

    raise TypeError(f"design_colmap not implemented for {type(x)}")


def design_meta(x: object) -> pd.DataFrame:
    """Return first-class design column metadata.

    For :class:`~fmrimod.design.event_model.EventModel`, metadata is read from
    construction-time ``column_facts`` instead of reconstructing semantics from
    rendered column names.
    """
    from .design.event_model import EventModel

    if isinstance(x, EventModel):
        return _meta_event_model(x)

    try:
        from .baseline.baseline_model import BaselineModel

        if isinstance(x, BaselineModel):
            meta = _colmap_baseline_model(x)
            return meta.drop(columns=["pretty_name"])
    except ImportError:
        pass

    raise TypeError(f"design_meta not implemented for {type(x)}")


def _colmap_event_model(model: Any) -> pd.DataFrame:
    """Build column metadata DataFrame for an EventModel.

    Parameters
    ----------
    model : EventModel
        Event model with design matrix and column metadata.

    Returns
    -------
    pandas.DataFrame
        Column metadata (see :func:`design_colmap`).
    """
    meta = design_meta(model)
    if not meta.empty:
        colmap = meta.copy()
        loc = colmap.columns.get_loc("is_block_diagonal")
        if not isinstance(loc, int):
            raise TypeError(
                "is_block_diagonal must be a single column in the meta frame"
            )
        colmap.insert(loc, "pretty_name", colmap["name"])
        return colmap

    dm = model.design_matrix
    n_cols = dm.shape[1]

    if n_cols == 0:
        return _empty_colmap()

    cn = model.column_names
    col_inds = model.column_indices

    # Map each column to its term
    term_index_by_col: list[Optional[int]] = [None] * n_cols
    term_tag_by_col: list[Optional[str]] = [None] * n_cols

    if col_inds:
        for i, (tag, indices) in enumerate(col_inds.items()):
            for idx in indices:
                if idx < n_cols:
                    term_index_by_col[idx] = i + 1
                    term_tag_by_col[idx] = tag

    # Parse basis index from column names (pattern: _b1, _b2, etc.)
    basis_ix: list[Optional[int]] = [None] * n_cols
    for i, name in enumerate(cn):
        m = re.search(r'_b(\d+)$', name)
        if m:
            basis_ix[i] = int(m.group(1))

    # Extract condition from column name (remove term tag prefix and basis suffix)
    condition = []
    for i, name in enumerate(cn):
        c = re.sub(r'_b\d+$', '', name)  # Remove basis suffix
        tag = term_tag_by_col[i]
        if tag and c.startswith(tag + '_'):
            c = c[len(tag) + 1:]
        condition.append(c)

    # Basis info from HRF
    basis_name: list[Optional[str]] = [None] * n_cols
    basis_total: list[Optional[int]] = [None] * n_cols
    for i, term in enumerate(model.terms):
        if term.hrf is not None:
            try:
                from ._warnings import call_safely, suppress_fmrimod_warnings

                with suppress_fmrimod_warnings():
                    from .hrf import library as _hrf_library
                    from .hrf import registry as _hrf_registry

                if isinstance(term.hrf, str):
                    hrf_lower = term.hrf.lower()
                    if hrf_lower in ('spm', 'spm_canonical', 'canonical', 'spmg1'):
                        hrf_obj: Any = _hrf_library.SPM_CANONICAL
                    elif hrf_lower == 'spmg2':
                        hrf_obj = _hrf_library.SPM_WITH_DERIVATIVE
                    elif hrf_lower == 'spmg3':
                        hrf_obj = _hrf_library.SPM_WITH_DISPERSION
                    else:
                        hrf_obj = call_safely(_hrf_registry.get_hrf, hrf_lower)
                else:
                    hrf_obj = term.hrf

                nb = hrf_obj.nbasis
                bn = _hrf_basis_family(
                    getattr(hrf_obj, 'name', str(type(hrf_obj).__name__))
                )
            except Exception:
                nb = 1
                bn = None

            term_name = term.name
            if term_name in col_inds:
                for idx in col_inds[term_name]:
                    if idx < n_cols:
                        basis_name[idx] = bn
                        basis_total[idx] = nb

    # Basis labels
    basis_label: list[Optional[str]] = [None] * n_cols
    for i in range(n_cols):
        bix = basis_ix[i]
        if bix is None:
            continue
        basis_label[i] = _basis_label(basis_name[i], bix) or f"component_{bix:02d}"

    return pd.DataFrame({
        'col': range(1, n_cols + 1),
        'name': cn,
        'term_tag': term_tag_by_col,
        'term_index': term_index_by_col,
        'condition': condition,
        'run': [None] * n_cols,
        'role': ['task'] * n_cols,
        'model_source': ['event'] * n_cols,
        'basis_name': basis_name,
        'basis_ix': basis_ix,
        'basis_total': basis_total,
        'basis_label': basis_label,
        'pretty_name': cn,
        'is_block_diagonal': [False] * n_cols,
        'modulation_type': ['amplitude'] * n_cols,
        'modulation_id': [None] * n_cols,
    })


def _meta_event_model(model: Any) -> pd.DataFrame:
    """Build first-class event-model column metadata from column facts."""
    dm = model.design_matrix
    n_cols = dm.shape[1]
    if n_cols == 0:
        return _empty_colmap().drop(columns=["pretty_name"])

    facts = getattr(model, "column_facts", None)
    if not facts:
        return _empty_colmap().drop(columns=["pretty_name"])

    rows = []
    for fact in facts:
        basis_total = _int_or_none(fact.get("basis_total"))
        basis_ix = _int_or_none(fact.get("basis_ix"))
        basis_name = _str_or_none(fact.get("basis_name"))
        if basis_total is not None and basis_total <= 1:
            basis_total = None
            basis_ix = None
        rows.append(
            {
                "col": int(fact["index"]) + 1,
                "name": str(fact["name"]),
                "term_tag": _str_or_none(fact.get("term_tag")),
                "term_index": _int_or_none(fact.get("term_index")),
                "condition": _str_or_none(fact.get("condition")),
                "run": None,
                "role": str(fact.get("role") or "task"),
                "model_source": str(fact.get("model_source") or "event"),
                "basis_name": basis_name,
                "basis_ix": basis_ix,
                "basis_total": basis_total,
                "basis_label": _basis_label(basis_name, basis_ix),
                "is_block_diagonal": False,
                "modulation_type": "amplitude",
                "modulation_id": None,
            }
        )
    return pd.DataFrame(rows, columns=_META_COLUMNS)


_META_COLUMNS = [
    "col",
    "name",
    "term_tag",
    "term_index",
    "condition",
    "run",
    "role",
    "model_source",
    "basis_name",
    "basis_ix",
    "basis_total",
    "basis_label",
    "is_block_diagonal",
    "modulation_type",
    "modulation_id",
]


# Basis families named in lower case, as spelled in ``hrf(basis=...)``
# (R fmridesign ``.hrf_basis_families``).
_HRF_BASIS_FAMILIES = (
    "spmg1", "spmg2", "spmg3", "gamma", "gaussian",
    "bspline", "tent", "fir", "fourier", "daguerre",
)
_HRF_DECORATIONS = re.compile(r"(_lag\([^)]*\)|_block\([^)]*\)|_norm)+$")
_HRF_NBASIS_SUFFIX = re.compile(r"_N\d+$")


def _hrf_basis_family(name: str | None) -> str | None:
    """The ``basis_name`` recorded for an HRF (R ``.hrf_basis_name``).

    Lag, block and normalisation decorations are dropped, and a built-in
    basis is reported by its ``hrf(basis=...)`` family in lower case
    (``SPMG3_lag(2.0)`` -> ``spmg3``, ``fir_N3`` -> ``fir``). Other HRFs keep
    their own (undecorated) name.
    """
    if name is None:
        return None
    base = _HRF_DECORATIONS.sub("", str(name))
    family = _HRF_NBASIS_SUFFIX.sub("", base).lower()
    return family if family in _HRF_BASIS_FAMILIES else base


def _basis_label(basis_name: Optional[str], basis_ix: Optional[int]) -> Optional[str]:
    if basis_name is None or basis_ix is None:
        return None
    family = _hrf_basis_family(basis_name)
    if family == "spmg3":
        labels = ["canonical", "derivative", "dispersion"]
        return labels[min(max(int(basis_ix), 1), 3) - 1]
    if family == "spmg2":
        labels = ["canonical", "derivative"]
        return labels[min(max(int(basis_ix), 1), 2) - 1]
    if family == "fir":
        return f"lag_{int(basis_ix) - 1:02d}"
    return f"component_{int(basis_ix):02d}"


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return str(value)


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return int(value)


def _colmap_baseline_model(model: Any) -> pd.DataFrame:
    """Build column metadata DataFrame for a BaselineModel.

    Walks the model's typed term structure (``terms[role]`` for
    ``role in ('drift', 'block', 'nuisance')``) and assigns colmap
    roles directly from term identity rather than inferring from
    column-name substrings. This preserves the distinction between
    drift, intercept, and confound regressors, and recovers the
    user-visible confound name (without the ``"nuis_runK_"`` prefix)
    in the ``term_tag`` field so typed lookup via
    ``cols.where(term="trans_x")`` resolves the right column.

    Parameters
    ----------
    model : BaselineModel
        Baseline model with design matrix.

    Returns
    -------
    pandas.DataFrame
        Column metadata (see :func:`design_colmap`).
    """
    dm = model.design_matrix
    if isinstance(dm, pd.DataFrame):
        cn = list(dm.columns)
        X = dm.values
    else:
        X = np.asarray(dm)
        # Prefer the typed ``BaselineModel.column_names`` over synthesized
        # ``V{k}`` placeholders so the prefix-stripping that recovers the
        # user-visible nuisance name resolves the right token.
        model_names = getattr(model, "column_names", None)
        if model_names is not None and len(list(model_names)) == X.shape[1]:
            cn = list(model_names)
        else:
            cn = [f"V{i+1}" for i in range(X.shape[1])]

    n_cols = len(cn)
    if n_cols == 0:
        return _empty_colmap()

    # Walk the typed term structure to assign roles correctly.
    from fmrimod.baseline.baseline_model import BASELINE_TERM_ORDER

    role_map = {
        "drift": "drift",
        "block": "intercept",
        "nuisance": "confound",
    }

    roles: list[str] = []
    term_tags: list[str | None] = []
    runs: list[int | None] = []
    block_diag: list[bool] = []
    cursor = 0
    terms = getattr(model, "terms", {}) or {}
    for term_name in BASELINE_TERM_ORDER:
        term = terms.get(term_name)
        if term is None:
            continue
        mat = term.design_matrix
        if isinstance(mat, pd.DataFrame):
            block_cols = mat.shape[1]
        else:
            block_cols = int(np.asarray(mat).shape[1])
        if block_cols == 0:
            continue
        block_role = role_map.get(term_name, "baseline")
        col_runs = _baseline_column_runs(term, term_name, block_cols)
        for local in range(block_cols):
            full_name = cn[cursor + local] if cursor + local < len(cn) else None
            roles.append(block_role)
            user_name, name_run = _split_baseline_column_name(
                full_name, term_role=term_name
            )
            term_tags.append(user_name)
            run_id = col_runs[local] if col_runs is not None else name_run
            runs.append(run_id)
            block_diag.append(run_id is not None)
        cursor += block_cols

    # Any trailing columns (defensive — should not normally happen)
    # get the generic ``baseline`` role + no term tag.
    while cursor < n_cols:
        roles.append("baseline")
        term_tags.append(None)
        runs.append(None)
        block_diag.append(False)
        cursor += 1

    return pd.DataFrame({
        'col': range(1, n_cols + 1),
        'name': cn,
        'term_tag': term_tags,
        'term_index': [None] * n_cols,
        'condition': [None] * n_cols,
        'run': runs,
        'role': roles,
        'model_source': ['baseline'] * n_cols,
        'basis_name': [None] * n_cols,
        'basis_ix': [None] * n_cols,
        'basis_total': [None] * n_cols,
        'basis_label': [None] * n_cols,
        'pretty_name': cn,
        'is_block_diagonal': block_diag,
        'modulation_type': [None] * n_cols,
        'modulation_id': [None] * n_cols,
    })


def _empty_colmap() -> pd.DataFrame:
    """Return an empty column map DataFrame with the correct schema."""
    return pd.DataFrame({
        'col': pd.Series(dtype=int),
        'name': pd.Series(dtype=str),
        'term_tag': pd.Series(dtype=str),
        'term_index': pd.Series(dtype=int),
        'condition': pd.Series(dtype=str),
        'run': pd.Series(dtype=int),
        'role': pd.Series(dtype=str),
        'model_source': pd.Series(dtype=str),
        'basis_name': pd.Series(dtype=str),
        'basis_ix': pd.Series(dtype=int),
        'basis_total': pd.Series(dtype=int),
        'basis_label': pd.Series(dtype=str),
        'pretty_name': pd.Series(dtype=str),
        'is_block_diagonal': pd.Series(dtype=bool),
        'modulation_type': pd.Series(dtype=str),
        'modulation_id': pd.Series(dtype=str),
    })
