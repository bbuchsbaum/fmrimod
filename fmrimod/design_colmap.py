"""Design matrix column metadata."""
import re
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


def design_colmap(x):
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
        - ``run`` : int or None -- Run index (if block-diagonal).
        - ``role`` : str -- ``'task'``, ``'intercept'``, ``'drift'``,
          ``'nuisance'``, or ``'baseline'``.
        - ``model_source`` : str -- ``'event'`` or ``'baseline'``.
        - ``basis_name`` : str or None -- HRF/basis name.
        - ``basis_ix`` : int or None -- 1-based basis index.
        - ``basis_total`` : int or None -- Total basis functions.
        - ``basis_label`` : str or None -- Human-readable basis label
          (e.g., ``'canonical'``, ``'derivative'``).
        - ``pretty_name`` : str -- Display-friendly column name.
        - ``is_block_diagonal`` : bool -- Whether the column is
          block-diagonal (non-zero in only one run).
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


def _colmap_event_model(model):
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
    dm = model.design_matrix
    n_cols = dm.shape[1]

    if n_cols == 0:
        return _empty_colmap()

    cn = model.column_names
    col_inds = model.column_indices

    # Map each column to its term
    term_index_by_col = [None] * n_cols
    term_tag_by_col = [None] * n_cols

    if col_inds:
        for i, (tag, indices) in enumerate(col_inds.items()):
            for idx in indices:
                if idx < n_cols:
                    term_index_by_col[idx] = i + 1
                    term_tag_by_col[idx] = tag

    # Parse basis index from column names (pattern: _b1, _b2, etc.)
    basis_ix = [None] * n_cols
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
    basis_name = [None] * n_cols
    basis_total = [None] * n_cols
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
                        hrf_obj = _hrf_library.SPM_CANONICAL
                    elif hrf_lower == 'spmg2':
                        hrf_obj = _hrf_library.SPM_WITH_DERIVATIVE
                    elif hrf_lower == 'spmg3':
                        hrf_obj = _hrf_library.SPM_WITH_DISPERSION
                    else:
                        hrf_obj = call_safely(_hrf_registry.get_hrf, hrf_lower)
                else:
                    hrf_obj = term.hrf

                nb = hrf_obj.nbasis
                bn = getattr(hrf_obj, 'name', str(type(hrf_obj).__name__))
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
    basis_label = [None] * n_cols
    for i in range(n_cols):
        bix = basis_ix[i]
        bn = basis_name[i]
        if bix is None:
            continue
        if bn and 'SPMG3' in str(bn):
            labels = ['canonical', 'derivative', 'dispersion']
            basis_label[i] = labels[min(bix - 1, 2)]
        elif bn and 'SPMG2' in str(bn):
            labels = ['canonical', 'derivative']
            basis_label[i] = labels[min(bix - 1, 1)]
        elif bn and 'FIR' in str(bn):
            basis_label[i] = f"lag_{bix-1:02d}"
        else:
            basis_label[i] = f"component_{bix:02d}"

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


def _colmap_baseline_model(model):
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
        for local in range(block_cols):
            full_name = cn[cursor + local] if cursor + local < len(cn) else None
            roles.append(block_role)
            user_name, run_id = _split_baseline_column_name(
                full_name, term_role=term_name
            )
            term_tags.append(user_name)
            runs.append(run_id)
        cursor += block_cols

    # Any trailing columns (defensive — should not normally happen)
    # get the generic ``baseline`` role + no term tag.
    while cursor < n_cols:
        roles.append("baseline")
        term_tags.append(None)
        runs.append(None)
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
        'is_block_diagonal': [True] * n_cols,
        'modulation_type': [None] * n_cols,
        'modulation_id': [None] * n_cols,
    })


def _empty_colmap():
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
