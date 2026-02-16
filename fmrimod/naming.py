"""Naming utilities for design matrix column names.

This module implements the naming scheme from the R fmridesign package,
ensuring consistent and R-compatible column names.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def zeropad(i: int, n_total: int) -> str:
    """Zero-pad an integer for consistent display width.

    The width is determined by the number of digits in ``n_total``,
    with a minimum width of 2 when ``n_total > 1``.

    Parameters
    ----------
    i : int
        Number to pad.
    n_total : int
        Total count (determines padding width).

    Returns
    -------
    str
        Zero-padded string representation of ``i``.

    Examples
    --------
    >>> zeropad(3, 100)
    '003'
    >>> zeropad(1, 5)
    '01'
    """
    if n_total < 1:
        width = 1
    else:
        log_width = len(str(n_total))
        width = max(2, log_width) if n_total > 1 else log_width
    
    return f"{i:0{width}d}"


def sanitize(x: str, allow_dot: bool = True) -> str:
    """Sanitize a string for use as a valid R/Python identifier.

    Replaces invalid characters with underscores, prepends ``'X'``
    if the string starts with a digit or underscore, and optionally
    replaces dots. Similar to R's ``make.names()``.

    Parameters
    ----------
    x : str
        String to sanitize.
    allow_dot : bool, optional
        If False, replace dots with underscores and collapse
        consecutive underscores. Default is True.

    Returns
    -------
    str
        Sanitized string safe for use as a column name.

    Examples
    --------
    >>> sanitize("my variable!")
    'my_variable_'
    >>> sanitize("123abc")
    'X123abc'
    """
    # Replace invalid characters
    # Similar to R's make.names()
    sanitized = re.sub(r'[^a-zA-Z0-9._]', '_', x)
    
    # Ensure doesn't start with digit or underscore
    if sanitized and (sanitized[0].isdigit() or sanitized[0] == '_'):
        sanitized = 'X' + sanitized
    
    # Handle dots
    if not allow_dot:
        sanitized = sanitized.replace('.', '_')
        # Replace multiple underscores with single
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
    
    # Ensure not empty
    if not sanitized:
        sanitized = 'X'
    
    return sanitized


def sanitize_level(level: str) -> str:
    """Sanitize a factor level name.

    Like :func:`sanitize` but strips the leading ``'X'`` that
    would be prepended to levels starting with a digit, preserving
    numeric level names like ``"1"`` or ``"2a"``.

    Parameters
    ----------
    level : str
        Factor level string to sanitize.

    Returns
    -------
    str
        Sanitized level string.

    Examples
    --------
    >>> sanitize_level("1")
    '1'
    >>> sanitize_level("face")
    'face'
    """
    sanitized = sanitize(level, allow_dot=True)
    
    # If original started with digit and sanitized starts with X, remove it
    if level and level[0].isdigit() and sanitized.startswith('X'):
        sanitized = sanitized[1:]
    
    return sanitized


def basis_suffix(j: int, nb: int) -> str:
    """Create a zero-padded basis function suffix.

    Parameters
    ----------
    j : int
        Basis index (1-based).
    nb : int
        Total number of basis functions (determines padding width).

    Returns
    -------
    str
        Suffix string, e.g., ``'_b01'``, ``'_b02'``.

    Examples
    --------
    >>> basis_suffix(1, 3)
    '_b01'
    >>> basis_suffix(10, 15)
    '_b10'
    """
    return f"_b{zeropad(j, nb)}"


def feature_suffix(j: int, nf: int) -> str:
    """Create a zero-padded feature suffix.

    Parameters
    ----------
    j : int
        Feature index (1-based).
    nf : int
        Total number of features (determines padding width).

    Returns
    -------
    str
        Suffix string, e.g., ``'f01'``, ``'f02'``.

    Examples
    --------
    >>> feature_suffix(2, 5)
    'f02'
    """
    return f"f{zeropad(j, nf)}"


def make_unique_tags(tags: List[str]) -> List[str]:
    """Ensure all tags are unique by appending ``#N`` suffixes.

    Duplicate tags receive a ``#0``, ``#1``, ... suffix. The first
    occurrence of each tag is left unchanged.

    Parameters
    ----------
    tags : list of str
        Tags that may contain duplicates.

    Returns
    -------
    list of str
        Tags with duplicates disambiguated.

    Examples
    --------
    >>> make_unique_tags(['a', 'b', 'a'])
    ['a', 'b', 'a#1']
    """
    seen = {}
    result = []
    
    for tag in tags:
        if tag in seen:
            seen[tag] += 1
            result.append(f"{tag}#{seen[tag] - 1}")
        else:
            seen[tag] = 1
            result.append(tag)
    
    return result


def make_term_tag(
    term_name: Optional[str] = None,
    event_names: Optional[List[str]] = None,
    hrf_name: Optional[str] = None,
    basis_type: Optional[str] = None,
    existing_tags: Optional[List[str]] = None
) -> Optional[str]:
    """Generate a sanitized, unique term tag for column naming.

    Produces a tag from an explicit term name, event names, or a
    combination of basis type and event name. When ``existing_tags``
    is provided, ensures the result is unique via ``#N`` suffixes.

    Parameters
    ----------
    term_name : str, optional
        Explicit term name/id. Used directly if provided.
    event_names : list of str, optional
        Event variable names. Joined with ``'_'`` if no
        ``term_name`` is given.
    hrf_name : str, optional
        HRF name (currently unused, reserved for future use).
    basis_type : str, optional
        Basis type (e.g., ``'poly'``, ``'spline'``). Prepended
        to the event name for single-event basis terms.
    existing_tags : list of str, optional
        Already-used tags; the result is made unique against
        this list.

    Returns
    -------
    str or None
        Sanitized term tag, or None for identity terms.

    Examples
    --------
    >>> make_term_tag(event_names=['condition'])
    'condition'
    >>> make_term_tag(event_names=['rt'], basis_type='poly')
    'poly_rt'
    """
    if term_name:
        # Explicit name provided
        tag_base = sanitize(term_name, allow_dot=False)
    elif event_names:
        # Build from event names
        if basis_type and len(event_names) == 1:
            # Special handling for basis functions
            var_name = sanitize(event_names[0], allow_dot=False)
            basis_prefix = {
                'poly': 'poly',
                'polynomial': 'poly',
                'bspline': 'bs',
                'spline': 'bs',
                'scale': 'z',
                'standardized': 'std',
                'robustscale': 'robz'
            }.get(basis_type.lower(), basis_type.lower())
            tag_base = f"{basis_prefix}_{var_name}"
        else:
            # Join event names
            sanitized_names = [sanitize(name, allow_dot=False) for name in event_names]
            tag_base = '_'.join(sanitized_names)
    else:
        # No information - return None (identity term)
        return None
    
    # Ensure not empty
    if not tag_base:
        tag_base = "empty_id_tag"
    
    # Make unique if needed
    if existing_tags:
        all_tags = existing_tags + [tag_base]
        unique_tags = make_unique_tags(all_tags)
        return unique_tags[-1]
    
    return tag_base


def level_token(var: str, level: str) -> str:
    """Create a ``variable.level`` token for a factor level.

    Sanitizes both the variable and level names before joining
    them with a dot separator. This is the canonical token format
    used throughout the naming system.

    Parameters
    ----------
    var : str
        Variable (factor) name.
    level : str
        Level name.

    Returns
    -------
    str
        Token in ``'var.level'`` format.

    Examples
    --------
    >>> level_token('condition', 'face')
    'condition.face'
    """
    s_var = sanitize(var, allow_dot=True)
    s_level = sanitize_level(level)
    return f"{s_var}.{s_level}"


def continuous_token(colname: str) -> str:
    """Sanitize a continuous variable's column name.

    Parameters
    ----------
    colname : str
        Proposed column name for the continuous variable.

    Returns
    -------
    str
        Sanitized token suitable for use in design matrix
        column names.

    Examples
    --------
    >>> continuous_token('response time')
    'response_time'
    """
    return sanitize(colname, allow_dot=True)


def make_cond_tag(tokens: List[str]) -> str:
    """Combine multiple tokens into a single condition tag.

    Joins tokens with ``'_'`` to form a composite condition name
    for interaction terms.

    Parameters
    ----------
    tokens : list of str
        Individual tokens (e.g., from :func:`level_token`).

    Returns
    -------
    str
        Combined condition tag.

    Examples
    --------
    >>> make_cond_tag(['condition.face', 'emotion.happy'])
    'condition.face_emotion.happy'
    """
    return '_'.join(tokens)


def add_basis_suffix(cond_tags: List[str], nb: int) -> List[str]:
    """Expand condition tags with basis function suffixes.

    When ``nb > 1``, each condition tag is replicated ``nb`` times
    with ``_b01``, ``_b02``, ... suffixes appended. When ``nb <= 1``,
    the tags are returned unchanged.

    Parameters
    ----------
    cond_tags : list of str
        Base condition tags (without basis suffix).
    nb : int
        Number of basis functions.

    Returns
    -------
    list of str
        Expanded tags. Length is ``len(cond_tags) * max(nb, 1)``.

    Examples
    --------
    >>> add_basis_suffix(['cond.A', 'cond.B'], nb=2)
    ['cond.A_b01', 'cond.B_b01', 'cond.A_b02', 'cond.B_b02']
    >>> add_basis_suffix(['cond.A'], nb=1)
    ['cond.A']
    """
    if nb <= 1:
        return cond_tags
    
    # Expand in basis-major order to match fmridesign:
    # [cond1_b01, cond2_b01, ..., cond1_b02, cond2_b02, ...]
    result = []
    for j in range(1, nb + 1):
        for tag in cond_tags:
            result.append(tag + basis_suffix(j, nb))
    
    return result


def make_column_names(
    term_tag: Optional[str],
    cond_tags: List[str],
    nb: int = 1
) -> List[str]:
    """Compose final design matrix column names.

    This is the single source of truth for creating design matrix
    column names. It combines a term tag prefix with condition tags
    and optional basis suffixes.

    Parameters
    ----------
    term_tag : str or None
        Sanitized, unique term tag. If None, condition tags are
        used directly (identity terms).
    cond_tags : list of str
        Base condition tags (without basis suffix).
    nb : int, default=1
        Number of basis functions. When > 1, basis suffixes
        (``_b01``, ``_b02``, ...) are appended.

    Returns
    -------
    list of str
        Final column names. Length is
        ``len(cond_tags) * max(nb, 1)``.

    Examples
    --------
    >>> make_column_names('cond', ['face', 'house'], nb=1)
    ['cond_face', 'cond_house']
    >>> make_column_names(None, ['face', 'house'], nb=1)
    ['face', 'house']

    See Also
    --------
    make_term_tag : Generate the term tag.
    add_basis_suffix : Add basis function suffixes.
    """
    # Add basis suffix if needed
    full_cond_tags = add_basis_suffix(cond_tags, nb)
    
    # Combine term tag and condition tags
    if term_tag is None:
        # Identity term - use condition tags directly
        return full_cond_tags
    else:
        # Normal case: term_tag_condition_tag[_b##]
        return [f"{term_tag}_{tag}" for tag in full_cond_tags]


def make_unique_colnames(colnames: List[str]) -> List[str]:
    """Ensure column names are unique using R-style suffixes.

    Duplicate names receive ``.1``, ``.2``, ... suffixes. The first
    occurrence of each name is left unchanged.

    Parameters
    ----------
    colnames : list of str
        Column names that may contain duplicates.

    Returns
    -------
    list of str
        Unique column names.

    Examples
    --------
    >>> make_unique_colnames(['a', 'b', 'a', 'a'])
    ['a', 'b', 'a.1', 'a.2']
    """
    seen = {}
    result = []
    
    for name in colnames:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    
    return result


def shortnames(longnames_list: List[str], acronym: Optional[str] = None) -> List[str]:
    """Convert long column names to abbreviated short names.

    Extracts the term-tag prefix from each name and replaces it
    with a shorter acronym derived from capital letters or first
    characters. Useful for AFNI-compatible labels or compact displays.

    Parameters
    ----------
    longnames_list : list of str
        Long column names to shorten (as produced by
        :func:`make_column_names`).
    acronym : str, optional
        Custom acronym to use for the term prefix. If None,
        an acronym is auto-generated from capital letters or
        first characters of each word.

    Returns
    -------
    list of str
        Shortened column names.

    Examples
    --------
    >>> shortnames(['condition_face_b01', 'condition_house_b01'])
    ['C_face_b01', 'C_house_b01']
    """
    if not longnames_list:
        return []
    
    # Extract unique term tags (before underscores)
    term_tags = set()
    for name in longnames_list:
        parts = name.split('_')
        if len(parts) > 1:
            term_tags.add(parts[0])
    
    # Create short version of each term tag
    term_map = {}
    if acronym:
        # Use provided acronym
        for i, tag in enumerate(sorted(term_tags)):
            term_map[tag] = f"{acronym}{i+1}" if len(term_tags) > 1 else acronym
    else:
        # Generate from first letters
        for tag in term_tags:
            # Extract capital letters or first letter of each word part
            short = ''.join(c for c in tag if c.isupper())
            if not short:
                # No capitals, use first letter of each underscore-separated part
                parts = tag.split('_')
                short = ''.join(p[0].upper() for p in parts if p)
            if not short:
                short = tag[:3].upper()  # Fallback to first 3 chars
            term_map[tag] = short
    
    # Apply mapping to create short names
    result = []
    for name in longnames_list:
        parts = name.split('_', 1)
        if len(parts) > 1 and parts[0] in term_map:
            # Replace term tag with short version
            short_name = term_map[parts[0]]
            if len(parts) > 1:
                short_name += '_' + parts[1]
            result.append(short_name)
        else:
            # No term tag, keep as is
            result.append(name)
    
    return result


def longnames(
    term_tag: Optional[str],
    cond_tags: List[str],
    nb: int = 1
) -> List[str]:
    """Generate long (verbose) column names.

    Alias for :func:`make_column_names` provided for API
    consistency with the R ``fmridesign`` package.

    Parameters
    ----------
    term_tag : str or None
        Sanitized, unique term tag (None for identity terms).
    cond_tags : list of str
        Base condition tags (without basis suffix).
    nb : int, default=1
        Number of basis functions.

    Returns
    -------
    list of str
        Long column names.

    See Also
    --------
    make_column_names : Canonical implementation.
    shortnames : Abbreviated column names.
    """
    return make_column_names(term_tag, cond_tags, nb)


def translate_legacy_pattern(pattern: str) -> str:
    """Translate legacy R-style pattern to modern naming.

    Performs three transformations:
    1. Replace Var[Level] -> Var.Level
    2. Replace :basis[digits] at end -> _b<digits>
    3. Replace standalone : (not ::) -> _

    Parameters
    ----------
    pattern : str
        Legacy pattern string

    Returns
    -------
    str
        Translated pattern

    Examples
    --------
    >>> translate_legacy_pattern("condition[A]")
    'condition.A'
    >>> translate_legacy_pattern("term:basis[2]")
    'term_b2'
    >>> translate_legacy_pattern("fac1:fac2")
    'fac1_fac2'
    >>> translate_legacy_pattern("fac1:fac2[level]")
    'fac1_fac2.level'
    """
    # 1. Replace :basis[digits] at end -> _b<digits> (MUST run before bracket substitution)
    pattern = re.sub(r':basis\[(\d+)\](\$?)$', r'_b\1\2', pattern)

    # 2. Replace Var[Level] -> Var.Level
    pattern = re.sub(r'([A-Za-z0-9_\.]+)\[([^\]]+)\]', r'\1.\2', pattern)

    # 3. Replace standalone : -> _ (not :: which is double colon)
    # Use negative lookbehind and lookahead to avoid matching ::
    pattern = re.sub(r'(?<!:):(?!:)', '_', pattern)

    return pattern


__all__ = [
    'zeropad',
    'sanitize',
    'sanitize_level',
    'basis_suffix',
    'feature_suffix',
    'make_unique_tags',
    'make_term_tag',
    'level_token',
    'continuous_token',
    'make_cond_tag',
    'add_basis_suffix',
    'make_column_names',
    'make_unique_colnames',
    'shortnames',
    'longnames',
    'translate_legacy_pattern',
]
