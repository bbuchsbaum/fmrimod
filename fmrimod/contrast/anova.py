"""High-level ANOVA contrast generator for typed factorial terms.

Given a fitted model's :class:`~fmrimod.design.columns.DesignColumns`
and a factorial term name, :func:`anova_contrasts` returns the full
ANOVA decomposition (main effects + every interaction up to the
N-way interaction + the joint omnibus F) as ready-to-use contrast
vectors of length ``n_total_columns`` — addressable by name.

Currently supports **binary factors** (two levels per factor). The
underlying ``fmrimod.contrast.factorial.generate_interaction_contrast``
machinery supports higher arity; extending this helper is a matter
of mapping its output rows onto the realised column indices via
:meth:`~fmrimod.design.columns.DesignColumns.cell`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..design.columns import DesignColumns

Array = NDArray[np.float64]


@dataclass(frozen=True)
class AnovaContrasts:
    """Result of :func:`anova_contrasts`.

    Attributes
    ----------
    term
        The factorial term covered (e.g. ``"trial_type:difficulty"``).
    factors
        Tuple of factor names in factorial order.
    factor_values
        Tuple of value tuples per factor, in the order used for
        coding.
    main
        Mapping ``factor_name -> t-contrast (length n_total)``.
        Each contrast has weight ``+1/n_per_other_factors`` on the
        positive-coded cells and ``-1/n_per_other_factors`` on the
        negative-coded cells, so the contrast estimate is a
        difference of cell means.
    interaction
        Mapping ``(factor_a, factor_b, ...) -> t-contrast`` keyed by
        ordered tuples of factor names. Includes every interaction
        from 2-way through the full N-way interaction.
    omnibus
        F-contrast (shape ``(n_cells, n_total)``) with one-hot rows
        on each cell; tests joint significance of the factorial
        block.
    """

    term: str
    factors: tuple[str, ...]
    factor_values: tuple[tuple[str, ...], ...]
    main: Mapping[str, Array]
    interaction: Mapping[tuple[str, ...], Array]
    omnibus: Array


def anova_contrasts(
    columns: DesignColumns,
    term: str,
    factors: Sequence[str],
) -> AnovaContrasts:
    """Build the full ANOVA contrast set for a typed factorial term.

    Parameters
    ----------
    columns
        Realised design columns from a fitted model.
    term
        Name of the factorial term (e.g.
        ``"trial_type:difficulty:context"``). Must appear on every
        cell column's ``DesignColumn.term``.
    factors
        Factor names in factorial order. Must match the ``term``
        split by ``":"`` and each factor must currently have
        exactly two levels.

    Returns
    -------
    AnovaContrasts
        Bundle carrying ``main[factor]``, ``interaction[(a,b,...)]``,
        and ``omnibus``. Use the contrast vectors as the
        ``contrast_vector`` argument to
        :meth:`fmrimod.glm.fmri_lm.FmriLm.contrast`.

    Notes
    -----
    Currently restricted to binary factors (two levels per factor).
    Each cell's factor-value pair is parsed from the column's
    ``level`` string via
    :meth:`DesignColumns.cell`; non-binary factors will raise
    ``NotImplementedError`` from this helper. Higher-arity support
    is a separate addition.
    """
    factor_list = list(factors)
    if list(term.split(":")) != factor_list:
        raise ValueError(
            f"anova_contrasts: ``term`` must equal ``':'.join(factors)``; "
            f"got term={term!r}, factors={factor_list!r}"
        )
    if not factor_list:
        raise ValueError(
            "anova_contrasts: at least one factor required"
        )

    # Discover the unique values for each factor by walking the columns.
    factor_values: dict[str, list[str]] = {f: [] for f in factor_list}
    seen: dict[str, set[str]] = {f: set() for f in factor_list}
    from ..design.columns import _parse_factorial_level
    for col in columns.where(term=term).columns:
        parsed = _parse_factorial_level(col.level, factor_list)
        if parsed is None:
            continue
        for f in factor_list:
            v = parsed.get(f)
            if v is not None and v not in seen[f]:
                seen[f].add(v)
                factor_values[f].append(v)
    fv_tuple = tuple(tuple(factor_values[f]) for f in factor_list)
    for f, vals in zip(factor_list, fv_tuple):
        if len(vals) != 2:
            raise NotImplementedError(
                f"anova_contrasts: factor {f!r} has {len(vals)} levels "
                f"({vals!r}); only binary factors are supported in this "
                f"helper. Use the lower-level "
                f"``fmrimod.contrast.factorial.generate_interaction_contrast`` "
                f"for higher arity."
            )

    n_total = len(columns)
    # Locate cell indices via typed cell lookup.
    cell_indices: dict[tuple[str, ...], int] = {}
    for combo in itertools.product(*fv_tuple):
        kwargs = dict(zip(factor_list, combo))
        cell_indices[combo] = columns.cell(term=term, **kwargs).index

    # Factor-level coding for binary factors: first value = -1, second = +1.
    code: dict[str, dict[str, float]] = {
        f: {vals[0]: -1.0, vals[1]: +1.0}
        for f, vals in zip(factor_list, fv_tuple)
    }
    n_cells = len(cell_indices)
    # Number of cells over which a given factor combination is "constant" —
    # used as the contrast normaliser so the main effect estimates a
    # difference of cell means and interactions follow the same scale.
    n_per_main = n_cells // 2  # binary factors: 2 levels each
    n_per_pair = max(1, n_cells // 4)

    def _build(weight_fn) -> Array:
        v = np.zeros(n_total, dtype=np.float64)
        for combo, idx in cell_indices.items():
            v[idx] = weight_fn(*combo)
        return v

    main: dict[str, Array] = {}
    for f in factor_list:
        # 1/n_per_main scaling -> contrast value is (mean of cells where
        # factor == +1) minus (mean of cells where factor == -1).
        def weight(*combo, _f=f):
            combo_dict = dict(zip(factor_list, combo))
            return code[_f][combo_dict[_f]] / float(n_per_main)
        main[f] = _build(weight)

    interaction: dict[tuple[str, ...], Array] = {}
    # Every nonempty subset of factors with size >= 2 = an interaction.
    for k in range(2, len(factor_list) + 1):
        for subset in itertools.combinations(factor_list, k):
            scale = max(1, n_cells // (2 ** k))

            def weight(*combo, _sub=subset, _scale=scale):
                combo_dict = dict(zip(factor_list, combo))
                w = 1.0
                for f in _sub:
                    w *= code[f][combo_dict[f]]
                return w / float(_scale)
            interaction[tuple(subset)] = _build(weight)

    omnibus = np.zeros((n_cells, n_total), dtype=np.float64)
    for row, (_, idx) in enumerate(sorted(cell_indices.items())):
        omnibus[row, idx] = 1.0

    return AnovaContrasts(
        term=term,
        factors=tuple(factor_list),
        factor_values=fv_tuple,
        main=main,
        interaction=interaction,
        omnibus=omnibus,
    )
