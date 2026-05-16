"""Tutorial-support: seed a known effect located by typed contrast identity.

Internal helper (deliberately not in ``fmrimod.simulate.__all__``): it exists
so flagship tutorials can build a demo dataset with a known injected effect
*resolved through the same typed ``condition()`` object the lesson's contrast
uses* — never the rendered-name string lookup
``design_colmap(built).query("condition == 'term.face'")`` the project disowns.
Promotion to a public/exported signature is a separate steward decision
(bd-01KRSFY2XK499A7P15QBEPXFCS).
"""

from __future__ import annotations

import numpy as np

import fmrimod as fm

from ..contrast.semantic import ConditionRef
from .bold import simulate_bold

__all__: list[str] = []


def dataset_with_effect(
    model_spec: object,
    events: object,
    *,
    on: ConditionRef,
    effect: float,
    sampling_frame: object,
    n_signal: int = 5,
    n_voxels: int = 10,
    noise: float = 1.5,
    tr: float = 1.0,
    seed: int = 0,
) -> object:
    """Return an :class:`FmriDataset` with ``effect`` injected on ``on``.

    ``on`` is a typed :func:`fmrimod.contrast.semantic.condition` reference.
    The injection column is resolved through the same typed
    ``design_columns()`` provenance path the lesson's
    ``fit.contrast(condition(...) - condition(...))`` uses, so a level/term
    that does not resolve raises a typed error rather than silently missing.
    The first ``n_signal`` voxels carry ``effect``; the rest are null.
    """
    if not isinstance(on, ConditionRef):
        raise TypeError(
            "dataset_with_effect(on=...) requires a typed condition() "
            f"reference (ConditionRef), got {type(on).__name__}; pass "
            "condition('face', term='condition'), not a raw string."
        )

    from fmrimod.spec import compile as compile_spec

    built, _baseline = compile_spec(
        model_spec, data=events, sampling_frame=sampling_frame
    )
    design = np.asarray(built.design_matrix, dtype=float)
    n_time, n_event_cols = design.shape

    placeholder = fm.fmri_dataset(
        np.zeros((n_time, max(n_voxels, 1)), dtype=float), tr=tr, events=events
    )
    columns = fm.fmri_lm(model_spec, placeholder).design_columns()
    target = columns.where(term=on.term, level=on.level).one()
    if target.role != "task" or target.index >= n_event_cols:
        raise ValueError(
            f"condition {on.display_name!r} resolved to column "
            f"{target.name!r} (index={target.index}, role={target.role}), "
            f"which is not an injectable event-design task column "
            f"(event design has {n_event_cols} columns)."
        )

    rng = np.random.default_rng(seed)
    betas = np.zeros((n_event_cols, n_voxels), dtype=float)
    betas[target.index, :n_signal] = float(effect)
    bold = simulate_bold(design, betas, noise_sd=noise, n_voxels=n_voxels, rng=rng)
    return fm.fmri_dataset(bold, tr=tr, events=events)
