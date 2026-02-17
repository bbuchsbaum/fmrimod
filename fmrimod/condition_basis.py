"""Per-condition basis splitting for design matrices."""
import numpy as np


def condition_basis_list(event_term, hrf, sampling_frame, output="condition_list"):
    """Split convolved design matrix by condition.

    Convolves an event term with HRF and splits the result into
    per-condition sub-matrices.

    Parameters
    ----------
    event_term : EventTerm
        Event term to convolve
    hrf : HRF
        HRF object to apply
    sampling_frame : SamplingFrame
        Sampling frame defining temporal grid
    output : str
        'matrix' for full design matrix, 'condition_list' for split dict

    Returns
    -------
    dict or ndarray
        If output='condition_list': dict mapping condition names to
        (n_timepoints, nbasis) matrices.
        If output='matrix': full convolved design matrix.
    """
    from .design.event_model import EventModel
    from .formula.base import Term

    if output not in {"condition_list", "matrix"}:
        raise ValueError("output must be 'condition_list' or 'matrix'")

    nb = hrf.nbasis

    # Get conditions from the event term
    conds = event_term.conditions(drop_empty=False)

    # Create a temporary EventModel for convolution
    events = {e.name: e for e in event_term.events}
    term = Term(events=[e.name for e in event_term.events])
    term.hrf = hrf

    model = EventModel(
        terms=[term],
        events=events,
        sampling_info=sampling_frame,
    )

    # Get the design matrix
    dm = model._convolve_term(
        event_term, term
    )

    if output == "matrix":
        return dm

    # Split by condition
    result = {}
    col = 0
    for cond in conds:
        end = col + nb
        if end <= dm.shape[1]:
            result[cond] = dm[:, col:end]
        col = end

    return result
