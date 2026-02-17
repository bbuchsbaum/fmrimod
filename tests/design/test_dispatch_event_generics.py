"""Regression tests for event-level generics in fmrimod.dispatch."""

import numpy as np

from fmrimod.dispatch import cells, columns, conditions, durations, elements, onsets
from fmrimod.basis import Poly
from fmrimod.events.basis import EventBasis
from fmrimod.events.factor import EventFactor
from fmrimod.events.matrix import EventMatrix


def test_dispatch_event_generics_work_for_event_factor():
    """EventFactor should dispatch without protocol-related NotImplementedError."""
    ev = EventFactor(
        name="condition",
        onsets=[1.0, 2.0, 3.0],
        durations=[1.0, 1.0, 1.0],
        values=["A", "B", "A"],
    )

    np.testing.assert_array_equal(onsets(ev), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(durations(ev), np.array([1.0, 1.0, 1.0]))
    assert columns(ev) == ["condition.A", "condition.B"]
    assert conditions(ev) == ["A", "B"]
    assert cells(ev) == 2
    assert elements(ev) == ["A", "B"]


def test_dispatch_conditions_and_cells_for_event_matrix():
    """EventMatrix conditions/cells should reflect column names and count."""
    ev = EventMatrix(
        name="motion",
        onsets=[1.0, 2.0],
        durations=[1.0, 1.0],
        values=np.array([[0.1, 0.2], [0.3, 0.4]]),
        column_names=["x", "y"],
    )

    assert conditions(ev) == ["x", "y"]
    assert cells(ev) == 2
    assert columns(ev) == ["x", "y"]
    np.testing.assert_array_equal(elements(ev), ev.values)


def test_dispatch_columns_for_event_variable():
    """EventVariable columns should default to the event name."""
    from fmrimod.events.variable import EventVariable

    ev = EventVariable(
        name="rt",
        onsets=[1.0, 2.0],
        durations=[1.0, 1.0],
        values=[0.2, 0.4],
        center=False,
    )
    assert columns(ev) == ["rt"]


def test_dispatch_event_basis_reports_multi_column_metadata():
    """EventBasis dispatch should reflect basis-expanded columns."""
    ev = EventBasis(
        name="rating",
        onsets=[1.0, 2.0, 3.0],
        durations=[1.0, 1.0, 1.0],
        values=[1.0, 2.0, 3.0],
        basis=Poly(degree=2),
    )

    assert columns(ev) == ev.basis_names
    assert conditions(ev) == ev.basis_names
    assert cells(ev) == ev.n_basis
    np.testing.assert_array_equal(elements(ev), ev.expanded_values)
