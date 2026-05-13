"""Typed Spec / Term object tree for fmrimod designs.

Use this module's builders to construct designs with full IDE introspection
and type checking::

    from fmrimod import hrf, drift, intercept

    spec = hrf("trial_type", basis="spm") + drift("cosine", cutoff=128) + intercept()
    fit = fm.fmri_lm(spec, dataset)

The string formula path (``fm.fmri_lm("hrf(trial_type)", dataset)``) continues
to work and is now treated as syntactic sugar that compiles down to the same
Spec tree.
"""

from ._compile import compile, compile_baseline, compile_events, legacy_formula_to_spec
from .builders import confounds, drift, hrf, intercept, trialwise
from .terms import (
    Confounds,
    Drift,
    HrfTerm,
    Intercept,
    Predicate,
    Spec,
    Term,
    as_spec,
    is_spec,
)

__all__ = [
    # Types
    "Term",
    "HrfTerm",
    "Drift",
    "Intercept",
    "Confounds",
    "Spec",
    "Predicate",
    # Builders
    "hrf",
    "drift",
    "intercept",
    "confounds",
    "trialwise",
    # Helpers
    "as_spec",
    "is_spec",
    # Compilation
    "compile",
    "compile_events",
    "compile_baseline",
    "legacy_formula_to_spec",
]
