"""Formula parsing and specification for fmrimod.

This module provides multiple interfaces for specifying event models:

1. String formulas (R-style):
   >>> model = EventModel("onset ~ hrf(condition)", data=df)

2. Builder pattern (explicit):
   >>> model = (EventModelBuilder()
   ...     .add_term(Term('condition').with_hrf('spm_canonical'))
   ...     .build())

3. DSL with operators (mathematical):
   >>> model = EventModel(
   ...     condition @ hrf.spm_canonical,
   ...     (condition * block) @ hrf.gamma
   ... )

4. Functional with pipes:
   >>> model = EventModel([
   ...     term('condition') | hrf('spm_canonical'),
   ...     term('condition', 'block') | hrf('gamma')
   ... ])
"""

# Core classes
from .base import (
    Term,
    EventModelBuilder,
    term as base_term,
    interaction,
)

# String formula parsing
from .parser import (
    Formula,
    FormulaTerm,
    FormulaParser,
    FormulaEvaluator,
    parse_formula,
)

# Re-export FormulaContext from types
from ..types import FormulaContext

# DSL with operators
from .dsl import (
    EventVar,
    EventExpr,
    Transform,
    HRFTransform,
    BasisTransform,
    ChainedTransform,
    event,
    hrf as dsl_hrf,
    basis as dsl_basis,
)

# Functional interface
from .functional import (
    PipeTerm,
    term,
    hrf,
    basis,
    poly,
    spline,
    scale,
    name,
    compose,
)

__all__ = [
    # Core
    "Term",
    "EventModelBuilder",
    "base_term",
    "interaction",
    # Parser
    "Formula",
    "FormulaTerm", 
    "FormulaParser",
    "FormulaEvaluator",
    "parse_formula",
    # DSL
    "EventVar",
    "EventExpr",
    "Transform",
    "HRFTransform",
    "BasisTransform",
    "ChainedTransform",
    "event",
    "dsl_hrf",
    "dsl_basis",
    # Functional
    "PipeTerm",
    "term",
    "hrf",
    "basis",
    "poly",
    "spline",
    "scale",
    "name",
    "compose",
    # Types
    "FormulaContext",
]