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
# Re-export FormulaContext from types
from ..types import FormulaContext
from .base import (
    EventModelBuilder,
    Term,
    interaction,
)
from .base import (
    term as base_term,
)

# DSL with operators
from .dsl import (
    BasisTransform,
    ChainedTransform,
    EventExpr,
    EventVar,
    HRFTransform,
    Transform,
    event,
)
from .dsl import (
    basis as dsl_basis,
)
from .dsl import (
    hrf as dsl_hrf,
)

# Functional interface
from .functional import (
    PipeTerm,
    basis,
    compose,
    hrf,
    name,
    poly,
    scale,
    spline,
    term,
)

# String formula parsing
from .parser import (
    Formula,
    FormulaEvaluator,
    FormulaParser,
    FormulaTerm,
    parse_formula,
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