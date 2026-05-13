"""Domain-specific language (DSL) for formula specification using operators.

This module provides an operator-based syntax for building model terms.
Event variables are created with :func:`event`, and transformations
(HRFs, basis functions) are applied with the ``@`` operator via
:data:`hrf` and :data:`basis` namespaces. Interactions use ``*``.

Example::

    from fmrimod.formula.dsl import event, hrf

    condition = event('condition')
    block = event('block')

    term1 = condition @ hrf.spm_canonical
    term2 = (condition * block) @ hrf.gamma
"""

from __future__ import annotations

from typing import Any, List, Union

from .base import Term, term
from ..types import BasisProtocol, HRFProtocol


class EventVar:
    """Represents an event variable in the DSL.
    
    This class enables the use of Python operators to create terms
    in a natural, mathematical syntax.
    
    Examples
    --------
    >>> from fmrimod.formula.dsl import event, hrf
    >>> 
    >>> # Create event variables
    >>> condition = event('condition')
    >>> block = event('block')
    >>> 
    >>> # Single event with HRF
    >>> term1 = condition @ hrf.spm_canonical
    >>> 
    >>> # Interaction with HRF
    >>> term2 = (condition * block) @ hrf.gamma
    """
    
    def __init__(self, name: str):
        """Initialize event variable.
        
        Parameters
        ----------
        name : str
            Variable name
        """
        self.name = name
        self._term = Term(name)
    
    def __mul__(self, other: EventVar) -> EventExpr:
        """Create interaction using * operator.
        
        Parameters
        ----------
        other : EventVar
            Other event to interact with
        
        Returns
        -------
        EventExpr
            Expression representing interaction
        """
        if not isinstance(other, EventVar):
            return NotImplemented
        return EventExpr([self.name, other.name])
    
    def __matmul__(self, transform: Transform) -> Term:
        """Apply transformation using @ operator.
        
        Parameters
        ----------
        transform : Transform
            Transformation to apply (HRF, basis, etc.)
        
        Returns
        -------
        Term
            Transformed term
        """
        if not isinstance(transform, Transform):
            return NotImplemented
        return transform.apply(self._term)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EventVar('{self.name}')"


class EventExpr:
    """Expression combining multiple events (e.g., interactions)."""
    
    def __init__(self, events: List[str]):
        """Initialize expression.
        
        Parameters
        ----------
        events : list of str
            Event names in expression
        """
        self.events = events
        self._term = Term(events)
    
    def __mul__(self, other: Union[EventVar, EventExpr]) -> EventExpr:
        """Extend interaction using * operator.
        
        Parameters
        ----------
        other : EventVar or EventExpr
            Additional event(s) to include
        
        Returns
        -------
        EventExpr
            Extended expression
        """
        if isinstance(other, EventVar):
            return EventExpr(self.events + [other.name])
        elif isinstance(other, EventExpr):
            return EventExpr(self.events + other.events)
        return NotImplemented
    
    def __matmul__(self, transform: Transform) -> Term:
        """Apply transformation using @ operator.
        
        Parameters
        ----------
        transform : Transform
            Transformation to apply
        
        Returns
        -------
        Term
            Transformed term
        """
        if not isinstance(transform, Transform):
            return NotImplemented
        return transform.apply(self._term)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EventExpr({' * '.join(self.events)})"


class Transform:
    """Base class for transformations applied to events via ``@``.

    Subclasses must implement :meth:`apply`. Transforms can be chained
    with ``@`` to produce a :class:`ChainedTransform`.
    """
    
    def apply(self, model_term: Term) -> Term:
        """Apply transformation to a term.
        
        Parameters
        ----------
        term : Term
            Term to transform
        
        Returns
        -------
        Term
            Transformed term
        """
        raise NotImplementedError
    
    def __matmul__(self, other: Transform) -> ChainedTransform:
        """Chain transformations using @ operator.
        
        Parameters
        ----------
        other : Transform
            Next transformation in chain
        
        Returns
        -------
        ChainedTransform
            Combined transformation
        """
        if not isinstance(other, Transform):
            return NotImplemented
        return ChainedTransform([self, other])


class HRFTransform(Transform):
    """HRF transformation."""
    
    def __init__(self, hrf: Union[str, HRFProtocol]):
        """Initialize HRF transform.
        
        Parameters
        ----------
        hrf : str or HRFProtocol
            HRF specification
        """
        self.hrf = hrf
    
    def apply(self, model_term: Term) -> Term:
        """Apply HRF to term."""
        return model_term.with_hrf(self.hrf)
    
    def __repr__(self) -> str:
        """String representation."""
        hrf_name = self.hrf if isinstance(self.hrf, str) else self.hrf.name
        return f"HRF({hrf_name})"


class BasisTransform(Transform):
    """Basis function transformation."""
    
    def __init__(self, basis: BasisProtocol):
        """Initialize basis transform.
        
        Parameters
        ----------
        basis : BasisProtocol
            Basis function
        """
        self.basis = basis
    
    def apply(self, model_term: Term) -> Term:
        """Apply basis to term."""
        return model_term.with_basis(self.basis)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Basis({self.basis.name})"


class ChainedTransform(Transform):
    """Multiple transformations chained together."""
    
    def __init__(self, transforms: List[Transform]):
        """Initialize chained transform.
        
        Parameters
        ----------
        transforms : list of Transform
            Transformations to apply in order
        """
        self.transforms = transforms
    
    def apply(self, model_term: Term) -> Term:
        """Apply all transformations in sequence."""
        result = model_term
        for transform in self.transforms:
            result = transform.apply(result)
        return result
    
    def __matmul__(self, other: Transform) -> ChainedTransform:
        """Add another transformation to the chain."""
        if not isinstance(other, Transform):
            return NotImplemented
        return ChainedTransform(self.transforms + [other])
    
    def __repr__(self) -> str:
        """String representation."""
        return " @ ".join(str(t) for t in self.transforms)


class HRFNamespace:
    """Namespace for HRF specifications accessed as attributes.

    Allows natural syntax like ``hrf.spm_canonical`` or explicit
    ``hrf('spmg2')``. Attribute names are passed through using the
    same underscore aliases accepted by the HRF registry.

    Examples
    --------
    >>> from fmrimod.formula.dsl import hrf
    >>> transform = hrf.spm_canonical   # HRFTransform("spm_canonical")
    >>> transform = hrf('spmg2')        # HRFTransform("spmg2")
    """
    
    def __getattr__(self, name: str) -> HRFTransform:
        """Get HRF by name.
        
        Parameters
        ----------
        name : str
            HRF name
        
        Returns
        -------
        HRFTransform
            HRF transformation
        """
        return HRFTransform(name)
    
    def __call__(self, hrf: Union[str, HRFProtocol]) -> HRFTransform:
        """Create HRF transform with explicit specification.
        
        Parameters
        ----------
        hrf : str or HRFProtocol
            HRF specification
        
        Returns
        -------
        HRFTransform
            HRF transformation
        """
        return HRFTransform(hrf)


class BasisNamespace:
    """Namespace for basis function specifications.

    Provides factory methods for common basis types (polynomial,
    B-spline) and a direct call interface for custom bases.

    Examples
    --------
    >>> from fmrimod.formula.dsl import basis
    >>> transform = basis.poly(degree=3)
    >>> transform = basis.spline(df=5)
    """
    
    def poly(self, degree: int, **kwargs) -> BasisTransform:
        """Polynomial basis.
        
        Parameters
        ----------
        degree : int
            Polynomial degree
        **kwargs
            Additional parameters
        
        Returns
        -------
        BasisTransform
            Basis transformation
        """
        # Import here to avoid circular dependency
        from ..basis import Poly
        return BasisTransform(Poly(degree=degree, **kwargs))
    
    def spline(self, df: int, **kwargs) -> BasisTransform:
        """B-spline basis.
        
        Parameters
        ----------
        df : int
            Degrees of freedom
        **kwargs
            Additional parameters
        
        Returns
        -------
        BasisTransform
            Basis transformation
        """
        # Import here to avoid circular dependency
        from ..basis import BSpline
        return BasisTransform(BSpline(df=df, **kwargs))
    
    def __call__(self, basis: BasisProtocol) -> BasisTransform:
        """Create basis transform with explicit specification.
        
        Parameters
        ----------
        basis : BasisProtocol
            Basis function
        
        Returns
        -------
        BasisTransform
            Basis transformation
        """
        return BasisTransform(basis)


# Global instances for natural syntax
hrf = HRFNamespace()
basis = BasisNamespace()


def event(name: str) -> EventVar:
    """Create an event variable for DSL syntax.
    
    Parameters
    ----------
    name : str
        Event variable name
    
    Returns
    -------
    EventVar
        Event variable for DSL operations
    
    Examples
    --------
    >>> condition = event('condition')
    >>> block = event('block')
    >>> 
    >>> # Create terms using operators
    >>> term1 = condition @ hrf.spm_canonical
    >>> term2 = (condition * block) @ hrf.gamma
    """
    return EventVar(name)


# Convenience re-export
__all__ = [
    'EventVar',
    'EventExpr',
    'Transform',
    'HRFTransform',
    'BasisTransform',
    'ChainedTransform',
    'hrf',
    'basis',
    'event',
]
