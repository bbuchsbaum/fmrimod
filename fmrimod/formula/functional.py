"""Functional interface for formula specification using pipe operators.

This module provides a pipe-based (``|``) syntax for building model
terms, inspired by R's ``magrittr`` pipe. Terms are created with
:func:`term` and transformations are applied by piping through
:func:`hrf`, :func:`poly`, :func:`spline`, etc.

Example::

    from fmrimod.formula.functional import term, hrf, poly

    t1 = term('condition') | hrf('spmg1')
    t2 = term('rating') | poly(3) | hrf('spmg1')
    t3 = term('condition', 'block') | hrf('spmg2')
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union, cast

from ..types import BasisProtocol, HRFProtocol
from .base import Term


class PipeTerm(Term):
    """Term that supports pipe operations.
    
    This extends the base Term class with the | operator for
    functional composition.
    
    Examples
    --------
    >>> from fmrimod.formula.functional import term, hrf, poly
    >>> 
    >>> # Simple term with HRF
    >>> t1 = term('condition') | hrf('spm_canonical')
    >>> 
    >>> # Term with basis and HRF
    >>> t2 = term('parametric') | poly(3) | hrf('gamma')
    >>> 
    >>> # Interaction term
    >>> t3 = term('condition', 'block') | hrf('spm_canonical')
    """
    
    def __or__(self, transform: Callable[[Term], Term]) -> PipeTerm:
        """Apply transformation using | operator.
        
        Parameters
        ----------
        transform : callable
            Function that transforms a term
        
        Returns
        -------
        PipeTerm
            Transformed term
        """
        # Apply the transformation
        result = transform(self)
        
        # Ensure we return a PipeTerm for continued chaining
        if isinstance(result, PipeTerm):
            return result
        elif isinstance(result, Term):
            # Convert regular Term to PipeTerm
            pipe_term = PipeTerm(
                events=result.events,
                hrf=result.hrf,
                basis=result.basis,
                name=result.name,
            )
            pipe_term._kwargs = result._kwargs
            return pipe_term
        else:
            raise TypeError(
                f"Transform must return a Term, got {type(result)}"
            )


def term(*events: str, **kwargs: object) -> PipeTerm:
    """Create a pipeable term.
    
    Parameters
    ----------
    *events : str
        Event variable name(s)
    **kwargs
        Additional parameters
    
    Returns
    -------
    PipeTerm
        New pipeable term
    
    Examples
    --------
    >>> # Single event
    >>> t = term('condition')
    >>> 
    >>> # Interaction
    >>> t = term('condition', 'block')
    >>> 
    >>> # With parameters
    >>> t = term('condition', amplitude='high')
    """
    forwarded = cast("dict[str, Any]", kwargs)
    if len(events) == 1:
        return PipeTerm(events[0], **forwarded)
    else:
        return PipeTerm(list(events), **forwarded)


def hrf(
    spec: Union[str, 'HRFProtocol'] = "spmg1",
    *,
    subset: Optional[object] = None,
    contrasts: Optional[object] = None,
    normalize: bool = False,
    summate: bool = True,
    hrf_fun: Optional[Callable[..., Any]] = None,
    id: Optional[str] = None,
    prefix: Optional[str] = None,
    lag: float = 0.0,
    nbasis: int = 1,
    onsets: Optional[object] = None,
    durations: Optional[object] = None,
) -> Callable[['Term'], 'Term']:
    """Create HRF transformation function.

    Parameters
    ----------
    spec : str or HRFProtocol, default "spmg1"
        HRF specification (basis type name or HRF object)
    subset : optional
        Subset indicator expression
    contrasts : optional
        Contrast specifications for this term (dict or list of ContrastSpec)
    normalize : bool, default False
        Peak-normalize each convolved regressor column
    summate : bool, default True
        Whether impulse amplitudes sum during block/epoch convolution
    hrf_fun : callable, optional
        Per-onset HRF generator function (e.g., from boxcar_hrf_gen())
    id : str, optional
        Explicit term identifier
    prefix : str, optional
        Term prefix for column naming
    lag : float, default 0.0
        Temporal offset in seconds
    nbasis : int, default 1
        Number of basis functions (for multi-basis HRFs)
    onsets : optional
        Onset specification override
    durations : optional
        Duration specification override

    Returns
    -------
    callable
        Function that applies HRF configuration to a term

    Examples
    --------
    >>> t = term('condition') | hrf('spm_canonical')
    >>> t = term('condition') | hrf('spmg1', normalize=True, lag=2.0)
    """
    def transform(t: 'Term') -> 'Term':
        # Apply HRF spec
        result = t.with_hrf(spec)

        # Store additional parameters in Term._kwargs
        if not hasattr(result, '_kwargs') or result._kwargs is None:
            result._kwargs = {}

        if normalize:
            result._kwargs['normalize'] = True
        if not summate:
            result._kwargs['summate'] = False
        if hrf_fun is not None:
            result._kwargs['hrf_fun'] = hrf_fun
        if subset is not None:
            result._kwargs['subset'] = subset
        if contrasts is not None:
            result._kwargs['contrasts'] = contrasts
        if id is not None:
            result = result.with_name(id)
        if prefix is not None:
            result._kwargs['prefix'] = prefix
        if lag != 0.0:
            result._kwargs['lag'] = lag
        if nbasis != 1:
            result._kwargs['nbasis'] = nbasis
        if onsets is not None:
            result._kwargs['onsets'] = onsets
        if durations is not None:
            result._kwargs['durations'] = durations

        return result

    transform.__name__ = f"hrf({spec})"
    return transform


def basis(spec: BasisProtocol) -> Callable[[Term], Term]:
    """Create basis transformation function.
    
    Parameters
    ----------
    spec : BasisProtocol
        Basis function
    
    Returns
    -------
    callable
        Function that applies basis to a term
    
    Examples
    --------
    >>> from fmrimod.basis import Poly
    >>> t = term('parametric') | basis(Poly(degree=3))
    """
    def transform(t: Term) -> Term:
        return t.with_basis(spec)
    
    transform.__name__ = f"basis({spec.name})"
    return transform


def poly(degree: int, **kwargs: object) -> Callable[[Term], Term]:
    """Create polynomial basis transformation.
    
    Parameters
    ----------
    degree : int
        Polynomial degree
    **kwargs
        Additional parameters for Poly
    
    Returns
    -------
    callable
        Function that applies polynomial basis to a term
    
    Examples
    --------
    >>> t = term('parametric') | poly(3) | hrf('spm_canonical')
    """
    # Import here to avoid circular dependency
    from ..basis import Poly
    
    poly_basis = Poly(degree=degree, **cast("dict[str, Any]", kwargs))

    def transform(t: Term) -> Term:
        return t.with_basis(cast(BasisProtocol, poly_basis))
    
    transform.__name__ = f"poly({degree})"
    return transform


def spline(df: int, **kwargs: object) -> Callable[[Term], Term]:
    """Create B-spline basis transformation.
    
    Parameters
    ----------
    df : int
        Degrees of freedom
    **kwargs
        Additional parameters for BSpline
    
    Returns
    -------
    callable
        Function that applies spline basis to a term
    
    Examples
    --------
    >>> t = term('parametric') | spline(df=4) | hrf('gamma')
    """
    # Import here to avoid circular dependency
    from ..basis import BSpline
    
    spline_basis = BSpline(df=df, **cast("dict[str, Any]", kwargs))

    def transform(t: Term) -> Term:
        return t.with_basis(cast(BasisProtocol, spline_basis))
    
    transform.__name__ = f"spline(df={df})"
    return transform


def scale(**kwargs: object) -> Callable[[Term], Term]:
    """Create scaling transformation.
    
    Parameters
    ----------
    **kwargs
        Parameters for Scale transform
    
    Returns
    -------
    callable
        Function that applies scaling to a term
    
    Examples
    --------
    >>> t = term('parametric') | scale(center=True) | hrf('spm_canonical')
    """
    # Import here to avoid circular dependency
    from ..basis import Scale
    
    scale_transform = Scale(**cast("dict[str, Any]", kwargs))

    def transform(t: Term) -> Term:
        return t.with_basis(cast(BasisProtocol, scale_transform))
    
    transform.__name__ = "scale()"
    return transform


def name(term_name: str) -> Callable[[Term], Term]:
    """Set custom name for a term.
    
    Parameters
    ----------
    term_name : str
        Name to assign
    
    Returns
    -------
    callable
        Function that sets term name
    
    Examples
    --------
    >>> t = term('condition') | name('main_effect') | hrf('spm_canonical')
    """
    def transform(t: Term) -> Term:
        return t.with_name(term_name)
    
    transform.__name__ = f"name('{term_name}')"
    return transform


def compose(*transforms: Callable[[Term], Term]) -> Callable[[Term], Term]:
    """Compose multiple transformations.
    
    Parameters
    ----------
    *transforms : callable
        Transformation functions to compose
    
    Returns
    -------
    callable
        Composed transformation function
    
    Examples
    --------
    >>> # Define a reusable transformation
    >>> my_transform = compose(poly(3), hrf('spm_canonical'))
    >>> 
    >>> # Apply to multiple terms
    >>> t1 = term('param1') | my_transform
    >>> t2 = term('param2') | my_transform
    """
    def composed(t: Term) -> Term:
        result = t
        for transform in transforms:
            result = transform(result)
        return result
    
    # Create readable name
    names = [getattr(t, '__name__', str(t)) for t in transforms]
    composed.__name__ = f"compose({', '.join(names)})"
    
    return composed


# Re-export for convenience
__all__ = [
    'PipeTerm',
    'term',
    'hrf',
    'basis',
    'poly',
    'spline',
    'scale',
    'name',
    'compose',
]
