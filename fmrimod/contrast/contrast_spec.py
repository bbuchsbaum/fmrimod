"""Contrast specification classes and constructors.

This module provides tools for defining statistical contrasts for hypothesis
testing in fMRI analyses. Contrasts can be defined using formulas, patterns,
or directly on design matrix columns.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Union, List, Dict, Any, Literal
from itertools import combinations
import warnings

# Simple placeholder for Formula type
class Formula:
    """Placeholder for formula objects."""
    def __init__(self, expr: str):
        self.expr = expr
    def __str__(self):
        return self.expr
    def __repr__(self):
        return f"Formula({self.expr!r})"


# Predicate inputs accepted by the contrast constructors. ``Formula`` is the
# original strict type; ``dict`` / ``str`` are the ergonomic v1 conveniences;
# ``callable`` is the escape hatch (stored on the Formula object's ``_fn``
# attribute, resolved later when contrast weights are computed).
Predicate = Union[Formula, str, Mapping[str, Any], Callable[..., Any]]


def _dict_to_predicate_string(d: Mapping[str, Any]) -> str:
    """Render a ``{column: value}`` dict as a Python-evaluable predicate.

    Multi-key dicts are joined with ``&`` (AND).  String values are quoted.
    """
    parts: List[str] = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f"{k} == {v!r}")
        elif isinstance(v, (list, tuple, set)):
            # Membership predicate: column ∈ values
            quoted = ", ".join(repr(x) if isinstance(x, str) else str(x) for x in v)
            parts.append(f"{k} in ({quoted})")
        else:
            parts.append(f"{k} == {v}")
    return " & ".join(parts) if parts else "True"


def _to_formula(predicate: Predicate, *, arg_name: str = "predicate") -> Formula:
    """Coerce a predicate input to a :class:`Formula`.

    Accepts:

    - :class:`Formula` — returned unchanged.
    - ``str`` — wrapped as ``Formula(expr)``.
    - ``Mapping[str, Any]`` — rendered to a string predicate via
      :func:`_dict_to_predicate_string` and wrapped.
    - ``callable`` — wrapped in a Formula with the callable attached as
      ``_fn``; the contrast-weights resolver should evaluate it against the
      term's condition cells when weights are computed.
    """
    if isinstance(predicate, Formula):
        return predicate
    if isinstance(predicate, str):
        return Formula(predicate)
    if isinstance(predicate, Mapping):
        return Formula(_dict_to_predicate_string(predicate))
    if callable(predicate):
        wrapped = Formula(f"<callable {arg_name}>")
        wrapped._fn = predicate  # type: ignore[attr-defined]
        return wrapped
    raise TypeError(
        f"{arg_name} must be a Formula, dict, string, or callable; got "
        f"{type(predicate).__name__}"
    )


class ContrastSpec:
    """Base class for contrast specifications."""
    
    def __init__(
        self,
        name: str,
        A: Optional[Formula] = None,
        B: Optional[Formula] = None,
        where: Optional[Formula] = None,
        **kwargs
    ):
        """Initialize contrast specification.
        
        Parameters
        ----------
        name : str
            Name of the contrast
        A : Formula, optional
            Primary formula or condition
        B : Formula, optional  
            Secondary formula (for comparisons)
        where : Formula, optional
            Subsetting condition
        **kwargs
            Additional parameters stored as attributes
        """
        self.name = name
        self.A = A
        self.B = B
        self.where = where
        
        # Store any additional parameters
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self) -> str:
        """Rich string representation."""
        cls_name = self.__class__.__name__
        parts = [f"{cls_name}(name='{self.name}'"]
        if self.A is not None:
            parts.append(f", A={self.A}")
        if self.B is not None:
            parts.append(f", B={self.B}")
        if self.where is not None:
            parts.append(f", where={self.where}")
        parts.append(")")
        return "".join(parts)

    def __sub__(self, other: 'ContrastSpec') -> 'PairContrastSpec':
        """Compute contrast difference via - operator.

        Creates a new PairContrastSpec representing self minus other.

        Parameters
        ----------
        other : ContrastSpec
            Contrast to subtract

        Returns
        -------
        PairContrastSpec
            New contrast representing the difference

        Examples
        --------
        >>> c1 = unit_contrast(Formula("A"), name="A")
        >>> c2 = unit_contrast(Formula("B"), name="B")
        >>> c_diff = c1 - c2  # Creates PairContrastSpec
        """
        if not isinstance(other, ContrastSpec):
            return NotImplemented

        return PairContrastSpec(
            name=f"{self.name}-{other.name}",
            A=self.A if self.A is not None else Formula(self.name),
            B=other.A if other.A is not None else Formula(other.name),
            where=self.where,
        )


class ContrastFormulaSpec(ContrastSpec):
    """Formula-based contrast specification.

    Represents a contrast defined by a formula expression (e.g., ``~ A - B``).
    Created by the :func:`contrast` constructor.

    Parameters
    ----------
    name : str
        Name of the contrast.
    A : Formula
        Formula expression defining the contrast.
    where : Formula, optional
        Subsetting condition.

    See Also
    --------
    contrast : Constructor function for formula-based contrasts.
    """
    pass


class UnitContrastSpec(ContrastSpec):
    """Unit contrast specification where weights sum to 1.

    Represents a contrast that tests a condition against the implicit
    baseline. The weights for selected conditions are equal and sum to 1.
    Created by the :func:`unit_contrast` constructor.

    Parameters
    ----------
    name : str
        Name of the contrast.
    A : Formula
        Formula selecting the conditions to test.
    where : Formula, optional
        Subsetting condition.

    See Also
    --------
    unit_contrast : Constructor function for unit contrasts.
    pair_contrast : For testing one condition against another.
    """
    pass


class PairContrastSpec(ContrastSpec):
    """Pairwise contrast specification with sum-to-zero weights.

    Represents a contrast between two groups of conditions, where group A
    receives positive weights summing to +1 and group B receives negative
    weights summing to -1, so overall weights sum to zero.
    Created by the :func:`pair_contrast` constructor.

    Parameters
    ----------
    name : str
        Name of the contrast.
    A : Formula
        Formula selecting positive-weight conditions.
    B : Formula
        Formula selecting negative-weight conditions.
    where : Formula, optional
        Subsetting condition.

    See Also
    --------
    pair_contrast : Constructor function for pairwise contrasts.
    pairwise_contrasts : Generate all pairwise comparisons for a factor.
    """
    pass


class ColumnContrastSpec(ContrastSpec):
    """Column-based contrast using regex patterns."""
    
    def __init__(
        self,
        pattern_A: str,
        pattern_B: Optional[str],
        name: str,
        where: Optional[Formula] = None
    ):
        """Initialize column contrast.
        
        Parameters
        ----------
        pattern_A : str
            Regex pattern for positive weights
        pattern_B : str, optional
            Regex pattern for negative weights
        name : str
            Contrast name
        where : Formula, optional
            Subsetting condition (currently unused)
        """
        super().__init__(name=name, where=where)
        self.pattern_A = pattern_A
        self.pattern_B = pattern_B


class PolyContrastSpec(ContrastSpec):
    """Polynomial contrast specification."""
    
    def __init__(
        self,
        A: Formula,
        name: str,
        where: Optional[Formula] = None,
        degree: int = 1,
        value_map: Optional[Dict[str, float]] = None
    ):
        """Initialize polynomial contrast.
        
        Parameters
        ----------
        A : Formula
            Factor to test
        name : str
            Contrast name
        where : Formula, optional
            Subsetting condition
        degree : int, default=1
            Polynomial degree
        value_map : dict, optional
            Mapping from factor levels to numeric values
        """
        super().__init__(name=name, A=A, where=where)
        self.degree = degree
        self.value_map = value_map


class OnewayContrastSpec(ContrastSpec):
    """One-way (main effect) contrast specification.

    Represents a contrast testing the overall effect of a factor using
    Helmert coding. For k levels, generates k-1 orthogonal contrasts.
    Created by the :func:`oneway_contrast` constructor.

    Parameters
    ----------
    name : str
        Name of the contrast.
    A : Formula
        Factor to test.
    where : Formula, optional
        Subsetting condition.

    See Also
    --------
    oneway_contrast : Constructor function for one-way contrasts.
    """
    pass


class InteractionContrastSpec(ContrastSpec):
    """Interaction contrast specification for factorial designs.

    Represents a contrast testing the interaction between two or more
    factors. Uses Kronecker products of component contrasts.
    Created by the :func:`interaction_contrast` constructor.

    Parameters
    ----------
    name : str
        Name of the contrast.
    A : Formula
        Interaction formula (e.g., ``~ A * B``).
    where : Formula, optional
        Subsetting condition.

    See Also
    --------
    interaction_contrast : Constructor function for interaction contrasts.
    """
    pass


class ContrastSet(list):
    """Collection of contrast specifications."""
    
    def __init__(self, *contrasts):
        """Initialize contrast set.
        
        Parameters
        ----------
        *contrasts : ContrastSpec
            Variable number of contrast specifications
        """
        # Validate all are ContrastSpec
        for c in contrasts:
            if not isinstance(c, ContrastSpec):
                raise TypeError(f"All elements must be ContrastSpec, got {type(c)}")
        
        super().__init__(contrasts)
    
    def __repr__(self) -> str:
        """Rich string representation."""
        n = len(self)
        if n == 0:
            return "ContrastSet(empty)"

        names = [c.name for c in self]
        types = {}
        for c in self:
            ctype = type(c).__name__
            types[ctype] = types.get(ctype, 0) + 1

        type_str = ", ".join(f"{k}: {v}" for k, v in types.items())
        names_preview = ", ".join(names[:5])
        if n > 5:
            names_preview += f", ... ({n - 5} more)"

        return f"ContrastSet(n={n}, names=[{names_preview}], types={{{type_str}}})"


def contrast(
    form: Predicate,
    name: str,
    where: Optional[Predicate] = None
) -> ContrastFormulaSpec:
    """Define a linear contrast using a formula expression.

    Accepts a :class:`Formula`, a string predicate, a dict, or a callable for
    ``form`` and ``where``.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    form_f = _to_formula(form, arg_name="form")
    where_f = _to_formula(where, arg_name="where") if where is not None else None
    return ContrastFormulaSpec(name=name, A=form_f, where=where_f)


def unit_contrast(
    A: Predicate,
    name: str,
    where: Optional[Predicate] = None
) -> UnitContrastSpec:
    """Construct a contrast that sums to 1 (for testing against baseline).
    
    Parameters
    ----------
    A : Formula
        Formula for the contrast expression
    name : str
        Name of the contrast
    where : Formula, optional
        Subset specification
        
    Returns
    -------
    UnitContrastSpec
        Unit contrast specification
        
    Examples
    --------
    >>> # Test main effect of Face against baseline  
    >>> con = unit_contrast(~ Face, name="Main_face")
    >>> 
    >>> # Test within specific blocks
    >>> con2 = unit_contrast(~ Face, name="Face_early", where=~ block <= 3)
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    A_f = _to_formula(A, arg_name="A")
    where_f = _to_formula(where, arg_name="where") if where is not None else None
    return UnitContrastSpec(name=name, A=A_f, where=where_f)


def pair_contrast(
    A: Predicate,
    B: Predicate,
    name: str,
    where: Optional[Predicate] = None,
) -> PairContrastSpec:
    """Construct a sum-to-zero contrast between two logical expressions.
    
    This function is useful for comparing specific conditions or combinations
    of conditions.
    
    Parameters
    ----------
    A : Formula
        First logical expression
    B : Formula
        Second logical expression
    name : str
        Contrast name
    where : Formula, optional
        Subsetting condition
        
    Returns
    -------
    PairContrastSpec
        Pairwise contrast specification
        
    Examples
    --------
    >>> # Compare faces vs scenes
    >>> c = pair_contrast(~ category == "face", ~ category == "scene", 
    ...                   name="face_vs_scene")
    >>> 
    >>> # Complex expressions
    >>> c = pair_contrast(~ stimulus == "face" & emotion == "happy",
    ...                   ~ stimulus == "face" & emotion == "sad",
    ...                   name="happy_vs_sad_faces")
    >>>
    >>> # Dict / string predicate forms (v1 convenience):
    >>> c = pair_contrast({"condition": "face"}, {"condition": "house"},
    ...                   name="face_vs_house")
    >>> c = pair_contrast("condition == 'face' & block <= 3",
    ...                   "condition == 'house' & block <= 3",
    ...                   name="early_face_vs_house")
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    A_f = _to_formula(A, arg_name="A")
    B_f = _to_formula(B, arg_name="B")
    where_f = _to_formula(where, arg_name="where") if where is not None else None
    return PairContrastSpec(name=name, A=A_f, B=B_f, where=where_f)


def column_contrast(
    pattern_A: str,
    name: str,
    pattern_B: Optional[str] = None,
    where: Optional[Formula] = None
) -> ColumnContrastSpec:
    """Define a contrast by targeting design matrix columns using regex patterns.
    
    This is useful for contrasts involving continuous variables or specific
    basis functions.
    
    Parameters
    ----------
    pattern_A : str
        Regex pattern for positive (+) weights
    pattern_B : str, optional
        Regex pattern for negative (-) weights
    name : str
        Contrast name
    where : Formula, optional
        Currently unused, kept for API consistency
        
    Returns
    -------
    ColumnContrastSpec
        Column contrast specification
        
    Examples
    --------
    >>> # Test main effect of continuous RT
    >>> cc1 = column_contrast(pattern_A="^z_RT$", name="Main_RT")
    >>> 
    >>> # Compare conditions for RT effect  
    >>> cc2 = column_contrast(pattern_A="^Condition\\.A_z_RT$",
    ...                       name="CondA_vs_CondB_for_RT",
    ...                       pattern_B="^Condition\\.B_z_RT$")
    >>> 
    >>> # Test specific basis function
    >>> cc3 = column_contrast(pattern_A="_b03$", name="Basis_3_Effect")
    """
    if not isinstance(pattern_A, str):
        raise TypeError("pattern_A must be a string")
    if pattern_B is not None and not isinstance(pattern_B, str):
        raise TypeError("pattern_B must be a string")
    if not isinstance(name, str):
        raise TypeError("name must be a string")
        
    if where is not None:
        warnings.warn("'where' argument is currently ignored for column_contrast")
        
    return ColumnContrastSpec(
        pattern_A=pattern_A,
        pattern_B=pattern_B,
        name=name,
        where=where
    )


def poly_contrast(
    A: Predicate,
    name: str,
    where: Optional[Predicate] = None,
    degree: int = 1,
    value_map: Optional[Dict[str, float]] = None,
) -> PolyContrastSpec:
    """Create polynomial contrasts for testing trends across ordered factor levels.
    
    Useful for analyzing factors with natural ordering (e.g., time, dose).
    
    Parameters
    ----------
    A : Formula
        Ordered factor to test
    name : str
        Contrast name
    where : Formula, optional
        Subsetting condition
    degree : int, default=1
        Polynomial degree
    value_map : dict, optional
        Mapping from factor levels to numeric values
        
    Returns
    -------
    PolyContrastSpec
        Polynomial contrast specification
        
    Examples
    --------
    >>> # Linear trend across time points
    >>> pcon = poly_contrast(~ time, name="linear_time", degree=1)
    >>> 
    >>> # Cubic trend with custom spacing
    >>> pcon = poly_contrast(~ dose, name="dose_cubic", degree=3,
    ...                      value_map={"low": 0, "med": 2, "high": 5})
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if value_map is not None and not isinstance(value_map, dict):
        raise TypeError("value_map must be a dict")
    A_f = _to_formula(A, arg_name="A")
    where_f = _to_formula(where, arg_name="where") if where is not None else None
    return PolyContrastSpec(
        A=A_f,
        name=name,
        where=where_f,
        degree=degree,
        value_map=value_map,
    )


def oneway_contrast(
    A: Predicate,
    name: str,
    where: Optional[Predicate] = None,
) -> OnewayContrastSpec:
    """Create a one-way contrast specification.
    
    Parameters
    ----------
    A : Formula
        Factor to test
    name : str
        Contrast name
    where : Formula, optional
        Subsetting condition
        
    Returns
    -------
    OnewayContrastSpec
        One-way contrast specification
        
    Examples
    --------
    >>> # Main effect contrast
    >>> con = oneway_contrast(~ condition, name="Main_condition")
    >>> # Dict / string predicate (v1 convenience)
    >>> con = oneway_contrast({"condition": ["face", "house"]}, name="cat_main")
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    A_f = _to_formula(A, arg_name="A")
    where_f = _to_formula(where, arg_name="where") if where is not None else None
    return OnewayContrastSpec(name=name, A=A_f, where=where_f)


def interaction_contrast(
    A: Predicate,
    name: str,
    where: Optional[Predicate] = None,
) -> InteractionContrastSpec:
    """Create an interaction contrast specification.
    
    Parameters
    ----------
    A : Formula
        Interaction formula (e.g., ~ A * B)
    name : str
        Contrast name
    where : Formula, optional
        Subsetting condition
        
    Returns
    -------
    InteractionContrastSpec
        Interaction contrast specification
        
    Examples
    --------
    >>> # Two-way interaction
    >>> con = interaction_contrast(~ condition * time, name="condition_by_time")
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    A_f = _to_formula(A, arg_name="A")
    where_f = _to_formula(where, arg_name="where") if where is not None else None
    return InteractionContrastSpec(name=name, A=A_f, where=where_f)


def contrast_set(*contrasts) -> ContrastSet:
    """Create a set of contrasts.
    
    Parameters
    ----------
    *contrasts : ContrastSpec
        Variable number of contrast specifications
        
    Returns
    -------
    ContrastSet
        Collection of contrasts
        
    Examples
    --------
    >>> c1 = contrast(~ A - B, name="A_vs_B")
    >>> c2 = contrast(~ B - C, name="B_vs_C")
    >>> cset = contrast_set(c1, c2)
    """
    return ContrastSet(*contrasts)


def pairwise_contrasts(
    levels: List[str],
    facname: str,
    where: Optional[Formula] = None,
    name_prefix: str = "con"
) -> ContrastSet:
    """Construct pairwise contrasts for all combinations of factor levels.
    
    Parameters
    ----------
    levels : list of str
        Factor levels to compare
    facname : str
        Name of the factor variable
    where : Formula, optional
        Subsetting condition
    name_prefix : str, default="con"
        Prefix for contrast names
        
    Returns
    -------
    ContrastSet
        Set of pairwise contrasts
        
    Examples
    --------
    >>> # All pairwise comparisons
    >>> pcon = pairwise_contrasts(["A", "B", "C"], facname="condition")
    """
    if len(levels) < 2:
        raise ValueError("pairwise_contrasts requires at least two levels")
        
    contrasts = []
    for lev1, lev2 in combinations(levels, 2):
        # Create formulas using string formatting
        # Note: In real implementation, we'd use proper Formula construction
        formula_A = Formula(f"{facname} == '{lev1}'")
        formula_B = Formula(f"{facname} == '{lev2}'") 
        
        con = pair_contrast(
            formula_A, 
            formula_B,
            where=where,
            name=f"{name_prefix}_{lev1}_{lev2}"
        )
        contrasts.append(con)
        
    return ContrastSet(*contrasts)


def one_against_all_contrast(
    levels: List[str],
    facname: str,
    where: Optional[Formula] = None
) -> ContrastSet:
    """Construct contrasts comparing each level against all others.

    Parameters
    ----------
    levels : list of str
        Factor levels
    facname : str
        Factor name
    where : Formula, optional
        Subsetting condition

    Returns
    -------
    ContrastSet
        Set of one-against-all contrasts

    Examples
    --------
    >>> fac_levels = ["A", "B", "C"]
    >>> con = one_against_all_contrast(fac_levels, "condition")
    """
    contrasts = []

    for i, lev in enumerate(levels):
        # Create formulas
        formula_A = Formula(f"{facname} == '{lev}'")
        formula_B = Formula(f"{facname} != '{lev}'")

        con = pair_contrast(
            formula_A,
            formula_B,
            where=where,
            name=f"con_{lev}_vs_other"
        )
        contrasts.append(con)

    return ContrastSet(*contrasts)


def sliding_window_contrasts(levels, facname, window_size=2, where=None, name_prefix="win"):
    """Create contrasts comparing adjacent sliding windows of conditions.

    For window_size k, creates contrasts where window A = levels[i:i+k]
    is compared against window B = levels[i+k:i+2k].

    Parameters
    ----------
    levels : list of str
        Ordered condition levels
    facname : str
        Name of the factor
    window_size : int
        Number of levels in each window (default 2)
    where : optional
        Subset specification
    name_prefix : str
        Prefix for contrast names (default "win")

    Returns
    -------
    ContrastSet
        Set of sliding window contrasts
    """
    if not isinstance(facname, str):
        raise ValueError("facname must be a string")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    L = len(levels)
    if L < 2:
        raise ValueError("sliding_window_contrasts requires at least two levels")
    if 2 * window_size > L:
        raise ValueError("window_size too large: requires 2*window_size <= len(levels)")

    n_con = L - 2 * window_size + 1
    contrasts = []

    for i in range(n_con):
        A_levels = levels[i:i + window_size]
        B_levels = levels[i + window_size:i + 2 * window_size]
        con_name = f"{name_prefix}_{'-'.join(A_levels)}_vs_{'-'.join(B_levels)}"

        # Create formula expressions for each window
        # Window A: facname in ['lev1', 'lev2', ...]
        # Window B: facname in ['lev3', 'lev4', ...]
        if len(A_levels) == 1:
            formula_A = Formula(f"{facname} == '{A_levels[0]}'")
        else:
            levels_str = "', '".join(A_levels)
            formula_A = Formula(f"{facname} in ['{levels_str}']")

        if len(B_levels) == 1:
            formula_B = Formula(f"{facname} == '{B_levels[0]}'")
        else:
            levels_str = "', '".join(B_levels)
            formula_B = Formula(f"{facname} in ['{levels_str}']")

        con = pair_contrast(
            formula_A,
            formula_B,
            name=con_name,
            where=where,
        )
        contrasts.append(con)

    return contrast_set(*contrasts)