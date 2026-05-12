"""Contrast weight computation using singledispatch.

This module implements the generic contrast_weights() function that computes
numerical weights for different types of contrast specifications.
"""

from __future__ import annotations

from typing import Optional, Union, List, Dict, Any
from functools import singledispatch
import ast
import numpy as np
import re
import warnings

from ..types import Array
from .contrast_spec import (
    ContrastSpec, 
    UnitContrastSpec,
    PairContrastSpec,
    ColumnContrastSpec,
    PolyContrastSpec,
    OnewayContrastSpec,
    InteractionContrastSpec,
    ContrastFormulaSpec,
    ContrastSet
)


# Tolerance for sum-to-zero checks
CONTRAST_TOLERANCE = 1e-8


class Contrast:
    """Computed contrast with numerical weights.

    Holds the result of applying :func:`contrast_weights` to a
    :class:`ContrastSpec`. The ``weights`` matrix maps design-matrix
    columns (conditions) to contrast components.

    Parameters
    ----------
    term : object
        The event term this contrast applies to.
    name : str
        Human-readable contrast name.
    weights : Array
        Weight matrix of shape ``(n_conditions, n_contrasts)``.
        For a t-contrast this is ``(n_conditions, 1)``;
        for an F-contrast the second dimension is > 1.
    condnames : list of str
        Condition names corresponding to rows of ``weights``.
    contrast_spec : ContrastSpec
        The original specification that produced this contrast.

    Attributes
    ----------
    term : object
        Event term.
    name : str
        Contrast name.
    weights : Array
        Weight matrix.
    condnames : list of str
        Condition names.
    contrast_spec : ContrastSpec
        Original specification.

    Examples
    --------
    >>> spec = pair_contrast(Formula("condition == 'A'"),
    ...                      Formula("condition == 'B'"),
    ...                      name="A_vs_B")
    >>> con = contrast_weights(spec, term)
    >>> con.weights.shape
    (3, 1)
    >>> con.is_fcontrast
    False
    """

    def __init__(
        self,
        term: Any,
        name: str,
        weights: Array,
        condnames: List[str],
        contrast_spec: ContrastSpec
    ):
        """Initialize a computed contrast.

        Parameters
        ----------
        term : object
            Event term this contrast applies to.
        name : str
            Contrast name.
        weights : Array
            Weight matrix, shape ``(n_conditions, n_contrasts)``.
        condnames : list of str
            Condition names (row labels for ``weights``).
        contrast_spec : ContrastSpec
            Original specification object.
        """
        self.term = term
        self.name = name
        self.weights = weights
        self.condnames = condnames
        self.contrast_spec = contrast_spec
    
    @property
    def is_fcontrast(self) -> bool:
        """Check if this is an F-contrast (multiple columns)."""
        return self.weights.ndim > 1 and self.weights.shape[1] > 1
    
    def __repr__(self) -> str:
        """String representation."""
        n_weights = self.weights.shape[0]
        n_cols = self.weights.shape[1] if self.weights.ndim > 1 else 1
        ctype = "F-contrast" if self.is_fcontrast else "t-contrast"
        return (f"Contrast(name='{self.name}', type={ctype}, "
                f"weights={n_weights}x{n_cols})")


@singledispatch
def contrast_weights(x, term=None, **kwargs):
    """Compute contrast weights for a specification.
    
    This is a generic function that dispatches to specific implementations
    based on the type of contrast specification.
    
    Parameters
    ----------
    x : ContrastSpec
        Contrast specification
    term : object
        Term to compute weights for
    **kwargs
        Additional arguments for specific implementations
        
    Returns
    -------
    Contrast
        Computed contrast with weights
        
    Raises
    ------
    NotImplementedError
        If no method exists for the specification type
    """
    raise NotImplementedError(
        f"No contrast_weights method for type {type(x).__name__}"
    )


def _calculate_mask_weights(
    names: List[str],
    A_mask: np.ndarray,
    B_mask: Optional[np.ndarray] = None,
    tol: float = CONTRAST_TOLERANCE
) -> np.ndarray:
    """Calculate contrast weights from logical masks.
    
    Parameters
    ----------
    names : list of str
        All condition names
    A_mask : array-like
        Boolean mask for group A
    B_mask : array-like, optional
        Boolean mask for group B
    tol : float
        Tolerance for sum-to-zero check
        
    Returns
    -------
    array
        Named weight vector
    """
    A_mask = np.asarray(A_mask, dtype=bool)
    nA = np.sum(A_mask)
    nB = 0
    
    if B_mask is not None:
        B_mask = np.asarray(B_mask, dtype=bool)
        if np.any(A_mask & B_mask):
            raise ValueError("Masks for group A and group B overlap")
        nB = np.sum(B_mask)
    
    # Check for empty selection
    if nA == 0 and (B_mask is None or nB == 0):
        raise ValueError(
            "No conditions were selected by the provided mask(s). "
            "Check your contrast specification."
        )
    
    # Initialize weights
    w = np.zeros(len(names))
    
    # Assign weights
    if nA > 0:
        w[A_mask] = 1.0 / nA
    
    if B_mask is not None and nB > 0:
        w[B_mask] = -1.0 / nB
    
    # Warnings for unbalanced contrasts
    if B_mask is not None:
        if nA == 0 and nB > 0:
            warnings.warn(
                "Mask A is empty but Mask B is not. "
                "Weights will not sum to zero."
            )
        elif nB == 0 and nA > 0:
            warnings.warn(
                "Mask B is empty but Mask A is not. "
                "Weights will not sum to zero."
            )
        elif nA > 0 and nB > 0 and abs(np.sum(w)) > tol:
            warnings.warn(
                f"Weights do not sum to zero (sum={np.sum(w):.6f})"
            )
    
    return w


def _get_conditions(term, expand_basis: bool = True) -> List[str]:
    """Get condition names from a term.
    
    Parameters
    ----------
    term : object
        Term object (must have conditions method)
    expand_basis : bool
        Whether to expand basis functions
        
    Returns
    -------
    list of str
        Condition names
    """
    # This would interface with the actual term implementation
    # For now, we'll assume the term has a conditions method
    if hasattr(term, 'conditions'):
        return term.conditions(drop_empty=False, expand_basis=expand_basis)
    else:
        # Fallback - try to get from design matrix columns
        if hasattr(term, 'design_matrix'):
            dm = term.design_matrix
            if hasattr(dm, 'columns'):
                return list(dm.columns)
            elif hasattr(dm, 'shape'):
                # Unnamed columns
                return [f"V{i+1}" for i in range(dm.shape[1])]
    
    warnings.warn(f"Could not get conditions for term {term}")
    return []


def _get_cells(term) -> Any:
    """Get cells (categorical combinations) from a term.

    Parameters
    ----------
    term : object
        Term object

    Returns
    -------
    DataFrame-like
        Cells data
    """
    if hasattr(term, 'cells'):
        return term.cells()
    else:
        # Return empty DataFrame-like object
        import pandas as pd
        return pd.DataFrame()


def _parse_formula_condition(formula_expr: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a formula expression to extract factor name and level.

    Handles expressions like:
    - "condition == 'A'" -> ('condition', 'A')
    - "category == 'face'" -> ('category', 'face')
    - "time" -> ('time', None)

    Parameters
    ----------
    formula_expr : str
        Formula expression string

    Returns
    -------
    tuple of (str or None, str or None)
        (factor_name, level_name) or (None, None) if cannot parse
    """
    if not formula_expr:
        return None, None

    # Try to match pattern: variable == 'value' or variable == "value"
    match = re.match(r"^\s*(\w+)\s*==\s*['\"]([^'\"]+)['\"]\s*$", formula_expr)
    if match:
        return match.group(1), match.group(2)

    # Try to match just a variable name
    match = re.match(r"^\s*(\w+)\s*$", formula_expr)
    if match:
        return match.group(1), None

    return None, None


def _match_conditions_to_formula(condnames: List[str], formula_expr: str) -> np.ndarray:
    """Match condition names against a formula expression.

    Parameters
    ----------
    condnames : list of str
        All condition names (e.g., ["condition.A", "condition.B", "condition.C"])
    formula_expr : str
        Formula expression (e.g., "condition == 'A'")

    Returns
    -------
    np.ndarray
        Boolean mask indicating which conditions match
    """
    mask = np.zeros(len(condnames), dtype=bool)

    # Check for special keywords first
    formula_lower = formula_expr.strip().lower()
    if formula_lower in ("all", "~1", "*"):
        # Match all conditions
        mask[:] = True
        return mask

    # Parse the formula
    factor_name, level_name = _parse_formula_condition(formula_expr)

    if factor_name is None:
        return mask

    # Match against condition names
    # Condition names can have format: "factor.level", "level", or just "factor"
    for i, cond in enumerate(condnames):
        if level_name is not None:
            # Look for exact match of "factor.level" pattern
            # Using naming.level_token format
            expected_token = f"{factor_name}.{level_name}"
            if cond == expected_token or cond.startswith(f"{expected_token}_"):
                mask[i] = True
            # Also match if condition name is just the level (for simple tests)
            elif cond == level_name or cond.startswith(f"{level_name}_"):
                mask[i] = True
        else:
            # Just matching factor name - match any condition with that factor
            # First check if it's a factor name (matches start of condition)
            if cond.startswith(f"{factor_name}.") or cond == factor_name:
                mask[i] = True
            # Also check if it's a bare level name (matches end of condition after ".")
            elif "." in cond:
                # Extract level from "factor.level" format
                parts = cond.split(".", 1)
                if len(parts) == 2 and parts[1].startswith(factor_name):
                    # Match if factor_name matches the level part
                    # E.g., "A" matches "condition.A"
                    level_part = parts[1].split("_")[0]  # Handle "A_basis1" cases
                    if level_part == factor_name:
                        mask[i] = True

    return mask


@contrast_weights.register(UnitContrastSpec)
def _(x: UnitContrastSpec, term, **kwargs) -> Contrast:
    """Compute weights for unit contrast."""
    # Get all condition names
    all_condnames = _get_conditions(term, expand_basis=False)

    if not all_condnames:
        warnings.warn(f"No conditions found for term {term}")
        weights = np.zeros((0, 1))
        return Contrast(term, x.name, weights, [], x)

    # Parse the A formula to find matching conditions
    if x.A is not None:
        formula_expr = x.A.expr if hasattr(x.A, 'expr') else str(x.A)
        mask_A = _match_conditions_to_formula(all_condnames, formula_expr)
    else:
        # No formula - select all conditions
        mask_A = np.ones(len(all_condnames), dtype=bool)

    # Calculate weights - unit contrast sums to 1
    # For unit contrast, B_mask should be None (not a pairwise contrast)
    nA = np.sum(mask_A)

    if nA == 0:
        warnings.warn(
            f"Unit contrast '{x.name}': No conditions were selected by formula. "
            f"Formula: '{formula_expr}', Conditions: {all_condnames}"
        )
        weights_vec = np.zeros(len(all_condnames))
    else:
        # Unit contrast: selected conditions get equal positive weight summing to 1
        weights_vec = np.zeros(len(all_condnames))
        weights_vec[mask_A] = 1.0 / nA

    weights = weights_vec.reshape(-1, 1)

    return Contrast(
        term=term,
        name=x.name,
        weights=weights,
        condnames=all_condnames,
        contrast_spec=x
    )


@contrast_weights.register(PairContrastSpec)
def _(x: PairContrastSpec, term, **kwargs) -> Contrast:
    """Compute weights for pairwise contrast."""
    # Get condition names
    base_condnames = _get_conditions(term, expand_basis=False)

    if not base_condnames:
        warnings.warn(f"No conditions found for term {term}")
        weights = np.zeros((0, 1))
        return Contrast(term, x.name, weights, [], x)

    # Parse formulas to find matching conditions
    if x.A is not None:
        formula_A_expr = x.A.expr if hasattr(x.A, 'expr') else str(x.A)
        mask_A = _match_conditions_to_formula(base_condnames, formula_A_expr)
    else:
        mask_A = np.zeros(len(base_condnames), dtype=bool)

    if x.B is not None:
        formula_B_expr = x.B.expr if hasattr(x.B, 'expr') else str(x.B)
        mask_B = _match_conditions_to_formula(base_condnames, formula_B_expr)
    else:
        mask_B = np.zeros(len(base_condnames), dtype=bool)

    # Calculate weights
    try:
        weights_vec = _calculate_mask_weights(base_condnames, mask_A, mask_B)
    except ValueError as e:
        warnings.warn(f"Contrast '{x.name}': {e}")
        weights_vec = np.zeros(len(base_condnames))

    weights = weights_vec.reshape(-1, 1)

    # Check for basis expansion
    if hasattr(term, 'nbasis') and term.nbasis > 1:
        # Expand weights for basis functions
        expanded_condnames = _get_conditions(term, expand_basis=True)
        if len(expanded_condnames) == len(base_condnames) * term.nbasis:
            # Repeat weights for each basis function
            weights_expanded = np.repeat(weights_vec, term.nbasis)
            weights = weights_expanded.reshape(-1, 1)
            condnames = expanded_condnames
        else:
            condnames = base_condnames
    else:
        condnames = base_condnames

    return Contrast(
        term=term,
        name=x.name,
        weights=weights,
        condnames=condnames,
        contrast_spec=x
    )


@contrast_weights.register(ColumnContrastSpec)
def _(x: ColumnContrastSpec, term, **kwargs) -> Contrast:
    """Compute weights for column contrast using regex patterns."""
    # Get expanded column names
    all_colnames = _get_conditions(term, expand_basis=True)
    
    if not all_colnames:
        warnings.warn(
            f"Term '{getattr(term, 'varname', 'unknown')}' has no columns"
        )
        weights = np.zeros((0, 1))
        return Contrast(term, x.name, weights, [], x)
    
    # Find matching columns
    idx_A = [i for i, name in enumerate(all_colnames) 
             if re.search(x.pattern_A, name)]
    nA = len(idx_A)
    
    if nA == 0:
        warnings.warn(
            f"Pattern '{x.pattern_A}' matched no columns in term "
            f"'{getattr(term, 'varname', 'unknown')}'"
        )
    
    idx_B = []
    if x.pattern_B is not None:
        idx_B = [i for i, name in enumerate(all_colnames)
                 if re.search(x.pattern_B, name)]
        nB = len(idx_B)
        
        if nB == 0:
            warnings.warn(
                f"Pattern '{x.pattern_B}' matched no columns in term "
                f"'{getattr(term, 'varname', 'unknown')}'"
            )
        
        # Check for overlap
        if set(idx_A) & set(idx_B):
            raise ValueError(
                f"Patterns '{x.pattern_A}' and '{x.pattern_B}' match "
                "overlapping columns"
            )
    
    # Create masks
    mask_A = np.zeros(len(all_colnames), dtype=bool)
    mask_A[idx_A] = True
    
    mask_B = None
    if x.pattern_B is not None:
        mask_B = np.zeros(len(all_colnames), dtype=bool)
        mask_B[idx_B] = True
    
    # Calculate weights
    try:
        weights_vec = _calculate_mask_weights(all_colnames, mask_A, mask_B)
    except ValueError as e:
        warnings.warn(f"Column contrast '{x.name}': {e}")
        weights_vec = np.zeros(len(all_colnames))
    
    weights = weights_vec.reshape(-1, 1)
    
    return Contrast(
        term=term,
        name=x.name,
        weights=weights,
        condnames=all_colnames,
        contrast_spec=x
    )


@contrast_weights.register(PolyContrastSpec)
def _(x: PolyContrastSpec, term, **kwargs) -> Contrast:
    """Compute polynomial contrast weights."""
    # Get condition names
    all_condnames = _get_conditions(term, expand_basis=False)

    if not all_condnames:
        warnings.warn(f"No conditions found for term {term}")
        weights = np.zeros((0, x.degree))
        return Contrast(term, x.name, weights, [], x)

    # Number of conditions
    n_cond = len(all_condnames)

    if n_cond <= x.degree:
        raise ValueError(
            f"Polynomial degree ({x.degree}) is too high for "
            f"number of conditions ({n_cond}). Need at least {x.degree + 1} levels."
        )

    # Generate orthogonal polynomial contrasts using Legendre polynomials
    # Map conditions to numeric values 0, 1, 2, ..., n_cond-1
    vals = np.arange(n_cond)

    # Use Legendre polynomials for orthogonal basis
    from numpy.polynomial.legendre import legvander

    # Scale values to [-1, 1] interval for better numerical stability
    if n_cond > 1:
        vals_scaled = 2 * (vals - vals.min()) / (vals.max() - vals.min()) - 1
    else:
        vals_scaled = vals

    # Generate Vandermonde matrix up to degree x.degree
    # legvander generates columns [1, L_1(x), L_2(x), ..., L_degree(x)]
    # We want columns 1 through degree (skip the constant column 0)
    vander = legvander(vals_scaled, x.degree)

    # Take columns 1 through degree (skip constant term)
    poly_matrix = vander[:, 1:(x.degree + 1)]

    # Orthonormalize the columns
    Q, R = np.linalg.qr(poly_matrix)

    # Each column of Q is an orthogonal polynomial contrast
    weights = Q

    return Contrast(
        term=term,
        name=x.name,
        weights=weights,
        condnames=all_condnames,
        contrast_spec=x
    )


@contrast_weights.register(OnewayContrastSpec)
def _(x: OnewayContrastSpec, term, **kwargs) -> Contrast:
    """Compute one-way contrast weights (main effect).

    Generates k-1 Helmert contrasts for k conditions.
    Each contrast i compares condition i+1 against the mean of conditions 0..i.
    """
    all_condnames = _get_conditions(term, expand_basis=False)

    if not all_condnames:
        warnings.warn(f"No conditions found for term {term}")
        weights = np.zeros((0, 0))
        return Contrast(term, x.name, weights, [], x)

    k = len(all_condnames)

    if k < 2:
        warnings.warn(f"Factor has only {k} level(s), cannot create contrast")
        weights = np.zeros((k, 0))
        return Contrast(term, x.name, weights, all_condnames, x)

    # Generate k-1 Helmert contrasts
    # Each row i contrasts condition i+1 against mean of conditions 0..i
    weights = np.zeros((k, k - 1))

    for i in range(k - 1):
        # Contrast i: compare condition i+1 vs mean of conditions 0..i
        # Conditions 0 to i get weight -1/(i+1)
        # Condition i+1 gets weight 1
        weights[:(i + 1), i] = -1.0 / (i + 1)
        weights[i + 1, i] = 1.0

    return Contrast(
        term=term,
        name=x.name,
        weights=weights,
        condnames=all_condnames,
        contrast_spec=x
    )


@contrast_weights.register(InteractionContrastSpec)
def _(x: InteractionContrastSpec, term, **kwargs) -> Contrast:
    """Compute interaction contrast weights.

    For a term with n conditions representing a factorial design,
    generates interaction contrast matrix using Kronecker products.
    For 2x2: [1, -1, -1, 1]
    For larger designs, uses Kronecker product of component contrasts.
    """
    all_condnames = _get_conditions(term, expand_basis=False)

    if not all_condnames:
        warnings.warn(f"No conditions found for term {term}")
        weights = np.zeros((0, 0))
        return Contrast(term, x.name, weights, [], x)

    n = len(all_condnames)

    if n < 4:
        warnings.warn(
            f"Interaction requires at least 4 cells, found {n}"
        )
        weights = np.zeros((n, 0))
        return Contrast(term, x.name, weights, all_condnames, x)

    # Determine the factorial structure
    # For simplicity, assume 2x2 for 4 conditions, 2x3 for 6, 3x3 for 9, etc.
    # Find factors that multiply to n
    if n == 4:
        # 2x2 interaction
        weights = np.array([[1], [-1], [-1], [1]], dtype=float)
    elif n == 6:
        # 2x3 interaction: 2 contrasts (1 x 2)
        # Factor 1: 2 levels (1 contrast), Factor 2: 3 levels (2 contrasts)
        # Interaction has 1*2 = 2 contrasts
        c1 = np.array([[1], [-1]])  # Factor 1 contrast
        c2 = np.array([[1, -1], [-1, 0], [0, 1]])  # Factor 2 contrasts (Helmert-like)
        weights = np.kron(c1, c2)
    elif n == 9:
        # 3x3 interaction: 4 contrasts (2 x 2)
        c1 = np.array([[1, -1], [-1, 0], [0, 1]])  # 3 levels -> 2 contrasts
        c2 = np.array([[1, -1], [-1, 0], [0, 1]])  # 3 levels -> 2 contrasts
        weights = np.kron(c1, c2)
    else:
        # General case: try to find factors
        # For now, use a simple pattern for 2xk designs
        import math
        # Try to factorize n
        k = n // 2
        if n == 2 * k:
            # Assume 2 x k design
            c1 = np.array([[1], [-1]])  # 2 levels -> 1 contrast
            # k levels -> k-1 contrasts (Helmert)
            c2 = np.zeros((k, k - 1))
            for i in range(k - 1):
                c2[:(i + 1), i] = -1.0 / (i + 1)
                c2[i + 1, i] = 1.0
            weights = np.kron(c1, c2)
        else:
            # Fallback: create simple contrasts
            warnings.warn(
                f"Cannot determine factorial structure for {n} conditions. "
                "Using simplified interaction contrast."
            )
            weights = np.zeros((n, 1))
            # Simple 2-way pattern
            for i in range(min(4, n)):
                weights[i, 0] = [1, -1, -1, 1][i]

    return Contrast(
        term=term,
        name=x.name,
        weights=weights,
        condnames=all_condnames,
        contrast_spec=x
    )


@contrast_weights.register(ContrastFormulaSpec)
def _(x: ContrastFormulaSpec, term, **kwargs) -> Contrast:
    """Compute weights for formula-based contrast.

    Parses formula expressions and computes proper weights. Supports:
    - Simple condition selection: "condition == 'A'"
    - Arithmetic operations: +, -, *, /
    - Parentheses for grouping
    - Simple condition names: "A" maps to condition matching "A"
    """
    import ast

    all_condnames = _get_conditions(term, expand_basis=False)
    if not all_condnames:
        weights = np.zeros((0, 1))
        return Contrast(term, x.name, weights, [], x)

    # Get the formula expression
    formula_expr = x.A.expr if hasattr(x.A, 'expr') else str(x.A)

    # Parse and evaluate the formula
    try:
        weights_vec = _evaluate_formula_expression(formula_expr, all_condnames)
    except Exception as e:
        warnings.warn(
            f"Formula contrast '{x.name}': Failed to parse formula '{formula_expr}': {e}. "
            "Using zero weights."
        )
        weights_vec = np.zeros(len(all_condnames))

    weights = weights_vec.reshape(-1, 1)

    # Check for basis expansion
    if hasattr(term, 'nbasis') and term.nbasis > 1:
        # Expand weights for basis functions
        expanded_condnames = _get_conditions(term, expand_basis=True)
        if len(expanded_condnames) == len(all_condnames) * term.nbasis:
            # Repeat weights for each basis function
            weights_expanded = np.repeat(weights_vec, term.nbasis)
            weights = weights_expanded.reshape(-1, 1)
            condnames = expanded_condnames
        else:
            condnames = all_condnames
    else:
        condnames = all_condnames

    return Contrast(
        term=term,
        name=x.name,
        weights=weights,
        condnames=condnames,
        contrast_spec=x
    )


def _evaluate_formula_expression(formula_expr: str, condnames: List[str]) -> np.ndarray:
    """Evaluate a formula expression to produce weight vector.

    Parameters
    ----------
    formula_expr : str
        Formula expression (e.g., "condition == 'A' - condition == 'B'")
    condnames : list of str
        All condition names

    Returns
    -------
    np.ndarray
        Weight vector
    """
    import ast
    import re

    # Preprocess: wrap comparison patterns in parentheses to fix operator precedence
    # Pattern: word == 'string' -> (word == 'string')
    # This prevents Python from parsing "condition == 'A' + condition == 'B'" as
    # a chained comparison "condition == ('A' + condition) == 'B'"
    preprocessed = re.sub(r'(\w+)\s*==\s*\'([^\']*)\'', r"(\1 == '\2')", formula_expr)

    # Parse the formula expression into an AST
    try:
        tree = ast.parse(preprocessed, mode='eval')
    except SyntaxError as e:
        # Try treating it as a simple condition name
        mask = _match_conditions_to_formula(condnames, formula_expr)
        weights = np.zeros(len(condnames), dtype=float)
        weights[mask] = 1.0
        return weights

    # Evaluate the AST
    return _eval_formula_ast(tree.body, condnames)


def _eval_formula_ast(node: ast.AST, condnames: List[str]) -> np.ndarray:
    """Recursively evaluate formula AST to produce weight vector.

    Parameters
    ----------
    node : ast.AST
        AST node to evaluate
    condnames : list of str
        All condition names

    Returns
    -------
    np.ndarray
        Weight vector
    """
    import ast

    if isinstance(node, ast.BinOp):
        # Binary operation: +, -, *, /
        left = _eval_formula_ast(node.left, condnames)
        right = _eval_formula_ast(node.right, condnames)

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    elif isinstance(node, ast.UnaryOp):
        # Unary operation: -, +
        operand = _eval_formula_ast(node.operand, condnames)

        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    elif isinstance(node, ast.Compare):
        # Comparison: condition == 'A'
        # After preprocessing, all comparisons are properly parenthesized,
        # so we only need to handle simple single comparisons

        # Single simple comparison: condition == 'A'
        # Try to use astor if available, otherwise reconstruct manually
        try:
            import astor
            expr_str = astor.to_source(node).strip()
        except (ImportError, Exception):
            # Fallback: reconstruct manually
            expr_str = _reconstruct_compare(node)

        mask = _match_conditions_to_formula(condnames, expr_str)
        weights = np.zeros(len(condnames), dtype=float)
        weights[mask] = 1.0
        return weights

    elif isinstance(node, ast.Name):
        # Simple variable name: "A"
        var_name = node.id
        mask = _match_conditions_to_formula(condnames, var_name)
        weights = np.zeros(len(condnames), dtype=float)
        weights[mask] = 1.0
        return weights

    elif isinstance(node, ast.Constant):
        # Numeric constant
        value = node.value

        # Return a constant vector
        return np.full(len(condnames), float(value))

    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def _reconstruct_compare(node: ast.Compare) -> str:
    """Reconstruct a comparison expression from AST node.

    Parameters
    ----------
    node : ast.Compare
        Comparison AST node

    Returns
    -------
    str
        Reconstructed expression
    """
    import ast

    # Get left side (should be a Name node for simple comparisons)
    if isinstance(node.left, ast.Name):
        left = node.left.id
    elif isinstance(node.left, ast.BinOp):
        # This is a complex case - the comparison's left side is arithmetic
        # This shouldn't be called for these cases, they should be handled
        # in _eval_formula_ast before calling _reconstruct_compare
        raise ValueError(f"Unsupported left side in comparison: {type(node.left).__name__}")
    else:
        raise ValueError(f"Unsupported left side in comparison: {type(node.left).__name__}")

    # Get operator (should be Eq for ==)
    if len(node.ops) != 1:
        raise ValueError("Multiple comparison operators not supported")

    op = node.ops[0]
    if not isinstance(op, ast.Eq):
        raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")

    # Get right side (should be a string constant)
    if len(node.comparators) != 1:
        raise ValueError("Multiple comparators not supported")

    right_node = node.comparators[0]
    if isinstance(right_node, ast.Constant):
        right = right_node.value
    else:
        raise ValueError(f"Unsupported right side in comparison: {type(right_node).__name__}")

    # Reconstruct the expression
    return f"{left} == '{right}'"


@contrast_weights.register(ContrastSet)
def _(x: ContrastSet, term, **kwargs) -> Dict[str, Contrast]:
    """Compute weights for a set of contrasts.

    Returns
    -------
    dict
        Dictionary mapping contrast names to Contrast objects
    """
    results = {}

    for contrast_spec in x:
        try:
            con = contrast_weights(contrast_spec, term, **kwargs)
            results[con.name] = con
        except Exception as e:
            warnings.warn(
                f"Failed to compute weights for contrast '{contrast_spec.name}': {e}"
            )

    return results


def _contrast_weights_event_model(model, **kwargs):
    """Compute contrast weights for an EventModel.

    Extracts contrasts from each term's contrast_specs attribute and maps
    local term weights to full design matrix space.

    Parameters
    ----------
    model : EventModel
        Event model with terms that may have contrast_specs
    **kwargs
        Additional arguments passed to contrast_weights

    Returns
    -------
    dict
        Nested dictionary: {term_name: {contrast_name: Contrast}}
        where Contrast objects have offset_weights in full design matrix space
    """
    from ..utils import term_indices

    # Get term indices mapping
    term_idx_map = term_indices(model)
    ncond = model.design_matrix.shape[1]

    # Result structure: {term_name: {contrast_name: Contrast}}
    result = {}

    # Process each term
    for term in model.terms:
        term_name = term.name

        # Check if term has contrast_specs
        if not hasattr(term, 'contrast_specs') or term.contrast_specs is None:
            continue

        # Get the EventTerm object for this term
        event_terms = model._create_event_terms()
        event_term = None
        for i, t in enumerate(model.terms):
            if t.name == term_name:
                event_term = event_terms[i]
                break

        if event_term is None:
            warnings.warn(f"Could not find event term for '{term_name}'")
            continue

        # Get term indices in full design matrix
        if term_name not in term_idx_map:
            warnings.warn(f"Term '{term_name}' not found in term indices")
            continue

        term_indices_list = term_idx_map[term_name]

        # Compute contrasts for this term
        term_contrasts = {}

        # Handle both single ContrastSpec and ContrastSet
        contrast_specs = term.contrast_specs
        if not isinstance(contrast_specs, (list, ContrastSet)):
            contrast_specs = [contrast_specs]

        for contrast_spec in contrast_specs:
            try:
                # Compute local weights for this term
                local_contrast = contrast_weights(contrast_spec, event_term, **kwargs)

                # Map local weights to full design matrix space
                offset_weights = np.zeros((ncond, local_contrast.weights.shape[1]))

                # Place local weights at appropriate indices
                for i, idx in enumerate(term_indices_list):
                    if i < local_contrast.weights.shape[0]:
                        offset_weights[idx, :] = local_contrast.weights[i, :]

                # Create new Contrast with offset weights
                full_contrast = Contrast(
                    term=term,
                    name=local_contrast.name,
                    weights=offset_weights,
                    condnames=model.column_names,
                    contrast_spec=contrast_spec
                )

                # Store the offset weights as an attribute for access
                full_contrast.offset_weights = offset_weights
                full_contrast.local_weights = local_contrast.weights

                term_contrasts[local_contrast.name] = full_contrast

            except Exception as e:
                warnings.warn(
                    f"Failed to compute contrast '{contrast_spec.name}' "
                    f"for term '{term_name}': {e}"
                )

        if term_contrasts:
            result[term_name] = term_contrasts

    return result


def _register_event_model_contrast_weights():
    """Register EventModel handler after module load to avoid circular imports."""
    try:
        from ..design.event_model import EventModel
        contrast_weights.register(EventModel)(_contrast_weights_event_model)
    except ImportError:
        pass


# Call registration at module load time
_register_event_model_contrast_weights()
