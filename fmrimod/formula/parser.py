"""Formula parsing for fmrimod.

This module provides parsing for R-style formulas used in event models.
A subset of R's formula syntax is supported, adapted for fMRI design
matrices.

Supported syntax examples::

    onset ~ hrf(condition)
    onset ~ hrf(condition, basis="spm")
    onset ~ hrf(condition) + hrf(block)
    onset ~ hrf(condition:block)            # interaction
    condition + rating                       # no LHS (event_model shorthand)
    hrf(condition) + trialwise(add_sum=True) # mixed terms
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, cast

from ..formula.base import Term
from ..types import FormulaContext


@dataclass
class FormulaTerm:
    """Parsed representation of a single formula term.

    Parameters
    ----------
    function : str
        Function name (e.g., ``"hrf"``, ``"var"``, ``"trialwise"``).
    arguments : list
        Positional arguments extracted from the function call.
    kwargs : dict
        Keyword arguments extracted from the function call.
    """

    function: str  # e.g., "hrf"
    arguments: List[Union[str, "FormulaTerm"]]
    kwargs: Dict[str, object]
    
    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{self.function}({all_args})"


@dataclass
class Formula:
    """Complete parsed formula with left- and right-hand sides.

    Parameters
    ----------
    lhs : str
        Left-hand side variable (e.g., ``"onset"``). Empty string if
        the formula has no ``~`` separator.
    rhs : list of FormulaTerm
        Right-hand side terms.
    """

    lhs: str  # Left-hand side (e.g., "onset")
    rhs: List[FormulaTerm]  # Right-hand side terms
    
    def __repr__(self) -> str:
        rhs_str = " + ".join(str(term) for term in self.rhs)
        return f"{self.lhs} ~ {rhs_str}"


class FormulaParser:
    """Parser for R-style formulas used in fMRI event models.

    Supports ``lhs ~ rhs`` syntax where ``rhs`` is a ``+``-separated
    list of terms. Each term may be a bare variable name or a function
    call (e.g., ``hrf(condition)``). Interaction terms use the ``:``
    operator inside function calls.

    The parser uses Python's :mod:`ast` module to safely evaluate
    function-call arguments, so arbitrary Python literals (strings,
    numbers, lists, dicts) are accepted as keyword values.

    Examples
    --------
    >>> parser = FormulaParser()
    >>> f = parser.parse("onset ~ hrf(condition, hrf='spmg2')")
    >>> f.lhs
    'onset'
    >>> len(f.rhs)
    1
    """
    
    # Regex patterns
    FORMULA_PATTERN = re.compile(r'^([^~]+)~(.+)$')
    FUNCTION_PATTERN = re.compile(r'(\w+)\s*\(')
    
    def parse(self, formula_str: str) -> Formula:
        """Parse a formula string.
        
        Parameters
        ----------
        formula_str : str
            Formula string to parse
        
        Returns
        -------
        Formula
            Parsed formula object
        
        Raises
        ------
        ValueError
            If formula syntax is invalid
        """
        formula_str = formula_str.strip()
        
        # Split into LHS and RHS
        match = self.FORMULA_PATTERN.match(formula_str)
        if not match:
            # If no ~ found, check if the string looks like valid terms
            # A valid formula without ~ must contain function calls or be
            # a simple variable name (no spaces unless separated by +)
            if ' ' in formula_str and '+' not in formula_str and '(' not in formula_str:
                raise ValueError(
                    f"Invalid formula syntax: '{formula_str}'. "
                    "Expected 'lhs ~ rhs' format or valid term expressions."
                )
            lhs = ""
            rhs = formula_str
        else:
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
        
        # Parse RHS terms
        terms = self._parse_rhs(rhs)
        
        return Formula(lhs=lhs, rhs=terms)
    
    def _parse_rhs(self, rhs: str) -> List[FormulaTerm]:
        """Parse right-hand side of formula.
        
        Parameters
        ----------
        rhs : str
            Right-hand side string
        
        Returns
        -------
        list of FormulaTerm
            Parsed terms
        """
        # Split by + (but not inside parentheses)
        terms: list[FormulaTerm] = []
        current_term: list[str] = []
        paren_depth = 0
        
        for char in rhs:
            if char == '(' :
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '+' and paren_depth == 0:
                term_str = ''.join(current_term).strip()
                if term_str:
                    terms.append(self._parse_term(term_str))
                current_term = []
                continue
            current_term.append(char)
        
        # Don't forget the last term
        term_str = ''.join(current_term).strip()
        if term_str:
            terms.append(self._parse_term(term_str))
        
        return terms
    
    def _parse_term(self, term_str: str) -> FormulaTerm:
        """Parse a single term.

        Parameters
        ----------
        term_str : str
            Term string (e.g., "hrf(condition)")

        Returns
        -------
        FormulaTerm
            Parsed term
        """
        # Check for var:hrf(args) or var:function(args) pattern first
        var_func_match = re.match(r'^(\w+):(\w+)\(([^)]*)\)$', term_str)
        if var_func_match:
            var_name = var_func_match.group(1)
            func_name = var_func_match.group(2)
            hrf_arg = var_func_match.group(3).strip()
            if func_name == 'hrf':
                kwargs = {'hrf': hrf_arg} if hrf_arg else {}
                return FormulaTerm(function="hrf", arguments=[var_name], kwargs=kwargs)

        # Find function name
        match = self.FUNCTION_PATTERN.match(term_str)
        if not match:
            # Simple variable reference
            return FormulaTerm(function="var", arguments=[term_str], kwargs={})

        func_name = match.group(1)
        
        # Extract arguments using AST parsing for safety
        # Convert to valid Python syntax first
        py_expr = term_str.replace(':', '*')  # : means interaction
        
        try:
            # Parse as Python function call
            tree = ast.parse(py_expr, mode='eval')
            if not isinstance(tree.body, ast.Call):
                raise ValueError(f"Expected function call, got {type(tree.body)}")
            
            # Extract arguments
            args = []
            kwargs = {}
            
            # Positional arguments
            for arg in tree.body.args:
                args.append(self._ast_to_value(arg))
            
            # Keyword arguments
            for keyword in tree.body.keywords:
                if keyword.arg is None:
                    raise ValueError("Keyword **kwargs are not supported in formula terms")
                kwargs[keyword.arg] = self._ast_to_value(keyword.value)

            return FormulaTerm(
                function=func_name,
                arguments=cast("list[Union[str, FormulaTerm]]", args),
                kwargs=kwargs,
            )
            
        except (SyntaxError, ValueError) as err:
            raise ValueError(f"Invalid term syntax '{term_str}': {err}") from err
    
    def _ast_to_value(self, node: ast.AST) -> object:
        """Convert AST node to value.
        
        Parameters
        ----------
        node : ast.AST
            AST node
        
        Returns
        -------
        Any
            Extracted value
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            # Handle interaction terms (we converted : to *)
            left = self._ast_to_value(node.left)
            right = self._ast_to_value(node.right)
            return f"{left}:{right}"
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._ast_to_value(cast(ast.AST, k)): self._ast_to_value(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")


class FormulaEvaluator:
    """Evaluates parsed formulas to create event specifications.

    Given a :class:`Formula` and an optional :class:`FormulaContext`
    (which maps variable names to data arrays), the evaluator resolves
    variable references and HRF specifications into a dictionary
    suitable for constructing event objects.
    """
    
    def __init__(self, context: Optional[FormulaContext] = None):
        """Initialize evaluator.
        
        Parameters
        ----------
        context : FormulaContext, optional
            Context for variable resolution
        """
        self.context = context or FormulaContext()
    
    def evaluate(self, formula: Formula) -> Dict[str, object]:
        """Evaluate a parsed formula.
        
        Parameters
        ----------
        formula : Formula
            Parsed formula
        
        Returns
        -------
        dict
            Event specification dictionary
        """
        # Get onset variable
        onset_var = self._resolve_variable(formula.lhs)
        
        # Process terms
        events = []
        for term in formula.rhs:
            event_spec = self._evaluate_term(term)
            events.append(event_spec)
        
        return {
            'onset_var': formula.lhs,
            'onsets': onset_var,
            'events': events
        }
    
    def _evaluate_term(self, term: FormulaTerm) -> Dict[str, object]:
        """Evaluate a single term.
        
        Parameters
        ----------
        term : FormulaTerm
            Term to evaluate
        
        Returns
        -------
        dict
            Term specification
        """
        if term.function == "hrf":
            return self._evaluate_hrf_term(term)
        elif term.function == "var":
            # Plain variable reference
            var_name = term.arguments[0]
            return {
                'type': 'variable',
                'name': var_name,
                'values': self._resolve_variable(cast(str, var_name)),
            }
        else:
            raise ValueError(f"Unknown function: {term.function}")
    
    def _evaluate_hrf_term(self, term: FormulaTerm) -> Dict[str, object]:
        """Evaluate an hrf() term.
        
        Parameters
        ----------
        term : FormulaTerm
            HRF term
        
        Returns
        -------
        dict
            HRF specification
        """
        if not term.arguments:
            raise ValueError("hrf() requires at least one argument")
        
        # First argument is the event specification
        event_arg = term.arguments[0]
        
        # Handle interaction terms
        if isinstance(event_arg, str) and ':' in event_arg:
            parts = event_arg.split(':')
            event_spec = {
                'type': 'interaction',
                'factors': [self._resolve_variable(p.strip()) for p in parts]
            }
        else:
            # Single variable
            event_spec = cast(
                "Dict[str, Any]",
                {
                    'type': 'event',
                    'name': str(event_arg),
                    'values': self._resolve_variable(str(event_arg)),
                },
            )
        
        # Add HRF parameters
        hrf_spec = {
            'event': event_spec,
            'hrf': term.kwargs.get('hrf', 'spm_canonical'),
            'basis': term.kwargs.get('basis'),
            **term.kwargs  # Pass through any other parameters
        }
        
        return hrf_spec
    
    def _resolve_variable(self, name: str) -> object:
        """Resolve a variable name in the context.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        Any
            Variable value
        """
        return self.context.get(name)


def parse_formula(
    formula_str: str,
    context: Optional[FormulaContext] = None,
    for_event_model: bool = False,
) -> Union[Dict[str, object], List["Term"]]:
    """Parse and evaluate a formula string.
    
    This is the main entry point for formula parsing.
    
    Parameters
    ----------
    formula_str : str
        Formula string
    context : FormulaContext, optional
        Context for variable resolution
    for_event_model : bool, optional
        If True, always convert the RHS into ``Term`` objects and ignore
        LHS evaluation. This is used by ``event_model()`` to support
        R-style formulas like ``onset ~ hrf(condition)`` while treating
        onset handling as a separate concern.
    
    Returns
    -------
    dict or list of Term
        Event specification or list of Term objects for event models
    
    Examples
    --------
    >>> # Simple formula
    >>> parse_formula("onset ~ hrf(condition)", context)
    {'onset_var': 'onset', 'onsets': array(...), 'events': [...]}
    
    >>> # With HRF specification
    >>> parse_formula('onset ~ hrf(condition, hrf="gamma")', context)
    {'onset_var': 'onset', 'onsets': array(...), 'events': [...]}
    
    >>> # For event models (no LHS)
    >>> parse_formula("condition + rating")
    [Term('condition'), Term('rating')]
    """
    parser = FormulaParser()
    
    formula = parser.parse(formula_str)
    
    # Convert to Term objects for event model mode, or formulas with no LHS.
    if for_event_model or not formula.lhs:
        from ..formula.base import Term
        from ..trialwise import trialwise
        terms = []
        for ft in formula.rhs:
            # Handle different term types
            if ft.function == 'var':
                # Simple variable like 'condition'
                term = Term(cast(str, ft.arguments[0]))
            elif ft.function == 'hrf' and ft.arguments:
                # HRF function like hrf(condition)
                event_name = ft.arguments[0]
                event_arg: Union[str, list[str]]
                if isinstance(event_name, str) and ':' in event_name:
                    event_arg = [part.strip() for part in event_name.split(':')]
                else:
                    event_arg = cast(str, event_name)
                # Accept both hrf=... and basis=... for compatibility.
                hrf = ft.kwargs.get('hrf', ft.kwargs.get('basis', 'simple'))
                extra = {
                    key: value
                    for key, value in ft.kwargs.items()
                    if key not in {'hrf', 'basis', 'normalize', 'summate', 'id'}
                }
                term = Term(
                    event_arg,
                    hrf=hrf,
                    name=cast("Optional[str]", ft.kwargs.get('id')),
                    normalize=bool(ft.kwargs.get('normalize', False)),
                    summate=bool(ft.kwargs.get('summate', True)),
                )
                term._kwargs.update(extra)
            elif ft.function == 'trialwise':
                # trialwise() function
                term = trialwise(
                    basis=ft.kwargs.get('basis', 'spmg1'),
                    lag=ft.kwargs.get('lag', 0.0),
                    nbasis=ft.kwargs.get('nbasis', 1),
                    add_sum=ft.kwargs.get('add_sum', False),
                    label=ft.kwargs.get('label', 'trial')
                )
            elif ft.function and ft.arguments:
                # Other function with args - treat first arg as event
                term = Term(cast(str, ft.arguments[0]))
                # Add function as transformation
                if ft.function == 'poly':
                    # Handle polynomial basis
                    degree = ft.arguments[1] if len(ft.arguments) > 1 else 2
                    from ..basis import Poly
                    term.basis = Poly(degree=int(cast(Any, degree)))
            else:
                # Just a function name with no args
                term = Term(ft.function)
            
            terms.append(term)
        return terms
    
    # Otherwise evaluate normally
    evaluator = FormulaEvaluator(context)
    return evaluator.evaluate(formula)
