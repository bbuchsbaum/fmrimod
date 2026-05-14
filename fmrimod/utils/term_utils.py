"""Utility functions for working with model terms and design matrices."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..types import Array, ModelProtocol


def term_indices(model: ModelProtocol) -> Dict[str, List[int]]:
    """Extract column indices for each term in a model.
    
    Maps each term name to the column indices it occupies in the
    design matrix. This is useful for extracting term-specific
    sub-matrices or computing term-wise statistics.
    
    Parameters
    ----------
    model : ModelProtocol
        Model object with terms and column names
    
    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping term names to column indices
    
    Examples
    --------
    >>> # Create a model with multiple terms
    >>> model = event_model("condition + rating:hrf('spm')", data=df)
    >>> indices = term_indices(model)
    >>> indices
    {'condition': [0, 1, 2], 'rating': [3]}
    >>> 
    >>> # Use indices to extract term-specific columns
    >>> X = model.design_matrix
    >>> condition_cols = X[:, indices['condition']]
    """
    # First check if model has pre-computed column indices (like R implementation)
    if hasattr(model, 'column_indices') and model.column_indices is not None:
        return model.column_indices
    
    # Fallback to pattern matching if column_indices not available
    if not hasattr(model, 'terms') or not hasattr(model, 'column_names'):
        raise AttributeError("Model must have 'terms' and 'column_names' attributes")
    
    # Get column names from model
    col_names = model.column_names
    
    # Initialize mapping
    indices_map = {}
    
    # Track which columns have been assigned
    assigned = set()
    
    # Process each term
    for term in model.terms:
        term_name = term.name
        term_indices_list = []
        
        # Find columns belonging to this term
        for i, col_name in enumerate(col_names):
            if i in assigned:
                continue
                
            # Check if column belongs to this term
            # This handles various naming conventions
            if (col_name.startswith(term_name + '.') or  # factor levels
                col_name.startswith(term_name + ':') or  # interactions  
                col_name == term_name or  # exact match
                col_name.startswith(term_name + '_')):  # basis functions
                term_indices_list.append(i)
                assigned.add(i)
        
        if term_indices_list:
            indices_map[term_name] = term_indices_list
    
    # Handle any unassigned columns (e.g., from baseline model)
    unassigned = [i for i in range(len(col_names)) if i not in assigned]
    if unassigned:
        # Check if model has baseline terms
        if hasattr(model, 'baseline_model') and model.baseline_model is not None:
            indices_map['baseline'] = unassigned
        else:
            # Group unassigned columns as 'other'
            indices_map['other'] = unassigned
    
    return indices_map


def term_matrices(model: ModelProtocol, 
                  term_names: Optional[Union[str, List[str]]] = None) -> Dict[str, Array]:
    """Extract term-specific sub-matrices from the design matrix.
    
    Returns a dictionary mapping term names to their corresponding
    columns in the design matrix. Useful for term-wise analysis,
    variance decomposition, or selective model fitting.
    
    Parameters
    ----------
    model : ModelProtocol
        Model object with design matrix and terms
    term_names : str or list of str, optional
        Specific terms to extract. If None, extracts all terms.
    
    Returns
    -------
    Dict[str, Array]
        Dictionary mapping term names to sub-matrices
    
    Examples
    --------
    >>> # Extract all term matrices
    >>> model = event_model("condition + rating + condition:rating", data=df)
    >>> matrices = term_matrices(model)
    >>> matrices.keys()
    dict_keys(['condition', 'rating', 'condition:rating'])
    >>> 
    >>> # Extract specific terms
    >>> main_effects = term_matrices(model, ['condition', 'rating'])
    >>> 
    >>> # Get variance explained by each term
    >>> for term, mat in matrices.items():
    ...     var_explained = np.var(mat @ np.linalg.pinv(mat).T @ y)
    ...     print(f"{term}: {var_explained:.2f}")
    """
    # Get full design matrix
    if hasattr(model, 'design_matrix'):
        if callable(model.design_matrix):
            X = model.design_matrix()
        else:
            X = model.design_matrix
    else:
        raise AttributeError("Model must have 'design_matrix' attribute or method")
    
    # Get term to column mapping
    indices = term_indices(model)
    
    # Handle term selection
    if term_names is None:
        term_names = list(indices.keys())
    elif isinstance(term_names, str):
        term_names = [term_names]
    
    # Validate requested terms
    missing = set(term_names) - set(indices.keys())
    if missing:
        available = ', '.join(sorted(indices.keys()))
        raise ValueError(
            f"Terms not found in model: {missing}. "
            f"Available terms: {available}"
        )
    
    # Extract sub-matrices
    matrices = {}
    for term in term_names:
        if term in indices:
            cols = indices[term]
            matrices[term] = X[:, cols]
    
    return matrices


def baseline_terms(model: ModelProtocol) -> Optional[Array]:
    """Extract baseline/nuisance terms from a model.
    
    Returns the columns of the design matrix corresponding to
    baseline terms (drift, motion regressors, etc.). Returns
    None if no baseline model is present.
    
    Parameters
    ----------
    model : ModelProtocol
        Model object potentially containing baseline terms
    
    Returns
    -------
    Array or None
        Baseline term matrix or None if no baseline
    
    Examples
    --------
    >>> # Model with polynomial drift
    >>> model = event_model(
    ...     "condition", 
    ...     data=df,
    ...     baseline=baseline_model(degree=2)
    ... )
    >>> baseline = baseline_terms(model)
    >>> baseline.shape
    (300, 3)  # intercept + linear + quadratic
    >>> 
    >>> # Model without baseline
    >>> model = event_model("condition", data=df)
    >>> baseline_terms(model) is None
    True
    """
    # Check if model has baseline
    if hasattr(model, 'baseline_model') and model.baseline_model is not None:
        # Get baseline columns through term_matrices
        matrices = term_matrices(model, 'baseline')
        if 'baseline' in matrices:
            return matrices['baseline']
    
    # Alternative: check for baseline in term indices
    indices = term_indices(model)
    if 'baseline' in indices:
        if hasattr(model, 'design_matrix'):
            if callable(model.design_matrix):
                X = model.design_matrix()
            else:
                X = model.design_matrix
            return X[:, indices['baseline']]
    
    return None


def split_by_term(model: ModelProtocol) -> List[Tuple[str, Array]]:
    """Split design matrix by terms, preserving order.
    
    Returns a list of (term_name, sub_matrix) tuples in the
    order terms appear in the model. Useful for sequential
    analysis or reporting.
    
    Parameters
    ----------
    model : ModelProtocol
        Model object with terms and design matrix
    
    Returns
    -------
    List[Tuple[str, Array]]
        Ordered list of term names and their matrices
    
    Examples
    --------
    >>> model = event_model("condition + rating", data=df)
    >>> for term_name, term_matrix in split_by_term(model):
    ...     print(f"{term_name}: {term_matrix.shape}")
    condition: (300, 3)
    rating: (300, 1)
    """
    # Get term matrices
    matrices = term_matrices(model)
    
    # Order by term appearance in model
    ordered_terms = []
    for term in model.terms:
        if term.name in matrices:
            ordered_terms.append((term.name, matrices[term.name]))
    
    # Add baseline if present
    if 'baseline' in matrices:
        ordered_terms.append(('baseline', matrices['baseline']))
    
    return ordered_terms