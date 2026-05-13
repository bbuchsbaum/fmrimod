"""Baseline model specification functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NuisanceSpec:
    """Specification for nuisance regressors.
    
    Represents external nuisance regressors (like motion parameters)
    that should be included in the baseline model.
    
    Attributes
    ----------
    name : str
        Name of the nuisance variable
    data : object
        The nuisance data, preserved for evaluation in model context.
    """
    name: str
    data: object
    
    def __repr__(self) -> str:
        return f"NuisanceSpec(name='{self.name}')"


@dataclass(frozen=True)
class BlockSpec:
    """Specification for block variables.
    
    Represents variables that are constant within each scanning run/block,
    such as run indicators or block-specific covariates.
    
    Attributes
    ----------
    name : str
        Name of the block variable
    label : str, optional
        Optional label for the block variable
    """
    name: str
    label: str | None = None
    
    def __repr__(self) -> str:
        return f"BlockSpec(name='{self.name}')"


def nuisance(x: object) -> NuisanceSpec:
    """Create a nuisance variable specification.
    
    Nuisance variables are external regressors (like motion parameters,
    physiological signals, etc.) that should be included in the baseline
    model to account for non-neural variance in the fMRI signal.
    
    Parameters
    ----------
    x : object
        A matrix, array, or reference to nuisance data.
        Can be:
        - A numpy array or matrix
        - A pandas DataFrame
        - A string reference to be evaluated later
        - Objects that can be converted to a matrix
    
    Returns
    -------
    NuisanceSpec
        A nuisance specification object
    
    Examples
    --------
    >>> # Direct matrix
    >>> motion_params = np.random.randn(100, 6)
    >>> nuisance(motion_params)
    NuisanceSpec(name='motion_params')
    
    >>> # String reference (evaluated in model context)
    >>> nuisance('motion_regressors')
    NuisanceSpec(name='motion_regressors')
    
    Notes
    -----
    When used in a baseline model, nuisance regressors are included
    as-is without any transformation (unlike drift terms which use
    basis functions).
    
    For multi-run data, provide a list of matrices, one per run.
    """
    # Try to get the variable name
    if isinstance(x, str):
        name = x
    else:
        # Try to extract name from the variable
        # In actual use, this would be handled by the calling context
        name = getattr(x, '__name__', 'nuisance')
    
    return NuisanceSpec(name=name, data=x)


def block(x: object) -> BlockSpec:
    """Create a block variable specification.
    
    Block variables represent factors that are constant within each
    scanning run but may vary between runs. Common examples include
    run indicators, session variables, or scanner-specific parameters.
    
    Parameters
    ----------
    x : object
        A block variable or reference.
        Can be:
        - A string name of the block variable
        - An array of block indicators
        - Variables that identify blocks
    
    Returns
    -------
    BlockSpec
        A block specification object
    
    Examples
    --------
    >>> # String reference
    >>> block('run')
    BlockSpec(name='run')
    
    >>> # Direct block indicators
    >>> run_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    >>> block(run_ids)
    BlockSpec(name='run_ids')
    
    Notes
    -----
    When used in a baseline model with intercept='by_block', this
    creates separate intercepts for each unique value of the block
    variable.
    
    Block variables are typically used to:
    - Model run-specific baseline shifts
    - Account for session effects
    - Handle scanner drift differently per run
    """
    # Try to get the variable name
    if isinstance(x, str):
        name = x
        label = x
    else:
        # Try to extract name from the variable
        name = getattr(x, '__name__', 'block')
        label = name
    
    return BlockSpec(name=name, label=label)
