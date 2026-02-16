"""Baseline term implementation."""

from __future__ import annotations

from typing import List, Optional, Union
import numpy as np
import pandas as pd

from ..types import Array


class BaselineTerm:
    """A baseline term in an fMRI model.
    
    Represents a component of the baseline model (drift, block effects, nuisance)
    with support for block-wise structure.
    
    Parameters
    ----------
    varname : str
        Variable name for the term
    design_matrix : pd.DataFrame or Array
        The design matrix for this term
    colind : list of lists
        Column indices for each block
    rowind : list of lists
        Row indices for each block
    
    Attributes
    ----------
    varname : str
        Name of the term
    design_matrix : pd.DataFrame
        Design matrix as DataFrame
    colind : list
        Column indices per block
    rowind : list  
        Row indices per block
    """
    
    def __init__(
        self,
        varname: str,
        design_matrix: Union[pd.DataFrame, Array],
        colind: List[List[int]],
        rowind: List[List[int]]
    ):
        """Initialize baseline term.

        Parameters
        ----------
        varname : str
            Variable name identifying the term (e.g.,
            ``'drift'``, ``'block'``, ``'nuisance'``).
        design_matrix : pd.DataFrame or Array
            The design matrix for this term. Converted to
            a DataFrame internally if not already one.
        colind : list of list of int
            Column indices for each block. ``colind[i]`` lists
            the column positions belonging to block ``i``.
        rowind : list of list of int
            Row indices for each block. ``rowind[i]`` lists
            the row positions belonging to block ``i``.
        """
        self.varname = varname
        
        # Ensure design matrix is DataFrame
        if isinstance(design_matrix, pd.DataFrame):
            self.design_matrix = design_matrix
        else:
            self.design_matrix = pd.DataFrame(design_matrix)
        
        self.colind = colind
        self.rowind = rowind
    
    def get_block_matrix(
        self,
        blockid: Optional[Union[int, List[int]]] = None,
        allrows: bool = False
    ) -> pd.DataFrame:
        """Get design matrix for specific blocks.
        
        Parameters
        ----------
        blockid : int or list of int, optional
            Block ID(s) to extract. If None, returns full matrix.
        allrows : bool, default=False
            If True, return all rows but only columns for specified blocks.
            If False, return only rows and columns for specified blocks.
        
        Returns
        -------
        pd.DataFrame
            Subset of design matrix
        """
        if blockid is None:
            return self.design_matrix
        
        # Ensure blockid is a list
        if not isinstance(blockid, (list, tuple)):
            blockid = [blockid]
        
        # Convert to 0-based if needed (R uses 1-based)
        blockid = [b - 1 if b > 0 else b for b in blockid]
        
        # Get column indices for requested blocks
        col_idx = []
        for bid in blockid:
            if 0 <= bid < len(self.colind):
                col_idx.extend(self.colind[bid])
        
        if not col_idx:
            # No columns for these blocks
            return pd.DataFrame()
        
        if allrows:
            # Return all rows, selected columns
            return self.design_matrix.iloc[:, col_idx]
        else:
            # Get row indices for requested blocks
            row_idx = []
            for bid in blockid:
                if 0 <= bid < len(self.rowind):
                    row_idx.extend(self.rowind[bid])
            
            if not row_idx:
                return pd.DataFrame()
            
            # Return selected rows and columns
            return self.design_matrix.iloc[row_idx, col_idx]
    
    @property
    def n_blocks(self) -> int:
        """Number of blocks."""
        return len(self.colind)
    
    @property
    def n_columns(self) -> int:
        """Total number of columns."""
        return self.design_matrix.shape[1]
    
    @property
    def n_rows(self) -> int:
        """Total number of rows."""
        return self.design_matrix.shape[0]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BaselineTerm(varname='{self.varname}', "
            f"shape={self.design_matrix.shape}, "
            f"n_blocks={self.n_blocks})"
        )