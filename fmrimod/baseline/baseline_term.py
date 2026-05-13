"""Baseline term implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

from ..types import Array


@dataclass(frozen=True)
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
    
    varname: str
    design_matrix: pd.DataFrame | Array
    colind: Sequence[Sequence[int]]
    rowind: Sequence[Sequence[int]]

    def __post_init__(self) -> None:
        """Normalize array-like inputs to immutable term metadata."""
        if isinstance(self.design_matrix, pd.DataFrame):
            matrix = self.design_matrix.copy()
        else:
            matrix = pd.DataFrame(self.design_matrix)

        object.__setattr__(self, 'design_matrix', matrix)
        object.__setattr__(
            self,
            'colind',
            tuple(tuple(int(idx) for idx in block) for block in self.colind),
        )
        object.__setattr__(
            self,
            'rowind',
            tuple(tuple(int(idx) for idx in block) for block in self.rowind),
        )
    
    def get_block_matrix(
        self,
        blockid: int | Sequence[int] | None = None,
        allrows: bool = False,
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
        matrix = cast(pd.DataFrame, self.design_matrix)
        if blockid is None:
            return matrix
        
        if isinstance(blockid, int):
            block_ids = [int(blockid)]
        else:
            block_ids = [int(b) for b in blockid]
        
        # Convert to 0-based if needed (R uses 1-based)
        block_ids = [b - 1 if b > 0 else b for b in block_ids]
        
        # Get column indices for requested blocks
        col_idx: list[int] = []
        for bid in block_ids:
            if 0 <= bid < len(self.colind):
                col_idx.extend(self.colind[bid])
        
        if not col_idx:
            # No columns for these blocks
            return pd.DataFrame()
        
        if allrows:
            # Return all rows, selected columns
            return matrix.take(col_idx, axis=1)
        else:
            # Get row indices for requested blocks
            row_idx: list[int] = []
            for bid in block_ids:
                if 0 <= bid < len(self.rowind):
                    row_idx.extend(self.rowind[bid])
            
            if not row_idx:
                return pd.DataFrame()
            
            # Return selected rows and columns
            return matrix.take(row_idx, axis=0).take(col_idx, axis=1)
    
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
