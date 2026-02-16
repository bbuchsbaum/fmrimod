"""Basis functions for fmrimod.

This module provides various basis functions for expanding
continuous variables in fMRI design matrices:

- Polynomial bases (Poly, NaturalPoly)
- Spline bases (BSpline, NaturalSpline)
- Transformations (Scale, ScaleWithin, RobustScale, Ident)
"""

from .base import ParametricBasis
from .polynomial import Poly, NaturalPoly
from .spline import BSpline, NaturalSpline
from .transform import Scale, ScaleWithin, RobustScale, Ident, Standardized
from .sub_basis import SubBasis, sub_basis

__all__ = [
    "ParametricBasis",
    "Poly",
    "NaturalPoly", 
    "BSpline",
    "NaturalSpline",
    "Scale",
    "ScaleWithin",
    "RobustScale",
    "Ident",
    "Standardized",
    "SubBasis",
    "sub_basis",
]