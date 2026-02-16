"""Baseline model implementation for fMRI design matrices.

This module implements baseline models that account for drift, block-wise
intercepts, and nuisance regressors in fMRI time series.
"""

from __future__ import annotations

from typing import List, Optional, Union, Dict, Any, Literal
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.interpolate import BSpline as ScipyBSpline

from ..types import Array, SamplingInfo
from ..sampling import SamplingFrame
from .baseline_term import BaselineTerm


class BaselineModel:
    """Baseline model for fMRI time series.
    
    A baseline model represents low-frequency drift, block-wise intercepts,
    and nuisance regressors that are typically included in fMRI design matrices
    to account for non-neural signal variations.
    
    Parameters
    ----------
    terms : dict
        Dictionary of baseline terms (drift, block, nuisance)
    drift_spec : BaselineSpec
        Specification for drift modeling
    sampling_frame : SamplingInfo
        Sampling information
    
    Attributes
    ----------
    terms : dict
        Dictionary of baseline terms
    drift_spec : BaselineSpec
        Drift specification
    sampling_frame : SamplingInfo
        Sampling frame
    
    Examples
    --------
    >>> # Create baseline model with polynomial drift
    >>> bmodel = baseline_model(
    ...     basis='poly',
    ...     degree=3,
    ...     sframe=sampling_frame,
    ...     intercept='runwise'
    ... )
    >>> 
    >>> # Get design matrix
    >>> X_baseline = design_matrix(bmodel)
    """
    
    def __init__(
        self,
        terms: Dict[str, BaselineTerm],
        drift_spec: BaselineSpec,
        sampling_frame: SamplingInfo
    ):
        """Initialize baseline model.

        Parameters
        ----------
        terms : dict of str to BaselineTerm
            Dictionary of baseline terms keyed by role
            (``'drift'``, ``'block'``, ``'nuisance'``).
        drift_spec : BaselineSpec
            Specification object describing the drift basis.
        sampling_frame : SamplingInfo
            Sampling frame with block structure information.
        """
        self.terms = terms
        self.drift_spec = drift_spec
        self.sampling_frame = sampling_frame
    
    @property
    def design_matrix(self) -> Array:
        """Get full design matrix by combining all terms."""
        matrices = []
        
        for term_name in ['drift', 'block', 'nuisance']:
            if term_name in self.terms and self.terms[term_name] is not None:
                mat = self.terms[term_name].design_matrix
                if isinstance(mat, pd.DataFrame):
                    mat = mat.values
                matrices.append(mat)
        
        if matrices:
            return np.hstack(matrices)
        else:
            # Return empty matrix with correct number of rows
            n_rows = sum(getattr(self.sampling_frame, 'blocklens', []))
            return np.zeros((n_rows, 0))

    @property
    def column_names(self) -> List[str]:
        """Return baseline design-matrix column names in matrix order."""
        names: List[str] = []
        for term_name in ['drift', 'block', 'nuisance']:
            term = self.terms.get(term_name)
            if term is None:
                continue
            mat = term.design_matrix
            if isinstance(mat, pd.DataFrame):
                names.extend(list(mat.columns))
            else:
                n_cols = mat.shape[1] if mat.ndim > 1 else 1
                names.extend([f"{term_name}_{i + 1}" for i in range(n_cols)])
        return names

    @property
    def colnames(self) -> List[str]:
        """Compatibility alias used by generic column extractors."""
        return self.column_names
    
    def __repr__(self) -> str:
        """Rich string representation matching R's print.baseline_model()."""
        lines = ["BaselineModel"]
        lines.append(f"  Basis: {self.drift_spec.basis}, Degree: {self.drift_spec.degree}")
        lines.append(f"  Intercept: {self.drift_spec.intercept}")

        for term_name in ['drift', 'block', 'nuisance']:
            if term_name in self.terms and self.terms[term_name] is not None:
                term = self.terms[term_name]
                mat = term.design_matrix
                if isinstance(mat, pd.DataFrame):
                    cols = list(mat.columns)
                else:
                    cols = [f"col_{i}" for i in range(mat.shape[1] if mat.ndim > 1 else 1)]
                n_cols = len(cols)
                preview = ", ".join(cols[:4])
                if n_cols > 4:
                    preview += f", ... ({n_cols - 4} more)"
                lines.append(f"  {term_name.capitalize()}: {n_cols} columns [{preview}]")

        dm = self.design_matrix
        lines.append(f"  Design matrix: {dm.shape[0]} x {dm.shape[1]}")

        return "\n".join(lines)


class BaselineSpec:
    """Specification for baseline drift modeling.
    
    Parameters
    ----------
    degree : int
        Number of basis terms per block
    basis : str
        Type of basis ('constant', 'poly', 'bs', 'ns')
    intercept : str
        Type of intercept ('runwise', 'global', 'none')
    name : str, optional
        Name for the term
    """
    
    def __init__(
        self,
        degree: int = 1,
        basis: str = 'constant',
        intercept: str = 'runwise',
        name: Optional[str] = None
    ):
        """Initialize baseline specification.

        Parameters
        ----------
        degree : int, default=1
            Number of basis terms per block. Ignored when
            ``basis='constant'``.
        basis : {'constant', 'poly', 'bs', 'ns'}, default='constant'
            Type of drift basis function.
        intercept : {'runwise', 'global', 'none'}, default='runwise'
            Intercept strategy across runs.
        name : str, optional
            Label for this baseline specification.
        """
        self.degree = degree
        self.basis = basis
        self.intercept = intercept
        
        # Constant basis always has degree 1
        if basis == 'constant':
            self.degree = 1
        
        # Set name
        if name is None:
            name = f"baseline_{basis}_{degree}"
        self.name = name
        
        # Set basis function
        self.fun = self._get_basis_function()
    
    def _get_basis_function(self):
        """Get the appropriate basis function."""
        if self.basis == 'constant':
            return lambda x, degree=None: np.ones((len(x), 1))
        elif self.basis == 'poly':
            return self._poly_basis
        elif self.basis == 'bs':
            return self._bs_basis
        elif self.basis == 'ns':
            return self._ns_basis
        else:
            raise ValueError(f"Unknown basis type: {self.basis}")
    
    def _poly_basis(self, x, degree):
        """Polynomial basis function."""
        x = np.asarray(x, dtype=float)
        # Create orthogonal polynomial basis.
        # This mirrors the behavior of `stats::poly` by centering each raw
        # power before orthogonalization.
        n = len(x)
        if n == 0:
            return np.empty((n, degree), dtype=float)
        X = np.zeros((n, degree))

        x = x.ravel()
        # Generate polynomial terms
        for i in range(degree):
            power = x ** (i + 1)
            X[:, i] = power - np.mean(power)
        
        # Orthogonalize using QR decomposition
        Q, R = np.linalg.qr(X)
        # Preserve a deterministic sign convention (match reference behavior
        # and keep column orientation stable across runs).
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs
        return Q
    
    def _bs_basis(self, x, degree):
        """B-spline basis function."""
        x = np.asarray(x)
        # Align with R's `splines::bs` usage in fmridesign's baseline
        # specification, where `degree` is the only explicit argument.
        #
        # Patsy requires either `df` or `knots` when using `bs()`, while
        # R's `bs(x, degree = d)` with no explicit knots produces one basis
        # function per degree without interior knots. Passing an empty knots list
        # reproduces that behavior while preserving exact parity.
        return np.asarray(
            dmatrix(
                "0 + bs(x, degree=%d, knots=[], include_intercept=False)"
                % int(degree),
                {"x": x},
            )
        )

    def _ns_basis(self, x, df):
        """Natural spline basis function."""
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.ravel()

        x = x.astype(float)
        n = len(x)

        if n == 0:
            return np.empty((0, df), dtype=float)

        degree = 3

        # Mirroring `ns(x, df=..., intercept=False)` knot logic.
        x_range = [np.min(x), np.max(x)]
        x_no_na = x[~np.isnan(x)]
        if x_no_na.size == 0:
            return np.full((n, df), np.nan, dtype=float)

        n_interior = df - 1
        if n_interior > 0:
            probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            knots = np.quantile(x_no_na, probs)
        else:
            knots = np.array([], dtype=float)

        # Create cubic B-spline design on extended knots (mirrors R's Aknots).
        a_knots = np.concatenate(
            [
                np.repeat(x_range[0], degree + 1),
                knots,
                np.repeat(x_range[1], degree + 1),
            ]
        )
        n_basis = len(a_knots) - (degree + 1)
        if n_basis <= 2:
            return np.zeros((n, max(df, 0)), dtype=float)

        basis = np.zeros((n, n_basis), dtype=float)
        basis_deriv2 = np.zeros((2, n_basis), dtype=float)
        boundaries = np.array(x_range, dtype=float)

        for i in range(n_basis):
            coef = np.zeros(n_basis, dtype=float)
            coef[i] = 1.0
            spline = ScipyBSpline(a_knots, coef, degree)
            basis[:, i] = spline(x)

            deriv2 = spline.derivative(2)
            basis_deriv2[:, i] = deriv2(boundaries)

        # Remove intercept column for `intercept=False`, as R does.
        basis = basis[:, 1:]
        basis_deriv2 = basis_deriv2[:, 1:]

        # Enforce natural boundary constraints through the same R projection:
        # QR-based null-space transformation and dropping first two constrained
        # coordinates.
        q_const, _ = np.linalg.qr(basis_deriv2.T, mode="complete")
        projected = basis @ q_const[:, 2:]

        if projected.shape[1] != df:
            if projected.shape[1] < df:
                projected = np.pad(
                    projected, ((0, 0), (0, df - projected.shape[1]))
                )
            else:
                projected = projected[:, :df]

        return projected


def baseline_model(
    basis: Literal['constant', 'poly', 'bs', 'ns'] = 'constant',
    degree: int = 1,
    sframe: SamplingInfo = None,
    intercept: Literal['runwise', 'global', 'none'] = 'runwise',
    nuisance_list: Optional[List[Array]] = None
) -> BaselineModel:
    """Construct a baseline model for fMRI time series.
    
    Creates a model for low-frequency drift, block-wise intercepts, and
    nuisance regressors that account for non-neural signal variations.
    
    Parameters
    ----------
    basis : {'constant', 'poly', 'bs', 'ns'}, default='constant'
        Type of basis for drift modeling:
        - 'constant': Constant (intercept only)
        - 'poly': Polynomial basis
        - 'bs': B-spline basis
        - 'ns': Natural spline basis
    degree : int, default=1
        Number of basis functions per block (ignored for 'constant')
    sframe : SamplingInfo
        Sampling frame with block structure information
    intercept : {'runwise', 'global', 'none'}, default='runwise'
        Type of intercept to include:
        - 'runwise': Separate intercept per run/block
        - 'global': Single intercept for all runs
        - 'none': No intercept
    nuisance_list : list of arrays, optional
        List of nuisance regressor matrices, one per block
    
    Returns
    -------
    BaselineModel
        Constructed baseline model
    
    Examples
    --------
    >>> # Polynomial drift with runwise intercepts
    >>> bmodel = baseline_model(
    ...     basis='poly',
    ...     degree=3,
    ...     sframe=sampling_frame
    ... )
    >>> 
    >>> # B-spline drift with global intercept
    >>> bmodel = baseline_model(
    ...     basis='bs',
    ...     degree=4,
    ...     sframe=sampling_frame,
    ...     intercept='global'
    ... )
    >>> 
    >>> # Add nuisance regressors
    >>> nuisance = [np.random.randn(100, 6), np.random.randn(100, 6)]
    >>> bmodel = baseline_model(
    ...     basis='poly',
    ...     degree=2,
    ...     sframe=sampling_frame,
    ...     nuisance_list=nuisance
    ... )
    """
    if sframe is None:
        raise ValueError("sampling_frame (sframe) must be provided")
    
    # Validate basis
    if basis not in ['constant', 'poly', 'bs', 'ns']:
        raise ValueError(f"Invalid basis: {basis}")
    
    # Validate degree for splines
    if basis in ['bs', 'ns'] and degree < 3:
        raise ValueError(f"'{basis}' basis must have degree >= 3")
    
    # Create drift specification
    drift_spec = baseline(degree=degree, basis=basis, intercept=intercept)
    
    # Construct terms
    terms = {}
    
    # Always create drift term
    terms['drift'] = _construct_drift_term(drift_spec, sframe)
    
    # Create block term if needed
    if intercept != 'none' and basis != 'constant':
        terms['block'] = _construct_block_term('constant', sframe, intercept)
    else:
        terms['block'] = None
    
    # Create nuisance term if provided
    if nuisance_list is not None:
        terms['nuisance'] = _make_nuisance_term(nuisance_list, sframe)
    else:
        terms['nuisance'] = None
    
    return BaselineModel(terms, drift_spec, sframe)


def baseline(
    degree: int = 1,
    basis: Literal['constant', 'poly', 'bs', 'ns'] = 'constant',
    name: Optional[str] = None,
    intercept: Literal['runwise', 'global', 'none'] = 'runwise'
) -> BaselineSpec:
    """Create a baseline specification.
    
    Generates a specification for modeling low-frequency drift in fMRI time series.
    
    Parameters
    ----------
    degree : int, default=1
        Number of basis terms per block (ignored for 'constant')
    basis : {'constant', 'poly', 'bs', 'ns'}, default='constant'
        Type of basis function
    name : str, optional
        Name for the term
    intercept : {'runwise', 'global', 'none'}, default='runwise'
        Type of intercept
    
    Returns
    -------
    BaselineSpec
        Baseline specification
    """
    return BaselineSpec(degree=degree, basis=basis, intercept=intercept, name=name)


def _construct_drift_term(spec: BaselineSpec, sframe: SamplingInfo) -> BaselineTerm:
    """Construct drift term from specification.

    Generates a block-diagonal basis matrix where each block's
    columns are produced by the drift basis function evaluated
    on ``1, 2, ..., block_len``.

    Parameters
    ----------
    spec : BaselineSpec
        Drift specification with basis type and degree.
    sframe : SamplingInfo
        Sampling frame providing block lengths.

    Returns
    -------
    BaselineTerm
        Baseline term with block-diagonal drift regressors.
    """
    # Get block lengths
    if hasattr(sframe, 'blocklens'):
        block_lens = sframe.blocklens
    elif hasattr(sframe, 'n_scans'):
        # Single block
        block_lens = [sframe.n_scans]
    else:
        raise ValueError("Cannot determine block lengths from sampling frame")
    
    # Generate basis for each block
    matrices = []
    col_indices = []
    row_indices = []
    
    current_col = 0
    current_row = 0
    
    for i, block_len in enumerate(block_lens):
        # Generate time points for this block
        t = np.arange(1, block_len + 1)
        
        # Apply basis function
        if spec.basis == 'ns':
            # Natural splines use 'df' parameter
            block_matrix = spec.fun(t, df=spec.degree)
        elif spec.basis in ['poly', 'bs']:
            block_matrix = spec.fun(t, degree=spec.degree)
        else:
            # Constant
            block_matrix = spec.fun(t)
        
        # Ensure 2D
        if block_matrix.ndim == 1:
            block_matrix = block_matrix.reshape(-1, 1)
        
        matrices.append(block_matrix)
        
        # Track indices
        n_cols = block_matrix.shape[1]
        col_indices.append(list(range(current_col, current_col + n_cols)))
        row_indices.append(list(range(current_row, current_row + block_len)))
        
        current_col += n_cols
        current_row += block_len
    
    # Handle global constant intercept
    if spec.basis == 'constant' and spec.intercept == 'global':
        # Combine into single column
        full_matrix = np.vstack(matrices)
        col_names = [f"base_{spec.basis}"]
        col_indices = [[0] for _ in block_lens]
    else:
        # Block diagonal structure
        full_matrix = _block_diagonal(matrices)
        
        # Generate column names
        col_names = []
        for i, mat in enumerate(matrices):
            for j in range(mat.shape[1]):
                col_names.append(f"base_{spec.basis}{j+1}_block_{i+1}")
    
    # Create DataFrame
    df = pd.DataFrame(full_matrix, columns=col_names)
    
    return BaselineTerm(
        varname=spec.name,
        design_matrix=df,
        colind=col_indices,
        rowind=row_indices
    )


def _construct_block_term(
    basis: str,
    sframe: SamplingInfo,
    intercept: str
) -> BaselineTerm:
    """Construct block intercept term.

    Creates either a single global intercept column or separate
    run-wise intercept columns depending on the ``intercept`` mode.

    Parameters
    ----------
    basis : str
        Basis type (currently unused, kept for API consistency).
    sframe : SamplingInfo
        Sampling frame providing block lengths.
    intercept : {'runwise', 'global'}
        Intercept strategy.

    Returns
    -------
    BaselineTerm
        Baseline term with intercept columns.
    """
    # Get block information
    if hasattr(sframe, 'blocklens'):
        block_lens = sframe.blocklens
    else:
        block_lens = [sframe.n_scans]
    
    n_blocks = len(block_lens)
    total_rows = sum(block_lens)
    
    if intercept == 'global':
        # Single intercept column
        matrix = np.ones((total_rows, 1))
        col_names = ['constant_global']
        col_indices = [[0] for _ in range(n_blocks)]
    else:
        if n_blocks == 1:
            matrix = np.ones((total_rows, 1))
            col_names = ['constant_global']
            col_indices = [[0]]
        else:
            # Runwise intercepts
            matrices = []
            for block_len in block_lens:
                matrices.append(np.ones((block_len, 1)))
            
            matrix = _block_diagonal(matrices)
            col_names = [f'constant_{i+1}' for i in range(n_blocks)]
            col_indices = [[i] for i in range(n_blocks)]
    
    # Create row indices
    row_indices = []
    current_row = 0
    for block_len in block_lens:
        row_indices.append(list(range(current_row, current_row + block_len)))
        current_row += block_len
    
    df = pd.DataFrame(matrix, columns=col_names)
    
    return BaselineTerm(
        varname='block',
        design_matrix=df,
        colind=col_indices,
        rowind=row_indices
    )


def _make_nuisance_term(
    nuisance_list: List[Array],
    sframe: SamplingInfo
) -> BaselineTerm:
    """Create nuisance term from a list of per-block matrices.

    Arranges the per-block nuisance matrices into a block-diagonal
    structure in the full time-series space.

    Parameters
    ----------
    nuisance_list : list of array-like
        One matrix per block. Each matrix has shape
        ``(block_len, n_nuisance_regressors)``.
    sframe : SamplingInfo
        Sampling frame providing block lengths.

    Returns
    -------
    BaselineTerm
        Baseline term with block-diagonal nuisance regressors.

    Raises
    ------
    ValueError
        If the number of nuisance matrices does not match the
        number of blocks, or if row counts are inconsistent.
    """
    # Get block information
    if hasattr(sframe, 'blocklens'):
        block_lens = sframe.blocklens
    else:
        block_lens = [sframe.n_scans]
    
    # Validate input
    if len(nuisance_list) != len(block_lens):
        raise ValueError(
            f"Length of nuisance_list ({len(nuisance_list)}) must equal "
            f"number of blocks ({len(block_lens)})"
        )
    
    # Convert to matrices and validate sizes
    matrices = []
    total_cols = 0
    
    for i, (nuisance, block_len) in enumerate(zip(nuisance_list, block_lens)):
        mat = np.asarray(nuisance)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        
        if mat.shape[0] != block_len:
            raise ValueError(
                f"Nuisance matrix {i} has {mat.shape[0]} rows but "
                f"block has {block_len} timepoints"
            )
        
        matrices.append(mat)
        total_cols += mat.shape[1]
    
    # Create block diagonal matrix
    full_matrix = _block_diagonal(matrices)
    
    # Generate column names and indices
    col_names = []
    col_indices = []
    current_col = 0
    
    for i, mat in enumerate(matrices):
        n_cols = mat.shape[1]
        block_cols = []
        
        for j in range(n_cols):
            col_names.append(f'nuis_run{i+1}_c{j+1}')
            block_cols.append(current_col)
            current_col += 1
        
        col_indices.append(block_cols)
    
    # Generate row indices
    row_indices = []
    current_row = 0
    
    for block_len in block_lens:
        row_indices.append(list(range(current_row, current_row + block_len)))
        current_row += block_len
    
    df = pd.DataFrame(full_matrix, columns=col_names)
    
    return BaselineTerm(
        varname='nuisance',
        design_matrix=df,
        colind=col_indices,
        rowind=row_indices
    )


def _block_diagonal(matrices: List[Array]) -> Array:
    """Create a block-diagonal matrix from a list of matrices.

    Parameters
    ----------
    matrices : list of array-like
        Sub-matrices to place on the diagonal.

    Returns
    -------
    Array
        Block-diagonal matrix with shape
        ``(sum(nrows), sum(ncols))``.
    """
    if not matrices:
        return np.array([])

    # Get dimensions
    n_rows = sum(mat.shape[0] for mat in matrices)
    n_cols = sum(mat.shape[1] for mat in matrices)

    # Create output matrix
    result = np.zeros((n_rows, n_cols))

    # Fill blocks
    row_offset = 0
    col_offset = 0

    for mat in matrices:
        n_r, n_c = mat.shape
        result[row_offset:row_offset+n_r, col_offset:col_offset+n_c] = mat
        row_offset += n_r
        col_offset += n_c

    return result


def dctbasis(n: int, p: Optional[int] = None, const: bool = False) -> np.ndarray:
    """Discrete Cosine Transform basis matrix.

    Creates a DCT basis matrix for modeling low-frequency drift or other
    smooth trends in fMRI time series. The DCT basis provides an orthogonal
    decomposition similar to Fourier basis but using only cosine functions.

    Parameters
    ----------
    n : int
        Number of time points (rows in output matrix)
    p : int, optional
        Number of basis functions (columns in output matrix).
        If None, defaults to n.
    const : bool, default=False
        If True, prepend a constant (DC) column normalized by (1/n)^0.5.

    Returns
    -------
    np.ndarray
        DCT basis matrix of shape (n, p) if const=False, or (n, p+1) if const=True.
        Each column represents one DCT basis function, orthonormalized.

    Notes
    -----
    The DCT basis is defined as:

    .. math::
        X_{m,k} = \\sqrt{\\frac{2}{n}} \\cos\\left(\\frac{(2m-1)k\\pi}{2n}\\right)

    where m = 1, ..., n are the time points and k = 1, ..., p are the basis indices.

    The constant term (when const=True) is:

    .. math::
        X_{m,0} = \\sqrt{\\frac{1}{n}}

    This is faithful to the R fmridesign implementation in basis.R.

    Examples
    --------
    >>> # Create a 100-point DCT basis with 5 functions
    >>> basis = dctbasis(100, p=5)
    >>> basis.shape
    (100, 5)

    >>> # Include constant term
    >>> basis_const = dctbasis(100, p=5, const=True)
    >>> basis_const.shape
    (100, 6)

    >>> # Verify orthonormality
    >>> import numpy as np
    >>> gram = basis.T @ basis
    >>> np.allclose(gram, np.eye(5))
    True
    """
    if p is None:
        p = n

    m = np.arange(1, n + 1)
    X = np.zeros((n, p))

    for k in range(1, p + 1):
        X[:, k - 1] = (2 / n) ** 0.5 * np.cos(((2 * m - 1) * k * np.pi) / (2 * n))

    if const:
        const_col = np.full((n, 1), (1 / n) ** 0.5)
        X = np.hstack([const_col, X])

    return X
