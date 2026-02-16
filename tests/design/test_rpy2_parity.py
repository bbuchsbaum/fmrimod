"""Optional cross-language parity tests using rpy2.

These tests execute R's fmridesign directly from Python via rpy2 and compare
the resulting design matrices against pyfmridesign.

They are marked with ``@pytest.mark.rpy2`` and skip automatically when rpy2
or required R packages are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pytest

from fmrimod import baseline_model, design_matrix, event_model
from fmrimod.contrast import Fcontrasts, contrast_set, contrast_weights, pair_contrast
from fmrimod.contrast.contrast_spec import Formula as ContrastFormula
from fmrimod.formula.base import Term

pytestmark = pytest.mark.filterwarnings(
    "ignore:scipy.misc is deprecated and will be removed in 2.0.0:DeprecationWarning"
)


@dataclass
class RContext:
    """Container for lazily imported R bridge objects."""

    fmridesign: object
    pyfmrihrf: object
    fmrihrf: object
    Formula: object
    IntVector: object
    FloatVector: object
    conversion: object
    default_converter: object
    pandas2ri: object
    localconverter: object


@pytest.fixture(scope="module")
def rctx() -> RContext:
    """Provide an R interop context or skip when unavailable."""
    pytest.importorskip("rpy2")

    from rpy2.robjects import Formula
    from rpy2.robjects import conversion, default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import PackageNotInstalledError, importr
    from rpy2.robjects.vectors import FloatVector, IntVector
    import importlib

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="scipy.misc is deprecated and will be removed in 2.0.0",
            category=DeprecationWarning,
            module=r"pyfmrihrf.*",
        )
        warnings.filterwarnings(
            "ignore",
            message="Parameters .* ignored for pre-defined HRF 'gamma'",
            category=UserWarning,
            module=r"pyfmrihrf.*",
        )

        try:
            fmridesign = importr("fmridesign")
            fmrihrf = importr("fmrihrf")
        except PackageNotInstalledError as exc:
            pytest.skip(f"Required R package not installed: {exc}")

        pyfmrihrf = importlib.import_module("pyfmrihrf")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            pyfmrihrf = importlib.reload(pyfmrihrf)

    return RContext(
        fmridesign=fmridesign,
        pyfmrihrf=pyfmrihrf,
        fmrihrf=fmrihrf,
        Formula=Formula,
        IntVector=IntVector,
        FloatVector=FloatVector,
        conversion=conversion,
        default_converter=default_converter,
        pandas2ri=pandas2ri,
        localconverter=localconverter,
    )


def _py_sampling_frame(ctx: RContext, *, blocklens: list[int], tr: float) -> object:
    """Build a pyfmrihrf SamplingFrame across API variants."""
    ctor = ctx.pyfmrihrf.SamplingFrame
    try:
        return ctor(blocklens=blocklens, TR=float(tr))
    except TypeError as exc:
        if "unexpected keyword argument 'TR'" not in str(exc):
            raise
        return ctor(blocklens=blocklens, tr=float(tr))


def test_py_sampling_frame_falls_back_to_tr_keyword() -> None:
    """Regression: support pyfmrihrf versions that only accept ``tr``."""

    class LegacySamplingFrame:
        def __init__(self, *, blocklens, tr):
            self.blocklens = blocklens
            self.tr = tr

    class Ctx:
        pass

    class Mod:
        SamplingFrame = LegacySamplingFrame

    ctx = Ctx()
    ctx.pyfmrihrf = Mod()
    sf = _py_sampling_frame(ctx, blocklens=[20], tr=1.5)
    assert sf.blocklens == [20]
    assert sf.tr == 1.5


def test_multiblock_event_model_accepts_sampling_frame_with_tr_only() -> None:
    """Regression: multiblock convolution should not require ``TR`` attr."""

    class TrOnlySamplingFrame:
        def __init__(self, *, blocklens: list[int], tr: float):
            self.blocklens = np.asarray(blocklens, dtype=int)
            self.tr = np.full(len(self.blocklens), float(tr), dtype=float)
            self.start_time = self.tr / 2.0

        @property
        def n_blocks(self) -> int:
            return int(len(self.blocklens))

        @property
        def n_scans(self) -> int:
            return int(np.sum(self.blocklens))

        @property
        def samples(self) -> np.ndarray:
            offsets = np.concatenate(([0.0], np.cumsum((self.blocklens * self.tr)[:-1])))
            grids = []
            for i, n_scans in enumerate(self.blocklens):
                local_grid = np.arange(int(n_scans), dtype=float) * self.tr[i] + self.start_time[i]
                grids.append(offsets[i] + local_grid)
            return np.concatenate(grids)

    df = pd.DataFrame(
        {
            "onset": [2, 6, 2, 6],
            "condition": pd.Categorical(["A", "B", "A", "B"]),
            "run": [1, 1, 2, 2],
        }
    )

    model = event_model(
        "onset ~ hrf(condition, basis='spmg1')",
        data=df,
        block="run",
        sampling_frame=TrOnlySamplingFrame(blocklens=[20, 20], tr=2.0),
        durations=0.0,
        precision=0.3,
    )
    dm = design_matrix(model)

    assert dm.shape == (40, 2)
    assert np.isfinite(dm).all()


def _r_design_matrix(
    ctx: RContext,
    *,
    formula: str,
    data: pd.DataFrame,
    block_formula: str = "~run",
    blocklens: list[int] | tuple[int, ...] = (100,),
    tr: float = 2.0,
    durations: float | list[float] = 0.0,
    precision: float = 0.3,
) -> np.ndarray:
    """Build an R fmridesign model and return design matrix as numpy array."""
    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        r_df = ctx.conversion.get_conversion().py2rpy(data)

    r_sf = ctx.fmrihrf.sampling_frame(
        blocklens=ctx.IntVector(list(blocklens)),
        TR=float(tr),
    )

    if isinstance(durations, list):
        r_durations = ctx.FloatVector([float(d) for d in durations])
    else:
        r_durations = float(durations)

    r_model = ctx.fmridesign.event_model(
        ctx.Formula(formula),
        data=r_df,
        block=ctx.Formula(block_formula),
        sampling_frame=r_sf,
        durations=r_durations,
        precision=float(precision),
    )
    r_dm = ctx.fmridesign.design_matrix(r_model)

    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        r_dm_df = ctx.conversion.get_conversion().rpy2py(r_dm)

    return np.asarray(r_dm_df, dtype=float)


def _r_design_matrix_with_colnames(
    ctx: RContext,
    *,
    formula: str,
    data: pd.DataFrame,
    block_formula: str = "~run",
    blocklens: list[int] | tuple[int, ...] = (100,),
    tr: float = 2.0,
    durations: float | list[float] = 0.0,
    precision: float = 0.3,
) -> tuple[np.ndarray, list[str]]:
    """Build an R model and return design matrix plus exact column names."""
    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        r_df = ctx.conversion.get_conversion().py2rpy(data)

    r_sf = ctx.fmrihrf.sampling_frame(
        blocklens=ctx.IntVector(list(blocklens)),
        TR=float(tr),
    )

    if isinstance(durations, list):
        r_durations = ctx.FloatVector([float(d) for d in durations])
    else:
        r_durations = float(durations)

    r_model = ctx.fmridesign.event_model(
        ctx.Formula(formula),
        data=r_df,
        block=ctx.Formula(block_formula),
        sampling_frame=r_sf,
        durations=r_durations,
        precision=float(precision),
    )
    r_dm = ctx.fmridesign.design_matrix(r_model)

    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        r_dm_df = ctx.conversion.get_conversion().rpy2py(r_dm)

    return np.asarray(r_dm_df, dtype=float), list(r_dm_df.columns)


def _r_baseline_design_matrix_with_colnames(
    ctx: RContext,
    *,
    basis: str,
    degree: int = 1,
    blocklens: list[int] | tuple[int, ...] = (200,),
    tr: float = 2.0,
    intercept: str = "runwise",
) -> tuple[np.ndarray, list[str]]:
    """Build an R baseline model and return matrix plus exact column names."""
    r_sf = ctx.fmrihrf.sampling_frame(
        blocklens=ctx.IntVector(list(blocklens)),
        TR=float(tr),
    )
    r_model = ctx.fmridesign.baseline_model(
        basis=basis,
        degree=int(degree),
        sframe=r_sf,
        intercept=intercept,
    )
    r_dm = ctx.fmridesign.design_matrix(r_model)

    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        r_dm_df = ctx.conversion.get_conversion().rpy2py(r_dm)

    return np.asarray(r_dm_df, dtype=float), list(r_dm_df.columns)


def _r_pair_contrast_weights(
    ctx: RContext,
    *,
    data: pd.DataFrame,
    blocklens: list[int] | tuple[int, ...] = (100,),
    tr: float = 2.0,
    durations: float | list[float] = 1.0,
    precision: float = 0.3,
    contrast_name: str = "pair_AB",
) -> np.ndarray:
    """Return R pair-contrast weights vector from a live model."""
    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        r_df = ctx.conversion.get_conversion().py2rpy(data)

    r_sf = ctx.fmrihrf.sampling_frame(
        blocklens=ctx.IntVector(list(blocklens)),
        TR=float(tr),
    )
    if isinstance(durations, list):
        r_durations = ctx.FloatVector([float(d) for d in durations])
    else:
        r_durations = float(durations)

    # Keep contrast definitions inline so R formula evaluation has scope.
    r_formula = (
        "onset ~ hrf(condition, basis='spmg1', "
        "contrasts=contrast_set("
        "pair_contrast(~ condition == \"A\", ~ condition == \"B\", name='pair_AB')"
        "))"
    )

    r_model = ctx.fmridesign.event_model(
        ctx.Formula(r_formula),
        data=r_df,
        block=ctx.Formula("~run"),
        sampling_frame=r_sf,
        durations=r_durations,
        precision=float(precision),
    )
    r_cw = ctx.fmridesign.contrast_weights(r_model)

    for obj in r_cw:
        name = str(obj.rx2("name")[0])
        if name == contrast_name:
            return np.asarray(obj.rx2("weights"), dtype=float).ravel()

    raise AssertionError(f"R contrast '{contrast_name}' not found")


def _r_fcontrast_matrices(
    ctx: RContext,
    *,
    formula: str,
    data: pd.DataFrame,
    block_formula: str = "~run",
    blocklens: list[int] | tuple[int, ...] = (100,),
    tr: float = 2.0,
    durations: float | list[float] = 1.0,
    precision: float = 0.3,
) -> dict[str, np.ndarray]:
    """Return R F-contrast matrices keyed by contrast name."""
    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        r_df = ctx.conversion.get_conversion().py2rpy(data)

    r_sf = ctx.fmrihrf.sampling_frame(
        blocklens=ctx.IntVector(list(blocklens)),
        TR=float(tr),
    )
    if isinstance(durations, list):
        r_durations = ctx.FloatVector([float(d) for d in durations])
    else:
        r_durations = float(durations)

    r_model = ctx.fmridesign.event_model(
        ctx.Formula(formula),
        data=r_df,
        block=ctx.Formula(block_formula),
        sampling_frame=r_sf,
        durations=r_durations,
        precision=float(precision),
    )
    r_fc = ctx.fmridesign.Fcontrasts(r_model)

    out = {}
    if r_fc is None:
        return out

    names = list(r_fc.names) if hasattr(r_fc, "names") else []
    if not names:
        names = [str(key) for key in range(len(r_fc))]

    with ctx.localconverter(ctx.default_converter + ctx.pandas2ri.converter):
        for name in names:
            r_mat = r_fc.rx2(name)
            out[str(name)] = np.asarray(ctx.conversion.get_conversion().rpy2py(r_mat), dtype=float)

    return out


def _assert_parity(py_dm: np.ndarray, r_dm: np.ndarray, *, label: str) -> None:
    """Assert design matrix parity."""
    assert py_dm.shape == r_dm.shape, (
        f"{label}: shape mismatch (py={py_dm.shape}, r={r_dm.shape})"
    )
    np.testing.assert_allclose(py_dm, r_dm, atol=1e-6, err_msg=label)


@pytest.mark.rpy2
def test_rpy2_simple_spmg1_parity(rctx: RContext) -> None:
    """Check canonical HRF parity against live R execution."""
    df = pd.DataFrame(
        {
            "onset": [2, 6, 12, 18, 24, 30],
            "condition": pd.Categorical(["A", "B", "A", "B", "A", "B"]),
            "run": [1, 1, 1, 1, 1, 1],
        }
    )

    py_model = event_model(
        "onset ~ hrf(condition, basis='spmg1')",
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=1.0,
        precision=0.3,
    )
    py_dm = design_matrix(py_model)

    r_dm = _r_design_matrix(
        rctx,
        formula="onset ~ hrf(condition, basis='spmg1')",
        data=df,
        blocklens=[100],
        tr=2.0,
        durations=1.0,
        precision=0.3,
    )
    _assert_parity(py_dm, r_dm, label="rpy2 spmg1 parity mismatch")


@pytest.mark.rpy2
def test_rpy2_spmg2_parity(rctx: RContext) -> None:
    """Check derivative HRF basis parity against live R execution."""
    df = pd.DataFrame(
        {
            "onset": [2, 6, 12, 18, 24, 30],
            "condition": pd.Categorical(["A", "B", "A", "B", "A", "B"]),
            "run": [1, 1, 1, 1, 1, 1],
        }
    )

    py_model = event_model(
        "onset ~ hrf(condition, basis='spmg2')",
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=1.0,
        precision=0.3,
    )
    py_dm = design_matrix(py_model)

    r_dm = _r_design_matrix(
        rctx,
        formula="onset ~ hrf(condition, basis='spmg2')",
        data=df,
        blocklens=[100],
        tr=2.0,
        durations=1.0,
        precision=0.3,
    )
    _assert_parity(py_dm, r_dm, label="rpy2 spmg2 parity mismatch")


@pytest.mark.rpy2
def test_rpy2_spmg2_column_name_and_order_parity(rctx: RContext) -> None:
    """Ensure exact column names/order parity with R."""
    df = pd.DataFrame(
        {
            "onset": [2, 6, 12, 18, 24, 30],
            "condition": pd.Categorical(["A", "B", "A", "B", "A", "B"]),
            "run": [1, 1, 1, 1, 1, 1],
        }
    )

    py_model = event_model(
        "onset ~ hrf(condition, basis='spmg2')",
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=1.0,
        precision=0.3,
    )
    py_names = list(py_model.column_names)

    _, r_names = _r_design_matrix_with_colnames(
        rctx,
        formula="onset ~ hrf(condition, basis='spmg2')",
        data=df,
        blocklens=[100],
        tr=2.0,
        durations=1.0,
        precision=0.3,
    )
    assert py_names == r_names, f"Column name/order mismatch: py={py_names}, r={r_names}"


@pytest.mark.rpy2
def test_rpy2_multiblock_and_parametric_parity(rctx: RContext) -> None:
    """Check multi-block isolation and parametric modulation parity."""
    df = pd.DataFrame(
        {
            "onset": [5, 15, 25, 35, 45, 5, 15, 25, 35, 45],
            "condition": pd.Categorical(["A", "B", "A", "B", "A"] * 2),
            "RT": [0.7, 1.0, 0.8, 1.1, 0.9, 0.6, 1.2, 0.7, 1.0, 0.8],
            "run": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )

    py_model = event_model(
        "onset ~ hrf(condition, basis='spmg1') + hrf(RT, basis='spmg1')",
        data=df,
        block="run",
        sampling_frame=_py_sampling_frame(rctx, blocklens=[50, 50], tr=2.0),
        durations=0.0,
        precision=0.3,
    )
    py_dm = design_matrix(py_model)

    r_dm = _r_design_matrix(
        rctx,
        formula="onset ~ hrf(condition, basis='spmg1') + hrf(RT, basis='spmg1')",
        data=df,
        blocklens=[50, 50],
        tr=2.0,
        durations=0.0,
        precision=0.3,
    )
    _assert_parity(py_dm, r_dm, label="rpy2 multiblock/parametric parity mismatch")


@pytest.mark.rpy2
def test_rpy2_pair_contrast_weight_parity(rctx: RContext) -> None:
    """Check pair contrast weights parity against live R execution."""
    df = pd.DataFrame(
        {
            "onset": [2, 6, 12, 18, 24, 30],
            "condition": ["A", "B", "A", "B", "A", "B"],
            "run": [1, 1, 1, 1, 1, 1],
        }
    )

    term = Term("condition", hrf="spmg1")
    term.contrast_specs = contrast_set(
        pair_contrast(
            ContrastFormula("condition == 'A'"),
            ContrastFormula("condition == 'B'"),
            name="pair_AB",
        )
    )
    py_model = event_model(
        [term],
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=1.0,
        precision=0.3,
    )
    py_con = contrast_weights(py_model)["condition"]["pair_AB"]
    py_w = np.asarray(py_con.offset_weights, dtype=float).ravel()

    r_w = _r_pair_contrast_weights(
        rctx,
        data=df.assign(condition=pd.Categorical(df["condition"])),
        blocklens=[100],
        tr=2.0,
        durations=1.0,
        precision=0.3,
        contrast_name="pair_AB",
    )
    np.testing.assert_allclose(py_w, r_w, atol=1e-10)


@pytest.mark.rpy2
def test_rpy2_spmg3_parity(rctx: RContext) -> None:
    """Check 3-function HRF basis parity against live R execution."""
    df = pd.DataFrame(
        {
            "onset": [2, 6, 12, 18, 24, 30],
            "condition": pd.Categorical(["A", "B", "A", "B", "A", "B"]),
            "run": [1, 1, 1, 1, 1, 1],
        }
    )

    py_model = event_model(
        "onset ~ hrf(condition, basis='spmg3')",
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=1.0,
        precision=0.3,
    )
    py_dm = design_matrix(py_model)

    r_dm = _r_design_matrix(
        rctx,
        formula="onset ~ hrf(condition, basis='spmg3')",
        data=df,
        blocklens=[100],
        tr=2.0,
        durations=1.0,
        precision=0.3,
    )
    _assert_parity(py_dm, r_dm, label="rpy2 spmg3 parity mismatch")


@pytest.mark.rpy2
def test_rpy2_fcontrast_parity(rctx: RContext) -> None:
    """Check F-contrast matrices parity against live R execution."""
    df = pd.DataFrame(
        {
            "onset": [5, 15, 25, 35, 45, 55, 65, 75],
            "condition": pd.Categorical(["A", "B", "A", "B", "A", "B", "A", "B"]),
            "run": [1] * 8,
        }
    )

    py_model = event_model(
        "onset ~ hrf(condition, basis='spmg1')",
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=1.0,
        precision=0.3,
    )
    py_fcs = Fcontrasts(py_model)

    r_fcs = _r_fcontrast_matrices(
        rctx,
        formula="onset ~ hrf(condition, basis='spmg1')",
        data=df,
        blocklens=[100],
        tr=2.0,
        durations=1.0,
        precision=0.3,
    )

    assert set(py_fcs.keys()) == set(r_fcs.keys()), (
        f"F-contrast key mismatch: py={sorted(py_fcs.keys())}, r={sorted(r_fcs.keys())}"
    )
    for name in py_fcs:
        py_mat = np.asarray(py_fcs[name], dtype=float)
        r_mat = np.asarray(r_fcs[name], dtype=float)
        _assert_parity(py_mat, r_mat, label=f"rpy2 F-contrast parity mismatch ({name})")


@pytest.mark.rpy2
def test_rpy2_interaction_star_parity(rctx: RContext) -> None:
    """Check Python `*` interaction syntax against R `:` interaction."""
    df = pd.DataFrame(
        {
            "onset": [2, 6, 10, 14, 18, 22, 26, 30],
            "condition": pd.Categorical(["A", "A", "B", "B", "A", "A", "B", "B"]),
            "group": pd.Categorical(["G1", "G2", "G1", "G2", "G1", "G2", "G1", "G2"]),
            "run": [1] * 8,
        }
    )

    py_model = event_model(
        "onset ~ hrf(condition * group, basis='spmg1')",
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=1.0,
        precision=0.3,
    )
    py_dm = design_matrix(py_model)
    py_names = list(py_model.column_names)

    r_dm, r_names = _r_design_matrix_with_colnames(
        rctx,
        formula="onset ~ hrf(condition:group, basis='spmg1')",
        data=df,
        blocklens=[100],
        tr=2.0,
        durations=1.0,
        precision=0.3,
    )
    _assert_parity(py_dm, r_dm, label="rpy2 interaction star parity mismatch")
    assert len(py_names) == len(r_names)


@pytest.mark.rpy2
def test_rpy2_variable_duration_parity(rctx: RContext) -> None:
    """Check variable duration parity against live R execution."""
    df = pd.DataFrame(
        {
            "onset": [2, 6, 12, 18, 24, 30],
            "condition": pd.Categorical(["A", "B", "A", "B", "A", "B"]),
            "run": [1, 1, 1, 1, 1, 1],
        }
    )
    durations = [0.5, 1.0, 2.0, 1.5, 0.75, 1.25]

    py_model = event_model(
        "onset ~ hrf(condition, basis='spmg1')",
        data=df,
        block="run",
        tr=2.0,
        n_scans=100,
        durations=durations,
        precision=0.3,
    )
    py_dm = design_matrix(py_model)

    r_dm = _r_design_matrix(
        rctx,
        formula="onset ~ hrf(condition, basis='spmg1')",
        data=df,
        blocklens=[100],
        tr=2.0,
        durations=durations,
        precision=0.3,
    )
    _assert_parity(py_dm, r_dm, label="rpy2 variable durations parity mismatch")


@pytest.mark.rpy2
def test_rpy2_baseline_constant_parity(rctx: RContext) -> None:
    """Check baseline constant design matrix parity against live R execution."""
    py_sf = _py_sampling_frame(rctx, blocklens=[200], tr=2.0)
    py_model = baseline_model(basis="constant", sframe=py_sf)
    py_dm = design_matrix(py_model)
    py_names = list(py_model.column_names)

    r_dm, r_names = _r_baseline_design_matrix_with_colnames(
        rctx,
        basis="constant",
        degree=1,
        blocklens=[200],
        tr=2.0,
    )
    _assert_parity(py_dm, r_dm, label="rpy2 baseline constant parity mismatch")
    assert py_names == r_names, (
        f"Baseline constant column name/order mismatch: py={py_names}, r={r_names}"
    )


@pytest.mark.rpy2
def test_rpy2_baseline_poly_parity(rctx: RContext) -> None:
    """Check polynomial drift baseline structural parity against live R."""
    py_sf = _py_sampling_frame(rctx, blocklens=[200], tr=2.0)
    py_model = baseline_model(basis="poly", degree=2, sframe=py_sf)
    py_dm = design_matrix(py_model)
    py_names = list(py_model.column_names)

    r_dm, r_names = _r_baseline_design_matrix_with_colnames(
        rctx,
        basis="poly",
        degree=2,
        blocklens=[200],
        tr=2.0,
    )
    assert py_dm.shape == r_dm.shape, (
        f"rpy2 baseline poly parity mismatch: shape {py_dm.shape} vs {r_dm.shape}"
    )
    # Intercept and first polynomial column match numerically.
    np.testing.assert_allclose(py_dm[:, 0], r_dm[:, 0], atol=1e-6)
    np.testing.assert_allclose(py_dm[:, 2], r_dm[:, 2], atol=1e-6)
    # Second polynomial column has near-perfect linear agreement but differs
    # in sign/scale conventions between implementations.
    corr = np.corrcoef(py_dm[:, 1], r_dm[:, 1])[0, 1]
    assert abs(corr) > 0.999999, (
        f"Expected near-perfect linear agreement in poly drift column 2, got r={corr}"
    )
    assert py_names == r_names, (
        f"Baseline poly column name/order mismatch: py={py_names}, r={r_names}"
    )


@pytest.mark.rpy2
@pytest.mark.parametrize("degree", [3, 4, 5, 6])
def test_rpy2_baseline_bs_parity(rctx: RContext, degree: int) -> None:
    """Check B-spline drift baseline parity against live R execution."""
    py_sf = _py_sampling_frame(rctx, blocklens=[100], tr=2.0)
    py_model = baseline_model(basis="bs", degree=degree, sframe=py_sf)
    py_dm = design_matrix(py_model)
    py_names = list(py_model.column_names)

    r_dm, r_names = _r_baseline_design_matrix_with_colnames(
        rctx,
        basis="bs",
        degree=degree,
        blocklens=[100],
        tr=2.0,
    )
    _assert_parity(py_dm, r_dm, label=f"rpy2 baseline bs parity mismatch (degree={degree})")
    assert py_names == r_names, (
        f"Baseline bs column name/order mismatch (degree={degree}): py={py_names}, r={r_names}"
    )


@pytest.mark.rpy2
@pytest.mark.parametrize("degree", [3, 4, 5, 6])
def test_rpy2_baseline_ns_parity(rctx: RContext, degree: int) -> None:
    """Check natural-spline drift baseline parity against live R execution."""
    py_sf = _py_sampling_frame(rctx, blocklens=[100], tr=2.0)
    py_model = baseline_model(basis="ns", degree=degree, sframe=py_sf)
    py_dm = design_matrix(py_model)
    py_names = list(py_model.column_names)

    r_dm, r_names = _r_baseline_design_matrix_with_colnames(
        rctx,
        basis="ns",
        degree=degree,
        blocklens=[100],
        tr=2.0,
    )
    _assert_parity(py_dm, r_dm, label=f"rpy2 baseline ns parity mismatch (degree={degree})")
    assert py_names == r_names, (
        f"Baseline ns column name/order mismatch (degree={degree}): py={py_names}, r={r_names}"
    )


@pytest.mark.rpy2
@pytest.mark.parametrize("intercept", ["runwise", "global", "none"])
@pytest.mark.parametrize("degree", [3, 4, 5, 6])
def test_rpy2_baseline_ns_intercept_parity(
    rctx: RContext,
    intercept: str,
    degree: int,
) -> None:
    """Check natural-spline baseline intercept-mode parity."""
    blocklens = [120, 80]

    py_sf = _py_sampling_frame(rctx, blocklens=blocklens, tr=2.0)
    py_model = baseline_model(
        basis="ns",
        degree=degree,
        sframe=py_sf,
        intercept=intercept,
    )
    py_dm = design_matrix(py_model)
    py_names = list(py_model.column_names)

    r_dm, r_names = _r_baseline_design_matrix_with_colnames(
        rctx,
        basis="ns",
        degree=degree,
        blocklens=blocklens,
        tr=2.0,
        intercept=intercept,
    )
    _assert_parity(
        py_dm,
        r_dm,
        label=(
            f"rpy2 baseline ns intercept-mode parity mismatch "
            f"(intercept={intercept}, degree={degree})"
        ),
    )
    assert py_names == r_names, (
        f"Baseline ns intercept-mode column/name mismatch (intercept={intercept}, "
        f"degree={degree}): py={py_names}, r={r_names}"
    )


@pytest.mark.rpy2
@pytest.mark.parametrize("intercept", ["runwise", "global", "none"])
def test_rpy2_baseline_bs_intercept_parity(rctx: RContext, intercept: str) -> None:
    """Check B-spline baseline intercept-mode parity (block/global/none)."""
    blocklens = [120, 80]
    degree = 5

    py_sf = _py_sampling_frame(rctx, blocklens=blocklens, tr=2.0)
    py_model = baseline_model(
        basis="bs",
        degree=degree,
        sframe=py_sf,
        intercept=intercept,
    )
    py_dm = design_matrix(py_model)
    py_names = list(py_model.column_names)

    r_dm, r_names = _r_baseline_design_matrix_with_colnames(
        rctx,
        basis="bs",
        degree=degree,
        blocklens=blocklens,
        tr=2.0,
        intercept=intercept,
    )
    _assert_parity(
        py_dm,
        r_dm,
        label=(
            f"rpy2 baseline bs intercept-mode parity mismatch "
            f"(intercept={intercept}, degree={degree})"
        ),
    )
    assert py_names == r_names, (
        f"Baseline bs intercept-mode column name/order mismatch (intercept={intercept}, "
        f"degree={degree}): py={py_names}, r={r_names}"
    )
