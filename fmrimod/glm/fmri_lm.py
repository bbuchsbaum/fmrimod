"""Main entry point for fMRI GLM fitting.

Provides ``fmri_lm()`` — the primary user-facing function — and the
:class:`FmriLm` result class that holds fitted coefficients, residual
information, and methods for computing contrasts.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from ..hrf.normalization import NormMode
from ..model.config import AROptions, FmriLmConfig
from .contrasts import (
    ContrastIntent,
    ContrastResult,
    basis_label,
    contrast_f_vectorized,
    contrast_t,
    contrast_t_batch,
    design_id,
    provenance_id,
    weights_payload,
)
from .engine import DEFAULT_ENGINE_OPTIONS, EngineResult, EngineSelector
from .solver import ConditionReport, Projection, RunConditionReport

# ── Fit-level reproducibility metadata (VISION.md:99-103) ────────────────

MaskMode = Literal["volume", "surface", "explicit", "none"]
SeedStatus = Literal["not_randomized", "randomized", "unknown", "not_yet_carried"]
CarryStatus = Literal["carried", "unknown", "not_yet_carried"]


def _jsonable(value: object) -> object:
    """Convert NumPy/list-like config leaves to stable JSON values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _ar_options_to_dict(ar_config: AROptions) -> Dict[str, object]:
    """Serialize AR options without leaving NumPy arrays in the payload."""
    return {
        "struct": ar_config.struct,
        "p": ar_config.p,
        "iter_gls": ar_config.iter_gls,
        "global_ar": ar_config.global_ar,
        "voxelwise": ar_config.voxelwise,
        "exact_first": ar_config.exact_first,
        "censor": _jsonable(ar_config.censor),
        "method": ar_config.method,
        "q": ar_config.q,
        "p_max": ar_config.p_max,
        "pooling": ar_config.pooling,
        "convergence_tol": ar_config.convergence_tol,
        "parcels": _jsonable(ar_config.parcels),
    }


def _snapshot_ar_options(ar_config: AROptions) -> AROptions:
    """Copy AR options into an immutable provenance-friendly shape."""
    return AROptions(**_ar_options_to_dict(ar_config))


@dataclass(frozen=True)
class FitProvenance:
    """Operational reproducibility metadata carried on every fit.

    Operationalizes the VISION.md:99-103 commitment that "designs and
    fits carry the seed, the solver path, the HRF normalization knob,
    the AR configuration, the masking mode, and the version that
    produced them." Slice A of that work (bd-01KRGWNV5WGD71QZ1E6YY3ACRG)
    populates the fields as they become truthfully available. The AR
    slot stores a JSON-safe snapshot of :class:`AROptions` rather than
    the mutable config object itself. Value slots that cannot yet be
    populated remain ``None`` with explicit companion status fields.
    """

    fmrimod_version: str
    solver_path: str
    hrf_norm_modes: Tuple[Optional[NormMode], ...]
    seed: Optional[int] = None
    seed_status: SeedStatus = "not_randomized"
    ar_config: Optional[AROptions] = None
    ar_status: CarryStatus = "not_yet_carried"
    mask_mode: Optional[MaskMode] = None
    mask_status: CarryStatus = "not_yet_carried"
    design_source: Optional[str] = None
    design_source_status: CarryStatus = "not_yet_carried"

    def __post_init__(self) -> None:
        """Normalize config snapshots even for direct construction."""
        if self.ar_config is not None:
            object.__setattr__(
                self,
                "ar_config",
                _snapshot_ar_options(self.ar_config),
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-compatible provenance payload."""
        return {
            "schema_version": "FitProvenance/v1",
            "fmrimod_version": self.fmrimod_version,
            "solver_path": self.solver_path,
            "hrf_norm_modes": list(self.hrf_norm_modes),
            "seed": self.seed,
            "seed_status": self.seed_status,
            "ar_config": None
            if self.ar_config is None
            else _ar_options_to_dict(self.ar_config),
            "ar_status": self.ar_status,
            "mask_mode": self.mask_mode,
            "mask_status": self.mask_status,
            "design_source": self.design_source,
            "design_source_status": self.design_source_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FitProvenance":
        """Reconstruct a provenance object from :meth:`to_dict` output."""
        if payload.get("schema_version") != "FitProvenance/v1":
            raise ValueError("unsupported FitProvenance schema_version")
        ar_payload = payload.get("ar_config")
        ar_config = None if ar_payload is None else AROptions(**dict(ar_payload))
        return cls(
            fmrimod_version=str(payload["fmrimod_version"]),
            solver_path=str(payload["solver_path"]),
            hrf_norm_modes=tuple(payload["hrf_norm_modes"]),
            seed=payload.get("seed"),
            seed_status=cast(SeedStatus, payload["seed_status"]),
            ar_config=ar_config,
            ar_status=cast(CarryStatus, payload["ar_status"]),
            mask_mode=cast(Optional[MaskMode], payload.get("mask_mode")),
            mask_status=cast(CarryStatus, payload["mask_status"]),
            design_source=(
                None
                if payload.get("design_source") is None
                else str(payload["design_source"])
            ),
            design_source_status=cast(
                CarryStatus,
                payload.get("design_source_status", "not_yet_carried"),
            ),
        )

    def to_json(self) -> str:
        """Serialize provenance as stable, sorted JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "FitProvenance":
        """Reconstruct provenance from :meth:`to_json` output."""
        return cls.from_dict(json.loads(payload))

    @property
    def completeness_errors(self) -> tuple[str, ...]:
        """Reasons this provenance is not sufficient for exact replay.

        Status companions make partial provenance honest for ordinary fit
        inspection. Reproduction and receipt consumers need a stricter call
        site: if any status says "unknown" or "not_yet_carried", they must
        refuse the payload instead of treating optional fields as complete.
        """

        errors: list[str] = []
        if not self.fmrimod_version:
            errors.append("fmrimod_version is empty")
        if not self.solver_path:
            errors.append("solver_path is empty")

        if self.seed_status == "randomized" and self.seed is None:
            errors.append("seed_status is randomized but seed is missing")
        elif self.seed_status in {"unknown", "not_yet_carried"}:
            errors.append(f"seed provenance is {self.seed_status}")

        if self.ar_status != "carried":
            errors.append(f"ar_config provenance is {self.ar_status}")
        elif self.ar_config is None:
            errors.append("ar_status is carried but ar_config is missing")

        if self.mask_status != "carried":
            errors.append(f"mask provenance is {self.mask_status}")
        elif self.mask_mode is None:
            errors.append("mask_status is carried but mask_mode is missing")

        if self.design_source_status != "carried":
            errors.append(f"design_source provenance is {self.design_source_status}")
        elif self.design_source is None:
            errors.append(
                "design_source_status is carried but design_source is missing"
            )

        return tuple(errors)

    @property
    def is_complete(self) -> bool:
        """Whether this payload is sufficient for exact replay consumers."""

        return not self.completeness_errors

    def require_complete(self) -> "CompleteFitProvenance":
        """Return a complete-provenance wrapper or raise a typed error."""

        return CompleteFitProvenance.from_provenance(self)


class IncompleteFitProvenanceError(ValueError):
    """Raised when a replay/receipt consumer requires complete provenance."""


@dataclass(frozen=True)
class CompleteFitProvenance:
    """Typed boundary for consumers that require exact replay provenance."""

    provenance: FitProvenance

    @classmethod
    def from_provenance(cls, provenance: FitProvenance) -> "CompleteFitProvenance":
        errors = provenance.completeness_errors
        if errors:
            raise IncompleteFitProvenanceError(
                "FitProvenance is incomplete for exact replay: "
                + "; ".join(errors)
            )
        return cls(provenance=provenance)

    def to_dict(self) -> Dict[str, object]:
        """Return the wrapped complete provenance as a JSON-ready payload."""

        return self.provenance.to_dict()

    def to_json(self) -> str:
        """Serialize the wrapped complete provenance."""

        return self.provenance.to_json()


def _term_norm_mode(term: object) -> Optional[NormMode]:
    """Recover the declared HRF normalization mode for a term, if any."""
    hrf = getattr(term, "hrf", None)
    if hrf is None:
        return None
    mode = getattr(hrf, "norm_mode", None)
    if mode is None:
        return None
    return cast(NormMode, mode)


def _seed_provenance(
    engine: Any,
    fit_kwargs: Mapping[str, object],
) -> tuple[Optional[int], SeedStatus]:
    """Recover seed provenance from the resolved engine invocation."""
    seed = fit_kwargs.get("seed")
    if seed is not None:
        return int(seed), "randomized"

    if getattr(engine, "name", None) == "sketch":
        return None, "unknown"

    return None, "not_randomized"


def _mask_mode_provenance(model: Any) -> tuple[Optional[MaskMode], CarryStatus]:
    """Infer mask mode from the fitted dataset without adding new API."""
    dataset = getattr(model, "dataset", None)
    if dataset is None or not hasattr(dataset, "get_mask"):
        return None, "unknown"
    try:
        mask = np.asarray(dataset.get_mask(), dtype=bool)
    except Exception:
        return None, "unknown"

    if mask.size == 0:
        return None, "unknown"
    if bool(np.all(mask)):
        return "none", "carried"
    if mask.ndim >= 3:
        return "volume", "carried"
    return "explicit", "carried"


def _design_source_provenance(model: object) -> tuple[str, CarryStatus]:
    """Return the source of the realized design used for this fit."""
    source = getattr(model, "design_source", None)
    if source is not None:
        return str(source), "carried"
    if getattr(model, "event_model", None) is not None:
        return "spec", "carried"
    return "model", "carried"


def _build_fit_provenance(
    model: Any,
    config: FmriLmConfig,
    engine: Any,
    fit_kwargs: Mapping[str, object] | None = None,
) -> FitProvenance:
    """Construct the Slice A provenance object from a fitted model."""
    from fmrimod import __version__ as _fmrimod_version

    fit_kwargs = fit_kwargs or {}
    event_model = getattr(model, "event_model", model)
    terms = getattr(event_model, "terms", ())
    hrf_norm_modes: Tuple[Optional[NormMode], ...] = tuple(
        _term_norm_mode(t) for t in terms
    )
    solver_path = type(engine).__name__
    seed, seed_status = _seed_provenance(engine, fit_kwargs)
    mask_mode, mask_status = _mask_mode_provenance(model)
    design_source, design_source_status = _design_source_provenance(model)
    return FitProvenance(
        fmrimod_version=_fmrimod_version,
        solver_path=solver_path,
        hrf_norm_modes=hrf_norm_modes,
        seed=seed,
        seed_status=seed_status,
        ar_config=config.ar,
        ar_status="carried",
        mask_mode=mask_mode,
        mask_status=mask_status,
        design_source=design_source,
        design_source_status=design_source_status,
    )

if TYPE_CHECKING:
    from fmrimod.contrast.contrast_spec import ContrastSpec
    from fmrimod.contrast.omnibus import OmnibusContrast
    from fmrimod.contrast.semantic import LinearSemanticContrast
    from fmrimod.dataset import FmriDataset
    from fmrimod.dataset.protocols import DatasetProtocol
    from fmrimod.design import RealizedDesign
    from fmrimod.glm.spatial import SpatialContext
    from fmrimod.model.fmri_model import FmriModel
    from fmrimod.spec import Spec, Term


@runtime_checkable
class FmriModelLike(Protocol):
    """Runtime-checkable interface required by GLM fitting engines."""

    @property
    def dataset(self) -> object:
        """Dataset associated with the model."""
        ...

    @property
    def n_runs(self) -> int:
        """Number of runs in the model."""
        ...

    def design_matrix_array(self, run: int = 0) -> NDArray[np.float64]:
        """Return the design matrix for a run."""
        ...


class _RealizedDesignModel:
    """Adapt a typed realized design plus dataset to the fitting protocol."""

    def __init__(self, design: "RealizedDesign", dataset: object) -> None:
        self._design = design
        self.dataset = dataset
        self._run_lengths = _dataset_run_lengths(dataset, design.n_timepoints)
        total = sum(self._run_lengths)
        if total != design.n_timepoints:
            raise ValueError(
                "RealizedDesign row count must match dataset timepoints: "
                f"{design.n_timepoints} rows vs {total} dataset timepoints"
            )
        self.n_runs = len(self._run_lengths)

    @property
    def design_source(self) -> str:
        """Source label carried into fit provenance."""
        return self._design.source

    @property
    def n_timepoints(self) -> list[int]:
        """Per-run timepoint counts."""
        return list(self._run_lengths)

    def design_matrix_array(self, run: int = 0) -> NDArray[np.float64]:
        """Return the realized design matrix for a single run."""
        start, stop = self._run_slice(run)
        return np.asarray(self._design.matrix[start:stop, :], dtype=np.float64)

    def design_matrix(self, run: Optional[int] = None):
        """Return the realized design as a named DataFrame."""
        import pandas as pd

        if run is None:
            return self._design.as_dataframe()
        start, stop = self._run_slice(run)
        return pd.DataFrame(
            np.asarray(self._design.matrix[start:stop, :], dtype=np.float64),
            columns=list(self._design.column_names),
        )

    def design_columns(self):
        """Return typed column provenance for the realized design."""
        return self._design.design_columns()

    def contrast_weights(self) -> dict[str, NDArray[np.float64]]:
        """Realized designs do not invent named contrasts."""
        return {}

    def _run_slice(self, run: int) -> tuple[int, int]:
        if run < 0 or run >= self.n_runs:
            raise IndexError(f"run {run} out of range for {self.n_runs} runs")
        starts = np.cumsum([0, *self._run_lengths[:-1]])
        start = int(starts[run])
        return start, start + int(self._run_lengths[run])


def _dataset_run_lengths(dataset: object, default_total: int) -> list[int]:
    """Return per-run lengths from a dataset-like object."""
    if hasattr(dataset, "run_lengths"):
        lengths = getattr(dataset, "run_lengths")
    elif hasattr(dataset, "n_timepoints"):
        lengths = getattr(dataset, "n_timepoints")
    else:
        lengths = default_total

    if isinstance(lengths, int):
        return [int(lengths)]
    return [int(value) for value in lengths]


@dataclass(frozen=True)
class _FitColumnTerm:
    """Minimal term adapter for formula-backed ContrastSpec resolution."""

    columns: object

    def conditions(
        self,
        drop_empty: bool = False,
        expand_basis: bool = True,
    ) -> list[str]:
        """Return realized design-column names as ContrastSpec conditions."""
        del drop_empty, expand_basis
        names = getattr(self.columns, "names", None)
        if names is not None:
            return [str(name) for name in names]
        return [str(name) for name in self.columns]  # type: ignore[union-attr]


def _contrast_spec_weights_to_fit_weights(
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert ContrastSpec weights from condition-by-contrast to fit shape."""
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(
            f"ContrastSpec weights must be 1-D or 2-D; got {arr.ndim}-D"
        )
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr.T


@dataclass
class FmriLm:
    """Result of fitting a GLM to fMRI data.

    Holds beta coefficients, residual information, and pre-computed
    quantities needed for efficient contrast computation.

    Attributes
    ----------
    betas : NDArray
        Coefficient matrix, shape ``(p, V)`` where ``p`` is the number
        of design columns and ``V`` the number of voxels.
    sigma : NDArray
        Residual standard deviation per voxel, shape ``(V,)``.
    residual_df : float
        Residual degrees of freedom.
    XtXinv : NDArray
        ``(X'X)^{-1}`` matrix, shape ``(p, p)``.
    model : object
        The :class:`~fmrimod.model.FmriModel` that was fitted.
    config : FmriLmConfig
        Configuration used for fitting.
    contrasts : dict
        Dictionary of computed :class:`ContrastResult` objects.
    ar_params : NDArray or None
        Estimated AR parameters, shape ``(ar_order, V)`` or ``(ar_order,)``
        for global estimation.
    robust_weights : NDArray or None
        IRLS weights from robust fitting, shape ``(n, V)``.
    run_results : list
        Per-run fitting results (for diagnostics).
    projections : list
        Per-run :class:`Projection` objects.
    """

    betas: NDArray[np.float64]
    sigma: NDArray[np.float64]
    residual_df: float
    XtXinv: NDArray[np.float64]
    model: FmriModelLike
    config: FmriLmConfig
    contrasts: Dict[str, ContrastResult] = field(default_factory=dict)
    ar_params: Optional[NDArray[np.float64]] = None
    robust_weights: Optional[NDArray[np.float64]] = None
    run_results: Optional[List[object]] = None
    projections: Optional[List[Projection]] = None
    provenance: Optional[FitProvenance] = None
    _named_weights_cache: Optional[Dict[str, NDArray[np.float64]]] = field(
        default=None,
        init=False,
        repr=False,
    )

    # -- Accessors --

    def coef(self) -> NDArray[np.float64]:
        """Return the coefficient matrix ``(p, V)``."""
        return self.betas

    def se(self) -> NDArray[np.float64]:
        """Return standard errors for each coefficient, shape ``(p, V)``.

        ``SE_{j,v} = sigma_v * sqrt(XtXinv_{j,j})``
        """
        diag_XtXinv = np.diag(self.XtXinv)
        return np.asarray(
            self.sigma[np.newaxis, :] * np.sqrt(
                np.maximum(diag_XtXinv, 0.0)
            )[:, np.newaxis],
            dtype=np.float64,
        )

    def tstat(self) -> NDArray[np.float64]:
        """Return t-statistics for each coefficient, shape ``(p, V)``."""
        se_vals = self.se()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se_vals > 1e-15, self.betas / se_vals, 0.0)

    @property
    def n_voxels(self) -> int:
        """Number of voxels."""
        return self.betas.shape[1]

    @property
    def n_coefficients(self) -> int:
        """Number of regression coefficients."""
        return self.betas.shape[0]

    def design_columns(self):
        """Return typed provenance for the fitted design columns."""
        design_columns = getattr(self.model, "design_columns", None)
        if not callable(design_columns):
            raise TypeError("fitted model does not expose design_columns()")
        return design_columns()

    def _contrast_intent_payload_fields(
        self,
        weights: NDArray[np.float64],
    ) -> dict[str, object]:
        """Return production-derived payload fields for seam equality."""
        return {
            "basis_label": basis_label(self),
            "weights": weights_payload(weights),
            "design_id": design_id(self),
            "provenance_id": provenance_id(self),
        }

    # -- Rank diagnostics --

    def condition_report(self) -> ConditionReport:
        """Return rank/conditioning diagnostics for the fitted design.

        For each run, reports the design-matrix column count, numerical
        rank, residual degrees of freedom, and a best-effort list of
        column names that the rank-revealing QR identified as linearly
        dependent on earlier columns. Aggregate-level
        :attr:`ConditionReport.is_full_rank`,
        :attr:`ConditionReport.ill_conditioned`, and
        :attr:`ConditionReport.aliased_columns` summarise across runs.

        Useful for diagnosing collinear nuisance regressors, aliased task
        regressors, or any other design pathology routing the OLS solve
        through the SVD pseudoinverse path.
        """
        projections = self.projections or []
        names: list[str] | None
        try:
            names = list(self.design_columns().names)
        except Exception:  # pragma: no cover - design without typed columns
            names = None

        runs: list[RunConditionReport] = []
        for run_idx, proj in enumerate(projections):
            aliased: tuple[str, ...]
            if names is not None and proj.aliased_indices:
                aliased = tuple(
                    names[i] if 0 <= i < len(names) else f"col_{i}"
                    for i in proj.aliased_indices
                )
            else:
                aliased = tuple(
                    f"col_{i}" for i in proj.aliased_indices
                )
            runs.append(
                RunConditionReport(
                    run=run_idx,
                    n_columns=int(self.n_coefficients),
                    rank=int(proj.rank),
                    is_full_rank=bool(proj.is_full_rank),
                    ill_conditioned=bool(proj.ill_conditioned),
                    dfres=float(proj.dfres),
                    aliased_columns=aliased,
                )
            )
        return ConditionReport(runs=tuple(runs))

    @property
    def is_full_rank(self) -> bool:
        """True iff every run's realised design matrix is full rank."""
        projections = self.projections or []
        if not projections:
            return True
        return all(proj.is_full_rank for proj in projections)

    @property
    def ill_conditioned(self) -> bool:
        """True iff any run's solve was routed through the SVD pseudoinverse."""
        projections = self.projections or []
        return any(proj.ill_conditioned for proj in projections)

    # -- Contrast computation --

    def contrast(
        self,
        spec: Union[
            NDArray[np.float64],
            str,
            dict[str, object],
            "OmnibusContrast",
            "ContrastSpec",
            "LinearSemanticContrast",
        ],
        name: Optional[str] = None,
    ) -> ContrastResult:
        """Compute a contrast on the fitted model.

        Parameters
        ----------
        spec : NDArray, str, dict, OmnibusContrast, or ContrastSpec
            Contrast specification. Can be:
            - A 1-D vector for a t-contrast
            - A 2-D matrix for an F-contrast
            - A string name referring to a pre-defined contrast in
              the event model
            - A dict ``{"weights": array, "name": str}``
            - An :class:`~fmrimod.contrast.OmnibusContrast` typed intent
              value, resolved against the fit's :class:`DesignColumns`
            - A semantic condition/cell contrast authored with
              :func:`~fmrimod.contrast.condition` or
              :func:`~fmrimod.contrast.cell`
            - A formula-backed :class:`~fmrimod.contrast.ContrastSpec`
              value, resolved against realized design-column names
        name : str, optional
            Override contrast name.

        Returns
        -------
        ContrastResult
        """
        from fmrimod.contrast.contrast_spec import ContrastSpec
        from fmrimod.contrast.contrast_weights import contrast_weights
        from fmrimod.contrast.omnibus import OmnibusContrast
        from fmrimod.contrast.semantic import (
            ConditionRef,
            LinearSemanticContrast,
            SemanticContrast,
        )

        if isinstance(spec, OmnibusContrast):
            weights = spec.resolve(self.design_columns())
            return self._compute_contrast(
                weights,
                name=name or spec.display_name,
                intent=ContrastIntent(
                    kind="omnibus",
                    name=name or spec.display_name,
                    term=spec.term,
                    levels=spec.levels,
                    rows=int(weights.shape[0]),
                    **self._contrast_intent_payload_fields(weights),
                ),
            )

        if isinstance(spec, ConditionRef):
            semantic = SemanticContrast(positive=spec, name=name or spec.display_name)
            weights = semantic.resolve(self.design_columns())
            intent_payload = semantic.intent(rows=1)
            intent_payload.update(self._contrast_intent_payload_fields(weights))
            return self._compute_contrast(
                weights,
                name=name or semantic.display_name,
                intent=intent_payload,
            )

        if isinstance(spec, SemanticContrast):
            weights = spec.resolve(self.design_columns())
            intent_payload = spec.intent(rows=1)
            intent_payload.update(self._contrast_intent_payload_fields(weights))
            return self._compute_contrast(
                weights,
                name=name or spec.display_name,
                intent=intent_payload,
            )

        if isinstance(spec, LinearSemanticContrast):
            weights = spec.resolve(self.design_columns())
            intent_payload = spec.intent(rows=1)
            intent_payload.update(self._contrast_intent_payload_fields(weights))
            return self._compute_contrast(
                weights,
                name=name or spec.display_name,
                intent=intent_payload,
            )

        if isinstance(spec, ContrastSpec):
            resolved = contrast_weights(spec, _FitColumnTerm(self.design_columns()))
            weights = _contrast_spec_weights_to_fit_weights(resolved.weights)
            cname = name or resolved.name
            return self._compute_contrast(
                weights,
                name=cname,
                intent=ContrastIntent(
                    kind="contrast_spec",
                    name=cname,
                    term=type(spec).__name__,
                    rows=int(np.atleast_2d(weights).shape[0]),
                    **self._contrast_intent_payload_fields(weights),
                ),
            )

        if isinstance(spec, str):
            if name is None and spec in self.contrasts:
                return self.contrasts[spec]
            # Look up from model's contrast weights
            if self._named_weights_cache is None:
                contrast_weights = getattr(self.model, "contrast_weights", None)
                self._named_weights_cache = (
                    cast(Dict[str, NDArray[np.float64]], contrast_weights())
                    if callable(contrast_weights)
                    else {}
                )
            cw = self._named_weights_cache
            if spec not in cw:
                raise KeyError(f"Unknown contrast name: {spec!r}")
            weights = cw[spec]
            return self._compute_contrast(
                weights,
                name=name or spec,
                intent=ContrastIntent(
                    kind="named",
                    name=spec,
                    rows=int(np.atleast_2d(weights).shape[0]),
                    **self._contrast_intent_payload_fields(weights),
                ),
            )

        if isinstance(spec, dict):
            if "weights" not in spec:
                raise ValueError("Contrast dict spec must contain 'weights'")
            weights = np.asarray(spec["weights"], dtype=np.float64)
            cname = name or spec.get("name", "contrast")
            return self._compute_contrast(
                weights,
                name=cname,
                intent=ContrastIntent(
                    kind="dict",
                    name=cname,
                    rows=int(np.atleast_2d(weights).shape[0]),
                    **self._contrast_intent_payload_fields(weights),
                ),
            )

        weights = np.asarray(spec, dtype=np.float64)
        cname = name or "contrast"
        return self._compute_contrast(
            weights,
            name=cname,
            intent=ContrastIntent(
                kind="array",
                name=cname,
                rows=int(np.atleast_2d(weights).shape[0]),
                **self._contrast_intent_payload_fields(weights),
            ),
        )

    def _compute_contrast(
        self,
        weights: NDArray[np.float64],
        name: str,
        intent: ContrastIntent | dict[str, object] | None = None,
    ) -> ContrastResult:
        """Dispatch to t or F contrast based on weight dimensions."""
        weights = np.atleast_1d(weights)
        if weights.ndim == 1:
            result = contrast_t(
                weights, self.betas, self.XtXinv, self.sigma,
                self.residual_df, name=name,
            )
        else:
            result = contrast_f_vectorized(
                weights, self.betas, self.XtXinv, self.sigma,
                self.residual_df, name=name,
            )
        if intent is None:
            intent = ContrastIntent(
                kind="array",
                name=name,
                rows=int(np.atleast_2d(weights).shape[0]),
                **self._contrast_intent_payload_fields(weights),
            )
        column_details = self._touched_column_details(weights)
        result.intent = intent
        result.touched_columns = tuple(
            str(column["name"]) for column in column_details
        )
        result.touched_column_details = column_details
        result.spatial = self._spatial_context()
        self.contrasts[name] = result
        return result

    def _touched_column_details(
        self,
        weights: NDArray[np.float64],
    ) -> tuple[dict[str, object], ...]:
        """Return realized design-column details touched by a contrast."""
        weights_2d = np.atleast_2d(np.asarray(weights, dtype=np.float64))
        active = np.flatnonzero(np.any(np.abs(weights_2d) > 0.0, axis=0))
        try:
            columns = self.design_columns()
        except (AttributeError, TypeError):
            columns = ()
        if not columns:
            return tuple(
                {
                    "name": f"column[{int(index)}]",
                    "index": int(index),
                    "term": None,
                    "level": None,
                    "condition": None,
                    "basis_ix": None,
                    "provenance": {},
                }
                for index in active
            )
        return tuple(
            _design_column_detail(columns[int(index)], fallback_index=int(index))
            for index in active
        )

    def _spatial_context(self) -> SpatialContext | None:
        """Return the spatial context for this fit, if its model exposes one.

        Cached so repeated contrast calls don't re-walk the adapter.
        """
        from fmrimod.glm.spatial import SpatialContext

        if not hasattr(self, "_spatial_context_cache"):
            self._spatial_context_cache = SpatialContext.from_model(self.model)
        return self._spatial_context_cache

    def compute_contrasts(
        self,
        specs: Dict[str, NDArray[np.float64]],
    ) -> Dict[str, ContrastResult]:
        """Compute multiple contrasts, batching t-contrasts for speed.

        Parameters
        ----------
        specs : dict[str, NDArray]
            Mapping of contrast name to weight vector/matrix.

        Returns
        -------
        dict[str, ContrastResult]
            Computed contrast results keyed by name.
        """
        if not specs:
            return {}

        out: Dict[str, ContrastResult] = {}
        t_names: list[str] = []
        t_weights: list[NDArray[np.float64]] = []
        f_items: list[tuple[str, NDArray[np.float64]]] = []

        for cname, w in specs.items():
            if cname in self.contrasts:
                out[cname] = self.contrasts[cname]
                continue
            w_arr = np.asarray(w, dtype=np.float64)
            if w_arr.ndim == 1:
                t_names.append(cname)
                t_weights.append(w_arr)
            else:
                f_items.append((cname, np.atleast_2d(w_arr)))

        ctx = self._spatial_context()
        if t_weights:
            t_mat = np.vstack(t_weights)
            t_results = contrast_t_batch(
                t_mat,
                self.betas,
                self.XtXinv,
                self.sigma,
                self.residual_df,
                names=t_names,
            )
            for res, weights in zip(t_results, t_weights):
                column_details = self._touched_column_details(weights)
                res.intent = ContrastIntent(
                    kind="named",
                    name=res.name,
                    rows=1,
                    **self._contrast_intent_payload_fields(weights),
                )
                res.touched_columns = tuple(
                    str(column["name"]) for column in column_details
                )
                res.touched_column_details = column_details
                res.spatial = ctx
                self.contrasts[res.name] = res
                out[res.name] = res

        for cname, w_arr in f_items:
            res = contrast_f_vectorized(
                w_arr,
                self.betas,
                self.XtXinv,
                self.sigma,
                self.residual_df,
                name=cname,
            )
            column_details = self._touched_column_details(w_arr)
            res.intent = ContrastIntent(
                kind="named",
                name=cname,
                rows=int(np.atleast_2d(w_arr).shape[0]),
                **self._contrast_intent_payload_fields(w_arr),
            )
            res.touched_columns = tuple(
                str(column["name"]) for column in column_details
            )
            res.touched_column_details = column_details
            res.spatial = ctx
            self.contrasts[cname] = res
            out[cname] = res

        return out

    # -- Display --

    def __repr__(self) -> str:
        parts = [
            "FmriLm(",
            f"  n_coefficients={self.n_coefficients},",
            f"  n_voxels={self.n_voxels},",
            f"  residual_df={self.residual_df:.1f},",
            f"  config={self.config!r},",
            f"  contrasts={list(self.contrasts.keys())},",
            ")",
        ]
        return "\n".join(parts)


def _design_column_detail(
    column: object,
    *,
    fallback_index: int | None = None,
) -> dict[str, object]:
    """Return JSON-ready metadata for one realized design column."""
    if not hasattr(column, "name"):
        return {
            "name": str(column),
            "index": -1 if fallback_index is None else fallback_index,
            "term": None,
            "level": None,
            "condition": None,
            "basis_ix": None,
            "basis_name": None,
            "basis_total": None,
            "role": None,
            "model_source": None,
            "provenance": {},
        }
    provenance = getattr(column, "provenance", None)
    return {
        "name": str(getattr(column, "name")),
        "index": int(getattr(column, "index")),
        "term": getattr(column, "term", None),
        "level": getattr(column, "level", None),
        "condition": getattr(column, "condition", None),
        "basis_ix": getattr(column, "basis_ix", None),
        "basis_name": getattr(column, "basis_name", None),
        "basis_total": getattr(column, "basis_total", None),
        "role": getattr(column, "role", None),
        "model_source": getattr(column, "model_source", None),
        "provenance": dict(provenance) if provenance is not None else {},
    }


def fmri_lm(
    spec_or_model: "Spec | Term | str | Sequence[object] | FmriModelLike",
    dataset_or_config: "FmriDataset | FmriLmConfig | None" = None,
    *,
    baseline: object = None,
    block: Optional[Union[str, NDArray[np.float64]]] = None,
    durations: Optional[Union[str, float, NDArray[np.float64]]] = None,
    precision: Optional[float] = None,
    config: Optional[FmriLmConfig] = None,
    engine: EngineSelector = DEFAULT_ENGINE_OPTIONS,
    **engine_kwargs: object,
) -> FmriLm:
    """Fit a GLM to fMRI data.

    Canonical form (matches R ``fmrireg::fmri_lm``)::

        fit = fmri_lm(spec, dataset)
        fit = fmri_lm(spec, dataset, config=FmriLmConfig(ar="ar1"))

    Legacy form (pre-built model)::

        fit = fmri_lm(model)
        fit = fmri_lm(model, FmriLmConfig())

    Parameters
    ----------
    spec_or_model
        Either a *spec* (R-style formula string, list of
        :class:`~fmrimod.formula.base.Term`, or
        :class:`~fmrimod.formula.base.EventModelBuilder`) or a pre-built
        :class:`~fmrimod.model.FmriModel`-like object. The second positional
        argument is interpreted accordingly.
    dataset_or_config
        When ``spec_or_model`` is a spec: the :class:`FmriDataset` to fit
        (must carry an event table). When ``spec_or_model`` is a model: an
        :class:`FmriLmConfig` (back-compat positional). May be omitted in
        either case.
    baseline
        Optional pre-built :class:`~fmrimod.baseline.BaselineModel`. If not
        provided in the spec path, defaults to ``constant`` basis with
        runwise intercepts.
    block
        Block/run column name or array passed to
        :func:`~fmrimod.event_model`. Auto-detected from ``run`` or ``block``
        columns when omitted.
    durations
        Event durations passed to :func:`~fmrimod.event_model`. Auto-detected
        from a ``duration`` column when present.
    precision
        Temporal precision for HRF convolution.
    config
        Fitting configuration. Defaults to plain OLS.
    engine
        Typed engine options object. Built-ins are
        :class:`~fmrimod.glm.engine.RunwiseEngineOptions`,
        :class:`~fmrimod.glm.engine.ChunkwiseEngineOptions`, and
        :class:`~fmrimod.glm.engine.SketchEngineOptions`. Legacy string engine
        names remain accepted for compatibility.
    **engine_kwargs
        Legacy keyword bag forwarded to the engine's ``fit()`` method when
        ``engine`` is a string. Do not mix with typed engine options.

    Returns
    -------
    FmriLm
        Fitted model with coefficients, residuals, and contrast methods.

    Examples
    --------
    Canonical three-line shape::

        >>> ds = fm.fmri_dataset(img, mask=mask, tr=2.0, events=events)
        >>> fit = fm.fmri_lm("hrf(trial_type)", ds)
        >>> fit.contrast("trial_type[listening]")

    Pre-built realized design::

        >>> design = fm.design.RealizedDesign.from_array(X, columns, source="nilearn")
        >>> fit = fm.fmri_lm(design, ds)

    Sketch engine::

        >>> from fmrimod.glm import SketchEngineOptions
        >>> fit = fm.fmri_lm(model, engine=SketchEngineOptions(sketch_ratio=0.5))
    """
    from .engine import resolve_engine

    # -- Resolve second positional: dataset vs. config -----------------------
    if isinstance(dataset_or_config, FmriLmConfig):
        if config is not None and config is not dataset_or_config:
            raise ValueError(
                "fmri_lm: `config` was passed both positionally and as a kwarg"
            )
        config = dataset_or_config
        dataset = None
    else:
        dataset = dataset_or_config

    # -- Resolve first positional: realized design vs. spec vs. model -------
    from ..design import RealizedDesign

    if isinstance(spec_or_model, RealizedDesign):
        if dataset is None:
            raise ValueError(
                "fmri_lm(RealizedDesign, dataset) requires a dataset as the "
                "second argument"
            )
        model = _RealizedDesignModel(spec_or_model, dataset)
    elif _is_fmri_model_like(spec_or_model):
        if dataset is not None:
            raise ValueError(
                "fmri_lm: got a pre-built model and a dataset. "
                "Pass either (spec, dataset, ...) or (model, ...), not both."
            )
        model = cast(FmriModelLike, spec_or_model)
    else:
        if dataset is None:
            raise ValueError(
                "fmri_lm(spec, dataset) requires an FmriDataset as the "
                "second argument when the first argument is a spec."
            )
        model = _build_model_from_spec(
            spec=cast("Spec | Term | str | Sequence[object]", spec_or_model),
            dataset=dataset,
            baseline=baseline,
            block=block,
            durations=durations,
            precision=precision,
        )

    if config is None:
        config = FmriLmConfig()

    # Resolve and run the engine
    eng, fit_kwargs = resolve_engine(engine, engine_kwargs)
    eng.preflight(model, config)
    fit_result: EngineResult = eng.fit(model, config, **fit_kwargs)

    # Handle AR modeling if configured
    ar_params = fit_result.ar_params
    if ar_params is None and config.ar.enabled:
        from ..ar.integration import iterative_gls

        # Build a dict compatible with the legacy interface
        legacy = _engine_result_to_dict(fit_result)
        legacy, ar_params = iterative_gls(model, config, legacy)
        fit_result = _dict_to_engine_result(legacy, fit_result)

    # Handle robust fitting if configured
    robust_weights = fit_result.robust_weights
    if robust_weights is None and config.robust.enabled:
        from ..robust.irls import robust_refit

        legacy = _engine_result_to_dict(fit_result)
        legacy, robust_weights = robust_refit(model, config, legacy)
        fit_result = _dict_to_engine_result(legacy, fit_result)

    fit = FmriLm(
        betas=fit_result.betas,
        sigma=fit_result.sigma,
        residual_df=fit_result.dfres,
        XtXinv=fit_result.XtXinv,
        model=model,
        config=config,
        ar_params=ar_params,
        robust_weights=robust_weights,
        run_results=fit_result.run_results,
        projections=fit_result.projections,
        provenance=_build_fit_provenance(model, config, eng, fit_kwargs),
    )
    _warn_if_ill_conditioned(fit)
    return fit


def _warn_if_ill_conditioned(fit: "FmriLm") -> None:
    """Emit a UserWarning when any run's design is rank-deficient.

    The pseudoinverse path produces a valid minimum-norm solution and
    contrasts in the row space of X remain estimable, but individual
    betas on the aliased columns are not uniquely identified. The
    warning surfaces this so a typed-API user is not silently working
    with a rank-deficient design.
    """
    projections = fit.projections or []
    deficient = [
        (idx, proj) for idx, proj in enumerate(projections) if not proj.is_full_rank
    ]
    if not deficient:
        return
    report = fit.condition_report()
    parts = []
    for run in report.runs:
        if run.is_full_rank:
            continue
        names = (
            ", ".join(run.aliased_columns)
            if run.aliased_columns
            else "<unidentified>"
        )
        parts.append(
            f"run={run.run}: rank={run.rank}/{run.n_columns}, "
            f"dfres={run.dfres:g}, aliased columns: {names}"
        )
    detail = "; ".join(parts)
    warnings.warn(
        "fmri_lm(): realised design is rank-deficient. The solver fell "
        "back to the SVD pseudoinverse path, so contrasts in the row "
        "space of X (e.g. linear combinations of identifiable columns) "
        "stay estimable, but individual betas on the aliased columns "
        "are not uniquely identified. "
        f"{detail}. Inspect fit.condition_report() or pass "
        "nuisance_check='drop' to baseline_model() to prune redundant "
        "regressors.",
        UserWarning,
        stacklevel=2,
    )


def _is_fmri_model_like(obj: object) -> bool:
    """Check for the public :class:`FmriModelLike` fitting protocol."""
    return isinstance(obj, FmriModelLike)


def _build_model_from_spec(
    *,
    spec: "Spec | Term | str | Sequence[object]",
    dataset: "FmriDataset",
    baseline: object,
    block: object,
    durations: object,
    precision: Optional[float],
) -> "FmriModel":
    """Build an :class:`FmriModel` from a spec + :class:`FmriDataset`.

    Accepts either a typed :class:`fmrimod.spec.Spec` / :class:`Term` tree or
    a legacy string formula / list-of-Terms. The typed path lowers via
    :func:`fmrimod.spec.compile`. Supported legacy HRF formulas are first
    adapted into the typed spec tree; unsupported legacy forms fall back to
    :func:`fmrimod.event_model` directly.
    """
    from ..baseline.baseline_model import baseline_model as _build_baseline
    from ..design.event_model import event_model as _build_event
    from ..model.fmri_model import FmriModel
    from ..spec import Spec, Term, legacy_formula_to_spec
    from ..spec import compile as _compile_spec

    events_df = getattr(dataset, "event_table", None)
    if events_df is None:
        raise ValueError(
            "fmri_lm(spec, dataset) requires the dataset to carry an event "
            "table; pass `events=` to fmri_dataset(...) or supply a pre-built "
            "FmriModel instead."
        )

    sf = dataset.get_sampling_frame()

    # Auto-detect block column.
    resolved_block = block
    if resolved_block is None:
        if "run" in events_df.columns:
            resolved_block = "run"
        elif "block" in events_df.columns:
            resolved_block = "block"
        else:
            resolved_block = np.ones(len(events_df), dtype=int)

    # Auto-detect durations column.
    resolved_durations = durations
    if resolved_durations is None:
        if "duration" in events_df.columns:
            resolved_durations = "duration"
        else:
            resolved_durations = 0

    # -- Typed Spec / Term path -----------------------------------------
    if isinstance(spec, (Spec, Term)):
        em, default_bm = _compile_spec(
            spec,
            data=events_df,
            sampling_frame=sf,
            block=resolved_block,
            durations=resolved_durations,
            precision=precision,
        )
        bm = baseline if baseline is not None else default_bm
        return FmriModel(em, bm, cast("DatasetProtocol", dataset))

    # -- Convertible legacy HRF formula/list path -----------------------
    try:
        typed_spec = legacy_formula_to_spec(cast(Any, spec))
    except (TypeError, ValueError, NotImplementedError):
        typed_spec = None

    if typed_spec is not None:
        em, default_bm = _compile_spec(
            typed_spec,
            data=events_df,
            sampling_frame=sf,
            block=resolved_block,
            durations=resolved_durations,
            precision=precision,
        )
        bm = baseline if baseline is not None else default_bm
        return FmriModel(em, bm, cast("DatasetProtocol", dataset))

    # -- Legacy string / list path --------------------------------------
    em_kwargs: Dict[str, object] = dict(
        data=events_df,
        block=resolved_block,
        sampling_frame=sf,
        durations=resolved_durations,
    )
    if precision is not None:
        em_kwargs["precision"] = precision

    em = _build_event(cast(Any, spec), **em_kwargs)

    bm = baseline
    if bm is None:
        bm = _build_baseline(basis="constant", sframe=sf, intercept="runwise")

    return FmriModel(em, bm, cast("DatasetProtocol", dataset))


def _engine_result_to_dict(er: "EngineResult") -> Dict[str, object]:
    """Convert an EngineResult to the legacy dict format."""
    return {
        "betas": er.betas,
        "sigma": er.sigma,
        "dfres": er.dfres,
        "XtXinv": er.XtXinv,
        "projections": er.projections,
        "run_results": er.run_results,
        "residuals": er.residuals,
        "run_X": er.run_X,
    }


def _dict_to_engine_result(
    d: Dict[str, object],
    original: "EngineResult",
) -> "EngineResult":
    """Update an EngineResult from a legacy dict (after AR/robust)."""
    from .engine import EngineResult

    return EngineResult(
        betas=d["betas"],
        sigma=d["sigma"],
        dfres=d["dfres"],
        XtXinv=d["XtXinv"],
        projections=d.get("projections"),
        run_results=d.get("run_results"),
        residuals=d.get("residuals"),
        run_X=d.get("run_X"),
        ar_params=getattr(original, "ar_params", None),
        robust_weights=getattr(original, "robust_weights", None),
        extra=getattr(original, "extra", {}),
    )
