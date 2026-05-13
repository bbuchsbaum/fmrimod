"""Main entry point for fMRI GLM fitting.

Provides ``fmri_lm()`` — the primary user-facing function — and the
:class:`FmriLm` result class that holds fitted coefficients, residual
information, and methods for computing contrasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
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
    contrast_f_vectorized,
    contrast_t,
    contrast_t_batch,
)
from .engine import DEFAULT_ENGINE_OPTIONS, EngineResult, EngineSelector
from .solver import Projection


# ── Fit-level reproducibility metadata (VISION.md:99-103) ────────────────

MaskMode = Literal["volume", "surface", "explicit", "none"]
SeedStatus = Literal["not_randomized", "randomized", "unknown", "not_yet_carried"]
CarryStatus = Literal["carried", "unknown", "not_yet_carried"]


@dataclass(frozen=True)
class FitProvenance:
    """Operational reproducibility metadata carried on every fit.

    Operationalizes the VISION.md:99-103 commitment that "designs and
    fits carry the seed, the solver path, the HRF normalization knob,
    the AR configuration, the masking mode, and the version that
    produced them." Slice A of that work (bd-01KRGWNV5WGD71QZ1E6YY3ACRG)
    populates the three trivially-derivable fields at fit time. The
    other three value slots remain ``None`` with explicit companion
    status fields until the wiring slices land. The dataclass shape is
    otherwise closed.
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

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible provenance payload."""
        return {
            "schema_version": "FitProvenance/v1",
            "fmrimod_version": self.fmrimod_version,
            "solver_path": self.solver_path,
            "hrf_norm_modes": list(self.hrf_norm_modes),
            "seed": self.seed,
            "seed_status": self.seed_status,
            "ar_config": None if self.ar_config is None else {
                "struct": self.ar_config.struct,
                "p": self.ar_config.p,
                "iter_gls": self.ar_config.iter_gls,
                "global_ar": self.ar_config.global_ar,
                "voxelwise": self.ar_config.voxelwise,
                "exact_first": self.ar_config.exact_first,
                "censor": self.ar_config.censor,
                "method": self.ar_config.method,
                "q": self.ar_config.q,
                "p_max": self.ar_config.p_max,
                "pooling": self.ar_config.pooling,
                "convergence_tol": self.ar_config.convergence_tol,
                "parcels": None,
            },
            "ar_status": self.ar_status,
            "mask_mode": self.mask_mode,
            "mask_status": self.mask_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FitProvenance":
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
        )

    def to_json(self) -> str:
        """Serialize provenance as stable, sorted JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "FitProvenance":
        """Reconstruct provenance from :meth:`to_json` output."""
        return cls.from_dict(json.loads(payload))


def _term_norm_mode(term: Any) -> Optional[NormMode]:
    """Recover the declared HRF normalization mode for a term, if any."""
    hrf = getattr(term, "hrf", None)
    if hrf is None:
        return None
    mode = getattr(hrf, "norm_mode", None)
    if mode is None:
        return None
    return cast(NormMode, mode)


def _build_fit_provenance(model: Any, engine: Any) -> FitProvenance:
    """Construct the Slice A provenance object from a fitted model."""
    from fmrimod import __version__ as _fmrimod_version

    event_model = getattr(model, "event_model", model)
    terms = getattr(event_model, "terms", ())
    hrf_norm_modes: Tuple[Optional[NormMode], ...] = tuple(
        _term_norm_mode(t) for t in terms
    )
    solver_path = type(engine).__name__
    return FitProvenance(
        fmrimod_version=_fmrimod_version,
        solver_path=solver_path,
        hrf_norm_modes=hrf_norm_modes,
    )

if TYPE_CHECKING:
    from fmrimod.contrast.omnibus import OmnibusContrast
    from fmrimod.dataset import FmriDataset
    from fmrimod.dataset.protocols import DatasetProtocol
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
    run_results: Optional[List[Any]] = None
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

    # -- Contrast computation --

    def contrast(
        self,
        spec: Union[NDArray[np.float64], str, dict[str, Any], "OmnibusContrast"],
        name: Optional[str] = None,
    ) -> ContrastResult:
        """Compute a contrast on the fitted model.

        Parameters
        ----------
        spec : NDArray, str, dict, or OmnibusContrast
            Contrast specification. Can be:
            - A 1-D vector for a t-contrast
            - A 2-D matrix for an F-contrast
            - A string name referring to a pre-defined contrast in
              the event model
            - A dict ``{"weights": array, "name": str}``
            - An :class:`~fmrimod.contrast.OmnibusContrast` typed intent
              value, resolved against the fit's :class:`DesignColumns`
        name : str, optional
            Override contrast name.

        Returns
        -------
        ContrastResult
        """
        from fmrimod.contrast.omnibus import OmnibusContrast

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
            ),
        )

    def _compute_contrast(
        self,
        weights: NDArray[np.float64],
        name: str,
        intent: ContrastIntent | None = None,
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
    ) -> tuple[dict[str, Any], ...]:
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
) -> dict[str, Any]:
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

    # -- Resolve first positional: spec vs. model ---------------------------
    if _is_fmri_model_like(spec_or_model):
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

    return FmriLm(
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
        provenance=_build_fit_provenance(model, eng),
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
    em_kwargs: Dict[str, Any] = dict(
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


def _engine_result_to_dict(er: "EngineResult") -> Dict[str, Any]:
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
    d: Dict[str, Any],
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
