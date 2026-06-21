"""Tests for the pluggable engine registry."""

import numpy as np
import pytest

from fmrimod.glm.engine import (
    _ENGINES,
    ChunkwiseEngineOptions,
    EngineResult,
    FittingEngine,
    ReducedRankEngineOptions,
    RunwiseEngineOptions,
    SketchEngineOptions,
    _normalize_engine_name,
    get_engine,
    list_engines,
    register_engine,
    resolve_engine,
)
from fmrimod.model.config import FmriLmConfig


class _OneRunDataset:
    def __init__(self, y):
        self._y = np.asarray(y, dtype=np.float64)

    def get_data(self, run):
        if run != 0:
            raise IndexError
        return self._y


class _OneRunModel:
    def __init__(self, x, y):
        self._x = np.asarray(x, dtype=np.float64)
        self.dataset = _OneRunDataset(y)
        self.n_runs = 1

    def design_matrix_array(self, run=0):
        if run != 0:
            raise IndexError
        return self._x


class TestEngineRegistry:
    def test_builtin_engines_registered(self):
        engines = list_engines()
        assert "runwise" in engines
        assert "sketch" in engines
        assert "chunkwise" in engines
        assert "concat" in engines
        assert "reduced_rank" in engines
        assert "rrr_gls" in engines

    def test_get_runwise(self):
        eng = get_engine("runwise")
        assert hasattr(eng, "fit")
        assert hasattr(eng, "preflight")

    def test_get_sketch(self):
        eng = get_engine("sketch")
        assert hasattr(eng, "fit")

    def test_get_chunkwise(self):
        eng = get_engine("chunkwise")
        assert hasattr(eng, "fit")

    def test_get_reduced_rank_aliases(self):
        eng = get_engine("reduced_rank")
        alias = get_engine("rrr_gls")
        assert eng.name == "reduced_rank"
        assert alias.name == "reduced_rank"
        assert type(alias) is type(eng)

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError, match="Unknown engine"):
            get_engine("nonexistent_engine_xyz")

    def test_normalize_engine_name_is_builtin_only(self):
        assert _normalize_engine_name("runwise") == "runwise"
        assert _normalize_engine_name("chunkwise") == "chunkwise"
        assert _normalize_engine_name("concat") == "concat"
        assert _normalize_engine_name("sketch") == "sketch"
        assert _normalize_engine_name("reduced_rank") == "reduced_rank"
        assert _normalize_engine_name("rrr_gls") == "rrr_gls"

        with pytest.raises(KeyError, match="Unknown built-in engine"):
            _normalize_engine_name("plugin_engine")

    def test_resolve_legacy_builtin_selector_preserves_kwargs(self):
        eng, kwargs = resolve_engine("chunkwise", {"chunk_size": 9, "n_jobs": 2})

        assert eng.name == "chunkwise"
        assert kwargs == {"chunk_size": 9, "n_jobs": 2}

    def test_resolve_typed_runwise_options(self):
        eng, kwargs = resolve_engine(RunwiseEngineOptions(n_jobs=2, chunk_size=17))

        assert eng.name == "runwise"
        assert kwargs["n_jobs"] == 2
        assert kwargs["chunk_size"] == 17

    def test_resolve_typed_chunkwise_options(self):
        eng, kwargs = resolve_engine(
            ChunkwiseEngineOptions(chunk_size=13, n_jobs=2, blas_threads=1)
        )

        assert eng.name == "chunkwise"
        assert kwargs["chunk_size"] == 13
        assert kwargs["n_jobs"] == 2
        assert kwargs["blas_threads"] == 1

    def test_resolve_typed_reduced_rank_options(self):
        eng, kwargs = resolve_engine(
            ReducedRankEngineOptions(rank=2, target="all", bootstrap_seed=10)
        )

        assert eng.name == "reduced_rank"
        assert kwargs["rank"] == 2
        assert kwargs["target"] == "all"
        assert kwargs["bootstrap_seed"] == 10

    def test_typed_engine_options_reject_legacy_kwargs(self):
        with pytest.raises(ValueError, match="typed engine options"):
            resolve_engine(ChunkwiseEngineOptions(), {"chunk_size": 13})

    def test_typed_engine_options_validate_at_construction(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkwiseEngineOptions(chunk_size=0)
        with pytest.raises(ValueError, match="sketch_ratio"):
            SketchEngineOptions(sketch_ratio=0.0)
        with pytest.raises(ValueError, match="rank"):
            ReducedRankEngineOptions(rank=0)
        with pytest.raises(ValueError, match="bootstrap_n"):
            ReducedRankEngineOptions(se_mode="bootstrap", bootstrap_n=1)

    def test_register_with_decorator(self):
        @register_engine
        class _TestEngine1:
            name = "test_engine_1"

            def fit(self, model, config, **kw):
                return EngineResult(
                    betas=np.zeros((1, 1)),
                    sigma=np.zeros(1),
                    dfres=1.0,
                    XtXinv=np.eye(1),
                )

            def preflight(self, model, config):
                pass

        try:
            assert "test_engine_1" in list_engines()
            eng = get_engine("test_engine_1")
            assert isinstance(eng, _TestEngine1)
        finally:
            _ENGINES.pop("test_engine_1", None)

    def test_register_with_name_kwarg(self):
        @register_engine(name="test_engine_2")
        class _TestEngine2:
            def fit(self, model, config, **kw):
                return EngineResult(
                    betas=np.zeros((1, 1)),
                    sigma=np.zeros(1),
                    dfres=1.0,
                    XtXinv=np.eye(1),
                )

            def preflight(self, model, config):
                pass

        try:
            assert "test_engine_2" in list_engines()
        finally:
            _ENGINES.pop("test_engine_2", None)

    def test_register_no_name_raises(self):
        with pytest.raises(ValueError, match="no 'name' attribute"):
            @register_engine
            class _NoName:
                def fit(self, model, config, **kw):
                    pass
                def preflight(self, model, config):
                    pass


class TestEngineResult:
    def test_basic_construction(self):
        er = EngineResult(
            betas=np.ones((3, 10)),
            sigma=np.ones(10),
            dfres=97.0,
            XtXinv=np.eye(3),
        )
        assert er.betas.shape == (3, 10)
        assert er.dfres == 97.0
        assert er.projections is None
        assert er.extra == {}

    def test_with_extras(self):
        er = EngineResult(
            betas=np.zeros((2, 5)),
            sigma=np.zeros(5),
            dfres=10.0,
            XtXinv=np.eye(2),
            extra={"my_info": 42},
        )
        assert er.extra["my_info"] == 42


class TestRunwiseEngine:
    """Integration test: RunwiseEngine via get_engine."""

    def test_protocol_satisfaction(self):
        eng = get_engine("runwise")
        assert isinstance(eng, FittingEngine)

    def test_preflight_no_dataset_raises(self):
        eng = get_engine("runwise")
        with pytest.raises(ValueError, match="dataset"):
            eng.preflight(object(), FmriLmConfig())


class TestSketchEngine:
    def test_protocol_satisfaction(self):
        eng = get_engine("sketch")
        assert isinstance(eng, FittingEngine)

    def test_preflight_no_dataset_raises(self):
        eng = get_engine("sketch")
        with pytest.raises(ValueError, match="dataset"):
            eng.preflight(object(), FmriLmConfig())

    def test_legacy_method_and_m_drive_srht_sketch_config(self):
        rng = np.random.default_rng(44)
        n, p, v = 90, 4, 6
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        Y = rng.standard_normal((n, v))
        eng = get_engine("sketch")

        result = eng.fit(
            _OneRunModel(X, Y),
            FmriLmConfig(),
            method="srht",
            m=50,
            seed=3,
        )

        cfg = result.extra["lowrank_config"]
        assert cfg.sketch_kind == "srht"
        assert cfg.sketch_size == 50


class TestReducedRankEngine:
    def test_protocol_satisfaction(self):
        eng = get_engine("reduced_rank")
        assert isinstance(eng, FittingEngine)

    def test_preflight_no_dataset_raises(self):
        eng = get_engine("reduced_rank")
        with pytest.raises(ValueError, match="dataset"):
            eng.preflight(object(), FmriLmConfig())
