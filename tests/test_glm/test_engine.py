"""Tests for the pluggable engine registry."""

import numpy as np
import pytest

from fmrimod.glm.engine import (
    _ENGINES,
    ChunkwiseEngineOptions,
    EngineResult,
    FittingEngine,
    RunwiseEngineOptions,
    SketchEngineOptions,
    _normalize_engine_name,
    get_engine,
    list_engines,
    register_engine,
    resolve_engine,
)
from fmrimod.model.config import FmriLmConfig


class TestEngineRegistry:
    def test_builtin_engines_registered(self):
        engines = list_engines()
        assert "runwise" in engines
        assert "sketch" in engines
        assert "chunkwise" in engines

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

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError, match="Unknown engine"):
            get_engine("nonexistent_engine_xyz")

    def test_normalize_engine_name_is_builtin_only(self):
        assert _normalize_engine_name("runwise") == "runwise"
        assert _normalize_engine_name("chunkwise") == "chunkwise"
        assert _normalize_engine_name("sketch") == "sketch"

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

    def test_typed_engine_options_reject_legacy_kwargs(self):
        with pytest.raises(ValueError, match="typed engine options"):
            resolve_engine(ChunkwiseEngineOptions(), {"chunk_size": 13})

    def test_typed_engine_options_validate_at_construction(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkwiseEngineOptions(chunk_size=0)
        with pytest.raises(ValueError, match="sketch_ratio"):
            SketchEngineOptions(sketch_ratio=0.0)

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
