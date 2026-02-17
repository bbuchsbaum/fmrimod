"""Tests for the pluggable engine registry."""

import numpy as np
import pytest

from fmrimod.glm.engine import (
    EngineResult,
    FittingEngine,
    get_engine,
    list_engines,
    register_engine,
    _ENGINES,
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
