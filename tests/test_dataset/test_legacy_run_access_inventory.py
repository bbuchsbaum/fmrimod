"""Inventory temporary positional ``dataset.get_data(run)`` callers.

The dataset consolidation PRD makes rows/cols ``get_data(...)`` and explicit
``get_run_data(...)`` the canonical public contract.  A few production modeling
paths still call ``dataset.get_data(run)`` while the latent/study dataset
semantics are being settled.  This test makes that migration debt explicit so
new call sites cannot appear silently.
"""

from __future__ import annotations

import ast
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PositionalGetDataCall:
    path: str
    qualname: str
    receiver: str


@dataclass(frozen=True)
class LegacyRunAccessOwner:
    owner: str
    reason: str


class _GetDataVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stack: list[str] = []
        self.calls: list[PositionalGetDataCall] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_data"
            and node.args
        ):
            receiver = ast.unparse(node.func.value)
            self.calls.append(
                PositionalGetDataCall(
                    path=str(self.path.relative_to(REPO_ROOT)),
                    qualname=".".join(self.stack),
                    receiver=receiver,
                )
            )
        self.generic_visit(node)


def _production_positional_get_data_calls() -> Counter[PositionalGetDataCall]:
    calls: Counter[PositionalGetDataCall] = Counter()
    for path in sorted((REPO_ROOT / "fmrimod").rglob("*.py")):
        visitor = _GetDataVisitor(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(path.read_text())
        visitor.visit(tree)
        calls.update(visitor.calls)
    return calls


CANONICAL_RUN_ACCESS_BRIDGE = Counter(
    {
        PositionalGetDataCall(
            path="fmrimod/dataset/fmri_dataset.py",
            qualname="FmriDataset.get_run_data",
            receiver="self._source",
        ): 1,
    }
)


TOLERATED_LEGACY_MODEL_CALLERS: dict[
    PositionalGetDataCall, tuple[int, LegacyRunAccessOwner]
] = {
    PositionalGetDataCall(
        path="fmrimod/ar/integration.py",
        qualname="iterative_gls",
        receiver="model.dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="AR integration still consumes model datasets through the"
            " temporary run-positional protocol.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/ar/integration.py",
        qualname="iterative_ar_gls",
        receiver="model.dataset",
    ): (
        3,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Iterative AR GLS has three run-level accesses to migrate"
            " after canonical latent/study semantics settle.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/glm/engines/sketch.py",
        qualname="SketchEngine._fit_single",
        receiver="model.dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Sketch engine single-run fit still uses the compatibility"
            " protocol.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/glm/engines/sketch.py",
        qualname="SketchEngine._fit_multirun",
        receiver="model.dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Sketch engine multi-run fit still uses the compatibility"
            " protocol.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/glm/strategies.py",
        qualname="_fit_one_run",
        receiver="model.dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Core GLM strategy run access is migration-owned by the"
            " modeling integration bead.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/glm/strategies.py",
        qualname="fit_runwise",
        receiver="dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Runwise matrix assembly still calls a dataset-like object"
            " positionally for the one-run case.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/glm/strategies.py",
        qualname="fit_runwise",
        receiver="model.dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Runwise multi-run fitting still uses the compatibility"
            " protocol.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/glm/strategies.py",
        qualname="fit_chunkwise._fit_run",
        receiver="model.dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Chunkwise fitting still fetches each run through the"
            " temporary positional protocol.",
        ),
    ),
    PositionalGetDataCall(
        path="fmrimod/robust/irls.py",
        qualname="robust_refit",
        receiver="model.dataset",
    ): (
        1,
        LegacyRunAccessOwner(
            owner="bd-01KRFMW8P73FFC4A16M3N6NNQ9",
            reason="Robust refit is model-facing migration debt, not a"
            " dataset contract.",
        ),
    ),
}


def test_positional_get_data_run_inventory_is_explicit() -> None:
    expected = CANONICAL_RUN_ACCESS_BRIDGE + Counter(
        {
            call: count
            for call, (count, _owner) in TOLERATED_LEGACY_MODEL_CALLERS.items()
        }
    )

    assert _production_positional_get_data_calls() == expected


def test_tolerated_legacy_callers_name_owner_and_reason() -> None:
    assert TOLERATED_LEGACY_MODEL_CALLERS
    for _call, (_count, owner) in TOLERATED_LEGACY_MODEL_CALLERS.items():
        assert owner.owner == "bd-01KRFMW8P73FFC4A16M3N6NNQ9"
        assert owner.reason
