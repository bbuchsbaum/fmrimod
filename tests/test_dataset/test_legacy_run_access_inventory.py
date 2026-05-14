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
        PositionalGetDataCall(
            path="fmrimod/dataset/data_access.py",
            qualname="get_run_data",
            receiver="dataset",
        ): 1,
    }
)


TOLERATED_LEGACY_MODEL_CALLERS: dict[
    PositionalGetDataCall, tuple[int, LegacyRunAccessOwner]
] = {}


def test_positional_get_data_run_inventory_is_explicit() -> None:
    expected = CANONICAL_RUN_ACCESS_BRIDGE + Counter(
        {
            call: count
            for call, (count, _owner) in TOLERATED_LEGACY_MODEL_CALLERS.items()
        }
    )

    assert _production_positional_get_data_calls() == expected


def test_tolerated_legacy_callers_name_owner_and_reason() -> None:
    assert not TOLERATED_LEGACY_MODEL_CALLERS
    for _call, (_count, owner) in TOLERATED_LEGACY_MODEL_CALLERS.items():
        assert owner.owner == "bd-01KRFMW8P73FFC4A16M3N6NNQ9"
        assert owner.reason
