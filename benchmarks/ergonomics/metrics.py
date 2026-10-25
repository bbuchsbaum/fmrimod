"""Measurement for the ergonomics scenarios.

Three families of numbers, all computed from the scenario bodies themselves
rather than asserted by hand:

* **authoring cost** — statements, tokens, call sites, keyword arguments and
  imports in the function body (docstring, comments and ``# parity-knob``
  lines excluded). Tokens and statements are formatting-independent, so a
  long line wrapped by the formatter costs nothing extra.
* **runtime** — warm wall time (best of ``repeats`` after one warm-up call,
  the ``timeit`` convention: the minimum is the least noise-contaminated
  estimate on a shared machine) and the first call in the process, plus how
  many warnings the library printed at the user while doing the work.
* **documentation** — every library callable the body actually invoked,
  captured with a profiler hook, checked for a docstring, a ``Parameters``
  section and an ``Examples`` section.
"""

from __future__ import annotations

import ast
import gc
import inspect
import io
import sys
import textwrap
import time
import tokenize
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from types import CodeType, FunctionType
from typing import Any

PARITY_KNOB = "# parity-knob"
LIBRARY_PREFIXES = {"fmrimod": ("fmrimod", "fmridataset"), "nilearn": ("nilearn",)}
_COUNTED_TOKENS = {tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING}


@dataclass(frozen=True)
class AuthoringCost:
    """Static size of what the user had to write."""

    statements: int
    tokens: int
    lines: int
    calls: int
    keyword_args: int
    imports: int
    imported_names: tuple[str, ...]
    parity_knobs: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocEntry:
    """Documentation status of one library callable the user invoked."""

    qualname: str
    has_docstring: bool
    has_parameters: bool
    has_examples: bool


@dataclass
class RuntimeProfile:
    """Wall time, warnings and the callables touched in one scenario run."""

    first_call_seconds: float
    warm_seconds: float
    warm_samples: list[float]
    warnings_emitted: int
    warning_messages: list[str]
    docs: list[DocEntry] = field(default_factory=list)
    peak_mb: float | None = None


def _body_source(func: Callable[..., Any]) -> tuple[str, ast.FunctionDef]:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        raise TypeError(f"{func!r} is not a plain function")
    return source, node


def _excluded_lines(source: str, node: ast.FunctionDef) -> set[int]:
    """Lines that are signature, docstring, comment-only or parity knobs."""
    lines = source.splitlines()
    excluded = set(range(1, node.body[0].lineno))
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        excluded.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    for number, text in enumerate(lines, start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith("#") or stripped.endswith(PARITY_KNOB):
            excluded.add(number)
    return excluded


def authoring_cost(func: Callable[..., Any]) -> AuthoringCost:
    """Count what a reader of ``func``'s body has to take in."""
    source, node = _body_source(func)
    excluded = _excluded_lines(source, node)
    lines = source.splitlines()

    tokens = 0
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _COUNTED_TOKENS and tok.start[0] not in excluded:
            tokens += 1

    def counted(n: ast.AST) -> bool:
        return getattr(n, "lineno", 0) not in excluded

    statements = calls = keywords = imports = 0
    imported: list[str] = []
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, ast.stmt) and counted(child):
            statements += 1
        if isinstance(child, ast.Call) and counted(child):
            calls += 1
        if (
            isinstance(child, ast.keyword)
            and child.value is not None
            and counted(child.value)
        ):
            keywords += 1
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            imports += 1
            imported.extend(alias.asname or alias.name for alias in child.names)

    knobs = sum(1 for text in lines if text.rstrip().endswith(PARITY_KNOB))
    physical = sum(1 for i in range(1, len(lines) + 1) if i not in excluded)
    return AuthoringCost(
        statements=statements,
        tokens=tokens,
        lines=physical,
        calls=calls,
        keyword_args=keywords,
        imports=imports,
        imported_names=tuple(imported),
        parity_knobs=knobs,
    )


def _has_section(doc: str, name: str) -> bool:
    lines = [line.strip() for line in doc.splitlines()]
    for i, line in enumerate(lines[:-1]):
        if (
            line.lower() in {name.lower(), f"{name.lower()}:"}
            and set(lines[i + 1]) <= {"-", "="}
            and lines[i + 1]
        ):
            return True
    return (
        f"{name}:" in doc
        or f".. rubric:: {name}" in doc
        or (name == "Examples" and ">>>" in doc)
    )


def _function_for_code(code: CodeType) -> Any:
    for ref in gc.get_referrers(code):
        if isinstance(ref, FunctionType) and ref.__code__ is code:
            return ref
    return None


class _CallRecorder:
    """Profiler hook: record library callables invoked directly by user code."""

    def __init__(self, user_code: CodeType, prefixes: tuple[str, ...]) -> None:
        self.user_code = user_code
        self.prefixes = prefixes
        self.seen: dict[str, Any] = {}

    def _in_library(self, module: str | None) -> bool:
        return bool(module) and module.split(".")[0] in self.prefixes

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        if (
            event != "call"
            or frame.f_back is None
            or frame.f_back.f_code is not self.user_code
        ):
            return
        code = frame.f_code
        owner = frame.f_locals.get("self")
        if owner is not None and code.co_name == "__init__":
            cls = type(owner)
            if self._in_library(cls.__module__):
                self.seen.setdefault(f"{cls.__module__}.{cls.__qualname__}", cls)
            return
        if not self._in_library(frame.f_globals.get("__name__")):
            return
        if code.co_name.startswith("_") or code.co_name == "<module>":
            return
        if owner is not None:
            target = getattr(type(owner), code.co_name, None)
            target = getattr(target, "fget", target)
            name = f"{type(owner).__module__}.{type(owner).__qualname__}.{code.co_name}"
        else:
            target = _function_for_code(code)
            name = f"{frame.f_globals.get('__name__')}.{getattr(code, 'co_qualname', code.co_name)}"
        if target is not None:
            self.seen.setdefault(name, target)


def _doc_entry(name: str, obj: Any) -> DocEntry:
    doc = inspect.getdoc(obj) or ""
    try:
        params = [
            p for p in inspect.signature(obj).parameters if p not in {"self", "cls"}
        ]
    except (TypeError, ValueError):
        params = []
    return DocEntry(
        qualname=name,
        has_docstring=bool(doc.strip()),
        has_parameters=(not params) or _has_section(doc, "Parameters"),
        has_examples=_has_section(doc, "Examples"),
    )


def profile_runtime(
    func: Callable[[Any], Any],
    inputs: Any,
    library: str,
    *,
    repeats: int = 3,
    measure_memory: bool = False,
) -> tuple[Any, RuntimeProfile]:
    """Run ``func(inputs)`` once traced, then ``repeats`` times for timing.

    With ``measure_memory`` one extra call runs under :mod:`tracemalloc`
    (NumPy reports its buffers to it) and ``peak_mb`` records the largest
    amount allocated during the call, inputs excluded.
    """
    recorder = _CallRecorder(func.__code__, LIBRARY_PREFIXES[library])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        result = func(inputs)
        first = time.perf_counter() - start
    messages = sorted({str(w.message).splitlines()[0][:160] for w in caught})

    previous = sys.getprofile()
    sys.setprofile(recorder)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func(inputs)
    finally:
        sys.setprofile(previous)

    samples: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            func(inputs)
            samples.append(time.perf_counter() - start)

    peak_mb = None
    if measure_memory:
        import gc
        import tracemalloc

        gc.collect()
        tracemalloc.start()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                func(inputs)
            peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
        finally:
            tracemalloc.stop()

    docs = [_doc_entry(name, obj) for name, obj in sorted(recorder.seen.items())]
    return result, RuntimeProfile(
        first_call_seconds=first,
        warm_seconds=min(samples),
        warm_samples=samples,
        warnings_emitted=len(caught),
        warning_messages=messages,
        docs=docs,
        peak_mb=peak_mb,
    )
