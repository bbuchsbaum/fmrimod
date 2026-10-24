"""Head-to-head ergonomics and speed scorecard: fmrimod vs Nilearn.

Runs every scenario in :mod:`benchmarks.ergonomics.scenarios`, checks that
the two paths agree numerically (so the comparison is between equivalent
analyses), and scores four axes per scenario:

``authoring``  tokens in the body (statements reported alongside)
``speed``      warm wall time, best of N
``noise``      warnings the library emitted at the user
``docs``       docstring / Parameters / Examples coverage of the library
               callables the body actually invoked

The output is a ranked list of **deficiencies** — every axis where fmrimod
loses, plus the per-scenario friction notes — because the purpose of the
suite is to tell us what to fix next, not to certify a win.

Usage::

    .venv/bin/python -m benchmarks.ergonomics.workflow            # small inputs
    .venv/bin/python -m benchmarks.ergonomics.workflow --large    # + scaling rows
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.ergonomics.metrics import (
    AuthoringCost,
    RuntimeProfile,
    authoring_cost,
    profile_runtime,
)
from benchmarks.ergonomics.scenarios import SCENARIOS, Scenario

REPORT_DIR = Path(__file__).resolve().parent / "reports"
SCHEMA_VERSION = 1

# An axis is a tie unless one side is better by more than this margin.
TOKEN_MARGIN = 0.05
SPEED_MARGIN = 0.10
DOCS_MARGIN = 0.05


@dataclass
class SideResult:
    library: str
    authoring: AuthoringCost
    runtime: RuntimeProfile

    @property
    def docs_score(self) -> float:
        """Mean of (docstring, Parameters, Examples) over invoked callables."""
        docs = self.runtime.docs
        if not docs:
            return 1.0
        points = sum(d.has_docstring + d.has_parameters + d.has_examples for d in docs)
        return points / (3 * len(docs))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "library": self.library,
            "authoring": self.authoring.as_dict(),
            "runtime": {k: v for k, v in asdict(self.runtime).items() if k != "docs"},
            "docs": [asdict(d) for d in self.runtime.docs],
            "docs_score": self.docs_score,
        }
        return payload


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    size: str
    fmrimod: SideResult
    nilearn: SideResult
    correlation: float
    max_abs_delta: float
    relative_max_delta: float
    min_correlation: float
    verdicts: dict[str, str]
    notes: tuple[str, ...]

    @property
    def agrees(self) -> bool:
        return bool(
            np.isfinite(self.correlation) and self.correlation >= self.min_correlation
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "size": self.size,
            "agreement": {
                "correlation": self.correlation,
                "min_correlation": self.min_correlation,
                "max_abs_delta": self.max_abs_delta,
                "relative_max_delta": self.relative_max_delta,
                "agrees": self.agrees,
            },
            "verdicts": self.verdicts,
            "notes": list(self.notes),
            "fmrimod": self.fmrimod.as_dict(),
            "nilearn": self.nilearn.as_dict(),
        }


def _lower_is_better(fm_value: float, nl_value: float, margin: float) -> str:
    if fm_value <= nl_value * (1.0 - margin):
        return "fmrimod"
    if fm_value >= nl_value * (1.0 + margin):
        return "nilearn"
    return "tie"


def score(fm: SideResult, nl: SideResult, *, large: bool = False) -> dict[str, str]:
    """Per-axis winner: ``fmrimod``, ``nilearn`` or ``tie``.

    Large-input rows rerun the small rows' code, so only their speed and
    noise are scored; authoring and docs read ``n/a`` there.
    """
    fm_docs, nl_docs = fm.docs_score, nl.docs_score
    if fm_docs > nl_docs + DOCS_MARGIN:
        docs = "fmrimod"
    elif nl_docs > fm_docs + DOCS_MARGIN:
        docs = "nilearn"
    else:
        docs = "tie"
    fm_warn, nl_warn = fm.runtime.warnings_emitted, nl.runtime.warnings_emitted
    authoring = _lower_is_better(fm.authoring.tokens, nl.authoring.tokens, TOKEN_MARGIN)
    return {
        "authoring": "n/a" if large else authoring,
        "speed": _lower_is_better(
            fm.runtime.warm_seconds, nl.runtime.warm_seconds, SPEED_MARGIN
        ),
        "noise": "tie"
        if fm_warn == nl_warn
        else ("fmrimod" if fm_warn < nl_warn else "nilearn"),
        "docs": "n/a" if large else docs,
    }


def run_scenario(
    scenario: Scenario, *, repeats: int = 3, large: bool = False
) -> ScenarioResult:
    """Run both sides of one scenario and score them."""
    factory = scenario.large_inputs if large else scenario.make_inputs
    if factory is None:
        raise ValueError(f"{scenario.scenario_id} has no large inputs")
    inputs = factory()
    sides: dict[str, SideResult] = {}
    arrays: dict[str, np.ndarray] = {}
    for library in ("fmrimod", "nilearn"):
        func = getattr(scenario, library)
        result, runtime = profile_runtime(func, inputs, library, repeats=repeats)
        sides[library] = SideResult(library, authoring_cost(func), runtime)
        arrays[library] = getattr(scenario, f"extract_{library}")(result, inputs)

    fm_arr, nl_arr = arrays["fmrimod"], arrays["nilearn"]
    if fm_arr.shape != nl_arr.shape:
        raise ValueError(
            f"{scenario.scenario_id}: output shapes differ {fm_arr.shape} vs {nl_arr.shape}"
        )
    finite = np.isfinite(fm_arr) & np.isfinite(nl_arr)
    corr = float(np.corrcoef(fm_arr[finite], nl_arr[finite])[0, 1])
    max_abs = float(np.max(np.abs(fm_arr[finite] - nl_arr[finite])))
    scale = float(np.max(np.abs(nl_arr[finite]))) or 1.0
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        size="large" if large else "small",
        fmrimod=sides["fmrimod"],
        nilearn=sides["nilearn"],
        correlation=corr,
        max_abs_delta=max_abs,
        relative_max_delta=max_abs / scale,
        min_correlation=scenario.min_correlation,
        verdicts=score(sides["fmrimod"], sides["nilearn"], large=large),
        notes=scenario.notes,
    )


def deficiencies(results: list[ScenarioResult]) -> list[dict[str, Any]]:
    """Every place fmrimod loses, most actionable first.

    Authoring, friction and docs findings come from the small-input rows only
    (the large rows run the same code); speed and agreement come from all
    rows. Documentation gaps are grouped per callable.
    """
    found: list[dict[str, Any]] = []
    doc_gaps: dict[str, dict[str, Any]] = {}
    for r in results:
        fm, nl = r.fmrimod, r.nilearn
        tag = f"{r.scenario_id}[{r.size}]"
        if not r.agrees:
            found.append(
                {
                    "scenario": tag,
                    "axis": "agreement",
                    "detail": (
                        f"outputs disagree (r={r.correlation:.4f} < {r.min_correlation})"
                    ),
                }
            )
        if r.verdicts["speed"] == "nilearn":
            found.append(
                {
                    "scenario": tag,
                    "axis": "speed",
                    "detail": (
                        f"{fm.runtime.warm_seconds * 1e3:.1f} ms vs Nilearn "
                        f"{nl.runtime.warm_seconds * 1e3:.1f} ms"
                    ),
                }
            )
        if fm.runtime.warnings_emitted:
            found.append(
                {
                    "scenario": tag,
                    "axis": "noise",
                    "detail": (
                        f"{fm.runtime.warnings_emitted} warning(s): "
                        + "; ".join(fm.runtime.warning_messages[:3])
                    ),
                }
            )
        if r.size != "small":
            continue
        if r.verdicts["authoring"] == "nilearn":
            found.append(
                {
                    "scenario": tag,
                    "axis": "authoring",
                    "detail": (
                        f"{fm.authoring.tokens} tokens / {fm.authoring.statements} statements vs "
                        f"Nilearn {nl.authoring.tokens} / {nl.authoring.statements}"
                    ),
                }
            )
        for note in r.notes:
            found.append({"scenario": tag, "axis": "friction", "detail": note})
        for d in fm.runtime.docs:
            missing = [
                label
                for label, ok in (
                    ("docstring", d.has_docstring),
                    ("Parameters", d.has_parameters),
                    ("Examples", d.has_examples),
                )
                if not ok
            ]
            if missing:
                gap = doc_gaps.setdefault(
                    d.qualname, {"missing": missing, "scenarios": []}
                )
                gap["scenarios"].append(r.scenario_id)
    for qualname, gap in sorted(
        doc_gaps.items(), key=lambda kv: -len(kv[1]["scenarios"])
    ):
        found.append(
            {
                "scenario": f"{len(gap['scenarios'])} scenario(s)",
                "axis": "docs",
                "detail": (f"`{qualname}` lacks {' + '.join(gap['missing'])}"),
            }
        )
    order = {
        "agreement": 0,
        "speed": 1,
        "authoring": 2,
        "friction": 3,
        "noise": 4,
        "docs": 5,
    }
    return sorted(found, key=lambda d: order[d["axis"]])


def summarize(results: list[ScenarioResult]) -> dict[str, dict[str, int]]:
    tally: dict[str, dict[str, int]] = {}
    for r in results:
        for axis, winner in r.verdicts.items():
            if winner != "n/a":
                tally.setdefault(axis, {"fmrimod": 0, "nilearn": 0, "tie": 0})[
                    winner
                ] += 1
    fm_tokens = sum(r.fmrimod.authoring.tokens for r in results if r.size == "small")
    nl_tokens = sum(r.nilearn.authoring.tokens for r in results if r.size == "small")
    tally["total_tokens"] = {"fmrimod": fm_tokens, "nilearn": nl_tokens}
    return tally


def build_report(results: list[ScenarioResult]) -> dict[str, Any]:
    import nilearn

    import fmrimod

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "ergonomics_vs_nilearn",
        "versions": {
            "fmrimod": getattr(fmrimod, "__version__", "unknown"),
            "nilearn": nilearn.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "machine": platform.machine(),
        "margins": {"tokens": TOKEN_MARGIN, "speed": SPEED_MARGIN, "docs": DOCS_MARGIN},
        "summary": summarize(results),
        "scenarios": [r.as_dict() for r in results],
        "deficiencies": deficiencies(results),
    }


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.1f}"


def render_progress(report: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Markdown lines comparing fmrimod's numbers with an earlier round."""
    before = {
        (s["scenario_id"], s["size"]): s["fmrimod"] for s in baseline["scenarios"]
    }
    out = [
        "",
        f"## fmrimod since {baseline.get('round', 'the previous round')}",
        "",
        "Nilearn's bodies are unchanged between rounds; only fmrimod's moved.",
        "",
        "| scenario | size | tokens before → after | stmts before → after | ms before → after |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in report["scenarios"]:
        prev = before.get((s["scenario_id"], s["size"]))
        if prev is None:
            continue
        now = s["fmrimod"]
        out.append(
            f"| {s['scenario_id']} | {s['size']} "
            f"| {prev['authoring']['tokens']} → {now['authoring']['tokens']} "
            f"| {prev['authoring']['statements']} → {now['authoring']['statements']} "
            f"| {_ms(prev['runtime']['warm_seconds'])} → "
            f"{_ms(now['runtime']['warm_seconds'])} |"
        )
    prev_total = baseline["summary"]["total_tokens"]["fmrimod"]
    now_total = report["summary"]["total_tokens"]["fmrimod"]
    out.append("")
    out.append(
        f"Total fmrimod tokens: {prev_total} → {now_total} "
        f"(Nilearn {report['summary']['total_tokens']['nilearn']})."
    )
    return out


def render_markdown(
    report: dict[str, Any], baseline: dict[str, Any] | None = None
) -> str:
    """Human-readable scorecard, with a before/after section when given a baseline."""
    out = [
        "# fmrimod vs Nilearn: ergonomics & speed scorecard",
        "",
        "Generated by `python -m benchmarks.ergonomics.workflow`. Do not edit by hand.",
        "",
        f"Versions: fmrimod {report['versions']['fmrimod']}, "
        f"nilearn {report['versions']['nilearn']}, numpy {report['versions']['numpy']}, "
        f"Python {report['versions']['python']} ({report['machine']}).",
        "",
        "Tokens/statements count the scenario body only (imports included; "
        "docstrings, comments and `# parity-knob` lines excluded). Time is the "
        "best of N warm calls. *Docs* scores the library callables each body actually "
        "invoked: one point each for a docstring, a Parameters section and an "
        "Examples section. It does not see gallery examples, which is where "
        "Nilearn keeps most of its usage documentation.",
        "",
        "| scenario | size | tokens fm / nl | stmts fm / nl | ms fm / nl | warnings fm / nl | docs fm / nl | r | winner (auth, speed, noise, docs) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report["scenarios"]:
        fm, nl = s["fmrimod"], s["nilearn"]
        v = s["verdicts"]
        out.append(
            f"| {s['scenario_id']} | {s['size']} "
            f"| {fm['authoring']['tokens']} / {nl['authoring']['tokens']} "
            f"| {fm['authoring']['statements']} / {nl['authoring']['statements']} "
            f"| {_ms(fm['runtime']['warm_seconds'])} / {_ms(nl['runtime']['warm_seconds'])} "
            f"| {fm['runtime']['warnings_emitted']} / {nl['runtime']['warnings_emitted']} "
            f"| {fm['docs_score']:.0%} / {nl['docs_score']:.0%} "
            f"| {s['agreement']['correlation']:.4f} "
            f"| {v['authoring']}, {v['speed']}, {v['noise']}, {v['docs']} |"
        )
    summary = report["summary"]
    out += ["", "## Tally", ""]
    for axis in ("authoring", "speed", "noise", "docs"):
        t = summary.get(axis, {})
        out.append(
            f"- **{axis}**: fmrimod {t.get('fmrimod', 0)}, nilearn {t.get('nilearn', 0)}, tie {t.get('tie', 0)}"
        )
    tt = summary["total_tokens"]
    out.append(
        f"- **total tokens (small inputs)**: fmrimod {tt['fmrimod']}, nilearn {tt['nilearn']}"
    )
    if baseline is not None:
        out += render_progress(report, baseline)
    out += ["", "## Deficiencies (where fmrimod should improve)", ""]
    if not report["deficiencies"]:
        out.append("None found.")
    for d in report["deficiencies"]:
        out.append(f"- `{d['scenario']}` **{d['axis']}** — {d['detail']}")
    out.append("")
    return "\n".join(out)


def run(
    *, repeats: int = 3, large: bool = False, only: list[str] | None = None
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for scenario in SCENARIOS:
        if only and scenario.scenario_id not in only:
            continue
        results.append(run_scenario(scenario, repeats=repeats))
        if large and scenario.large_inputs is not None:
            results.append(run_scenario(scenario, repeats=repeats, large=True))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--large", action="store_true", help="add scaling rows on larger images"
    )
    parser.add_argument(
        "--scenario", action="append", help="run only these scenario ids"
    )
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPORT_DIR / "round_4_report.json",
        help="earlier report to show fmrimod's before/after against",
    )
    args = parser.parse_args(argv)

    results = run(repeats=args.repeats, large=args.large, only=args.scenario)
    report = build_report(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ergonomics_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    baseline = (
        json.loads(args.baseline.read_text()) if args.baseline.is_file() else None
    )
    if baseline is not None:
        baseline.setdefault("round", args.baseline.stem.replace("_report", ""))
    markdown = render_markdown(report, baseline)
    (args.out / "REPORT.md").write_text(markdown)
    print(markdown)
    return 0 if all(r.agrees for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
