"""Report rendering for parity harness results."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from cross_testing.harness.compare import ParityResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return value


def render(result: ParityResult, out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for one parity result."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "parity_report.json"
    md_path = out_dir / "REPORT.md"

    payload = _json_safe(result)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Parity Report: {result.name}",
        "",
        f"Status: `{result.status}`",
        "",
        "## Array Deltas",
        "",
        "| quantity | shape | scale | max_abs | mae | pearson_r | spearman_rho | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for delta in result.deltas.values():
        lines.append(
            "| {name} | {shape} | {scale:.6g} | {max_abs:.6g} | {mae:.6g} | "
            "{pearson:.6g} | {spearman:.6g} | {passes} |".format(
                name=delta.name,
                shape=delta.shape,
                scale=delta.scale,
                max_abs=delta.max_abs,
                mae=delta.mae,
                pearson=delta.pearson_r,
                spearman=delta.spearman_rho,
                passes="yes" if delta.passes else "no",
            )
        )

    if result.caveats:
        lines.extend(["", "## Caveats", ""])
        for caveat in result.caveats:
            lines.append(
                f"- `{caveat.caveat_id}` ({caveat.quantity}): "
                f"{caveat.reason} Expected: {caveat.expected}. Link: {caveat.link}"
            )

    if result.column_alignment:
        lines.extend(["", "## Column Alignment", ""])
        for col in result.column_alignment:
            caveat = f" caveat={col.caveat_id}" if col.caveat_id else ""
            lines.append(f"- `{col.candidate}` -> `{col.reference}`{caveat}")

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path
