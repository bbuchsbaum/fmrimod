"""A successful notebook run must actually publish its declared figures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "audit_docs_artifact",
    Path(__file__).resolve().parents[2] / "scripts/audit_docs_artifact.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_declared_plot_cannot_disappear_from_rendered_page(tmp_path):
    docs = tmp_path / "docs"
    artifact = docs / "_site"
    artifact.mkdir(parents=True)
    (docs / "index.qmd").write_text(
        '```{python}\n#| fig-alt: "Model diagnostic"\nfig = model.plot()\n```\n'
    )
    html = (
        '<title>Example</title><a href="#quarto-document-content">Skip</a>'
        '<main id="quarto-document-content">{content}</main>'
    )
    page = artifact / "index.html"
    page.write_text(html.format(content=""))
    assert any("declared plot outputs" in finding for finding in MODULE.audit(artifact))
    page.write_text(html.format(content='<img src="model.png" alt="Model diagnostic">'))
    assert MODULE.audit(artifact) == []


def test_empty_artifact_cannot_pass(tmp_path):
    assert MODULE.audit(tmp_path) == ["No HTML pages found in the documentation artifact"]
