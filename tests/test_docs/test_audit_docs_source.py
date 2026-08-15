from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from scripts.audit_docs_source import audit


def _write(tutorials: Path, name: str, source: str) -> Path:
    tutorials.mkdir(parents=True, exist_ok=True)
    path = tutorials / name
    path.write_text(dedent(source), encoding="utf-8")
    return path


def _rules(tutorials: Path) -> set[str]:
    return {finding.rule for finding in audit(tutorials)}


def test_current_tutorial_sources_pass() -> None:
    assert audit(Path("docs/tutorials")) == []


def test_neuroim_and_exact_fiac_imports_are_allowed(tmp_path: Path) -> None:
    tutorials = tmp_path / "tutorials"
    _write(
        tutorials,
        "real-data-fiac.qmd",
        """
        ```{python}
        import neuroim as ni
        from nilearn.datasets import fetch_fiac_first_level
        from nilearn.plotting import plot_design_matrix

        image = ni.read_vec("run.nii.gz")
        ```
        """,
    )
    assert audit(tutorials) == []


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("import nibabel as nib", "nibabel-import"),
        ("from nilearn.image import smooth_img", "nilearn-import"),
        ("from nilearn.maskers import NiftiMasker", "nilearn-import"),
        ("data = image.get_fdata()", "manual-spatial-boundary"),
        ("image = Nifti1Image(data, affine)", "manual-spatial-boundary"),
        ("masker = NiftiMasker()", "nilearn-masker"),
        (
            "dataset = fmri_dataset(signals, mask=np.ones(signals.shape[1]))",
            "fabricated-mask",
        ),
        (
            "from numpy import ones as full_mask\n"
            "dataset = fmri_dataset(signals, mask=full_mask(signals.shape[1]))",
            "fabricated-mask",
        ),
        ("affine = np.eye(4)", "fabricated-affine"),
        (
            "import numpy as arraylib\naffine = arraylib.eye(4)",
            "fabricated-affine",
        ),
        ("volume = values.reshape(mask.shape)", "manual-reconstruction"),
    ],
)
def test_forbidden_spatial_shortcuts_fail(
    tmp_path: Path, source: str, rule: str
) -> None:
    tutorials = tmp_path / "tutorials"
    _write(tutorials, "bad.qmd", f"```{{python}}\n{source}\n```\n")
    assert rule in _rules(tutorials)


def test_fiac_imports_are_not_a_site_wide_nilearn_allowlist(tmp_path: Path) -> None:
    tutorials = tmp_path / "tutorials"
    _write(
        tutorials,
        "other.qmd",
        """
        ```{python}
        from nilearn.datasets import fetch_fiac_first_level
        ```
        """,
    )
    assert "nilearn-import" in _rules(tutorials)


def test_plain_python_and_eval_false_require_reasons(tmp_path: Path) -> None:
    tutorials = tmp_path / "tutorials"
    _write(tutorials, "plain.qmd", "```python\nanswer = 42\n```\n")
    _write(
        tutorials,
        "disabled.qmd",
        """
        ```{python}
        #| eval: false
        import optional_package
        ```
        """,
    )
    assert _rules(tutorials) == {
        "unlabeled-nonexecution",
        "unlabeled-python-fence",
    }


def test_adjacent_schematic_and_nonexecution_reasons_are_allowed(
    tmp_path: Path,
) -> None:
    tutorials = tmp_path / "tutorials"
    _write(
        tutorials,
        "labeled.qmd",
        """
        <!-- docs-schematic: algorithm-specific values are placeholders -->
        ```python
        result = EngineResult(beta=beta)
        ```

        <!-- docs-nonexecuted: optional legacy package is not a docs dependency -->
        ```{python}
        #| eval: false
        import legacy_package
        ```
        """,
    )
    assert audit(tutorials) == []
