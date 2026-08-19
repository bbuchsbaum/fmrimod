"""Wave 3 docs nits: stale summate wording and the precision split."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_convolve_docs_do_not_say_summate_false_takes_max() -> None:
    """fmrihrf 0.4.0: summate=False duration-averages; it does not take max.

    Cheap pass: leaving the stale 'max is used' wording in convolve.py.
    """
    text = (ROOT / "fmrimod" / "convolve.py").read_text()
    assert "max is used" not in text
    assert "take max" not in text
    assert "duration-averaged" in text or "duration-average" in text


def test_sampling_frame_documents_precision_split() -> None:
    """SamplingFrame.precision 0.1 vs regressor/R evaluate 0.33."""
    text = (ROOT / "fmrimod" / "sampling.py").read_text()
    assert "0.33" in text
    assert "0.1" in text
