"""Opt-in BIDS Stats Model translator generalization parity case.

The synthetic ``test_stats_model.py`` cases pin exact in-memory model
dicts so the default suite stays fast. This module adds the *opt-in*
counterpart from bd-01KRFKYRP9ERR436SA4MYM4HAH: run the translator
against a real on-disk BIDS-layout dataset (discover the BIDS Stats
Model JSON + a run's ``events.tsv`` + TR, then translate and assert
the result is structurally coherent — "translator generalization").

Two entry points share one harness:

* ``test_real_bids_dataset_translator_parity`` — gated behind the
  ``FMRIMOD_BIDS_PARITY_DATASET`` env var so CI / the fast CLI never
  needs real data; point it at a BIDS dataset root that carries a
  Stats Model JSON to exercise true external data.
* ``test_real_bids_layout_translator_generalizes`` — synthesizes a
  minimal but real BIDS-*layout* dataset in ``tmp_path`` so the
  discovery+translate harness itself is exercised every run rather
  than merely skipped.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.bids import translate_run_node
from fmrimod.bids.stats_model import load_stats_model

_ENV = "FMRIMOD_BIDS_PARITY_DATASET"


def _discover_stats_model(root: Path) -> Path:
    """Return the first BIDS Stats Model JSON under ``root``."""
    for pattern in (
        "models/*_smdl.json",
        "models/*.json",
        "*_smdl.json",
        "**/model-*_smdl.json",
        "**/*_smdl.json",
    ):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no BIDS Stats Model JSON under {root}")


def _discover_run(root: Path) -> tuple[pd.DataFrame, float, int]:
    """Return ``(events, tr, n_scans)`` for the first run under ``root``.

    TR comes from the matching ``*_bold.json`` sidecar (``RepetitionTime``)
    when present. ``n_scans`` is taken from a sidecar ``"dim4"`` /
    ``"NumberOfVolumes"`` hint if available, otherwise derived from the
    event timeline plus an HRF tail — the same fallback a real harness
    uses when the BOLD image is not loaded.
    """
    ev_paths = sorted(root.glob("sub-*/**/func/*_events.tsv")) or sorted(
        root.glob("**/*_events.tsv")
    )
    if not ev_paths:
        raise FileNotFoundError(f"no *_events.tsv under {root}")
    ev_path = ev_paths[0]
    events = pd.read_csv(ev_path, sep="\t")

    sidecar = ev_path.with_name(
        ev_path.name.replace("_events.tsv", "_bold.json")
    )
    tr = 2.0
    n_scans: int | None = None
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        tr = float(meta.get("RepetitionTime", tr))
        for key in ("NumberOfVolumes", "dim4"):
            if isinstance(meta.get(key), (int, float)):
                n_scans = int(meta[key])
                break

    if n_scans is None:
        last = float(
            (events["onset"] + events.get("duration", 0.0)).max()
        )
        n_scans = int(math.ceil((last + 24.0) / tr))
    return events, tr, max(n_scans, 2)


def _run_translator_parity(root: Path) -> None:
    """Translate the dataset's run node and assert structural coherence."""
    model = load_stats_model(_discover_stats_model(root))
    events, tr, n_scans = _discover_run(root)
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=tr)

    translated = translate_run_node(
        model, events=events, sampling_frame=sampling_frame
    )

    design = np.column_stack(
        [
            translated.event_model.design_matrix,
            translated.baseline_model.design_matrix,
        ]
    )
    assert design.shape[0] == n_scans
    assert np.all(np.isfinite(design))
    assert translated.column_names, "translator produced no design columns"
    for name, vec in translated.contrast_vectors.items():
        assert vec.shape == (len(translated.column_names),), (
            f"contrast {name!r} weight vector {vec.shape} does not match "
            f"{len(translated.column_names)} design columns"
        )


def _write_minimal_bids_dataset(root: Path) -> None:
    """Write a minimal but real BIDS-*layout* dataset under ``root``."""
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "fmrimod-parity-mini", "BIDSVersion": "1.8.0"})
    )
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "model-mini_smdl.json").write_text(
        json.dumps(
            {
                "Name": "mini",
                "BIDSModelVersion": "1.0.0",
                "Nodes": [
                    {
                        "Level": "run",
                        "Transformations": [
                            {"Name": "Factor", "Input": ["trial_type"]},
                            {
                                "Name": "Convolve",
                                "Input": [
                                    "trial_type.word",
                                    "trial_type.pseudoword",
                                ],
                                "Model": "spm",
                            },
                        ],
                        "Model": {
                            "X": [
                                "trial_type.word",
                                "trial_type.pseudoword",
                                1,
                            ]
                        },
                        "Contrasts": [
                            {
                                "Name": "word_gt_pseudoword",
                                "ConditionList": [
                                    "trial_type.word",
                                    "trial_type.pseudoword",
                                ],
                                "Weights": [1, -1],
                                "Test": "t",
                            }
                        ],
                    }
                ],
            }
        )
    )
    func = root / "sub-01" / "func"
    func.mkdir(parents=True, exist_ok=True)
    base = "sub-01_task-lang_run-1"
    pd.DataFrame(
        {
            "onset": [4.0, 12.0, 20.0, 28.0, 36.0, 44.0],
            "duration": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "trial_type": [
                "word",
                "pseudoword",
                "word",
                "pseudoword",
                "word",
                "pseudoword",
            ],
        }
    ).to_csv(func / f"{base}_events.tsv", sep="\t", index=False)
    (func / f"{base}_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "NumberOfVolumes": 40})
    )


@pytest.mark.skipif(
    os.environ.get(_ENV) is None,
    reason=f"set {_ENV} to a real BIDS dataset root (with a Stats Model) to run",
)
def test_real_bids_dataset_translator_parity() -> None:
    """Translator generalizes to a real external BIDS dataset (opt-in)."""
    root = Path(os.environ[_ENV]).expanduser()
    if not root.is_dir():
        pytest.skip(f"{_ENV}={root!s} is not a directory")
    _run_translator_parity(root)


def test_real_bids_layout_translator_generalizes(tmp_path: Path) -> None:
    """The discovery+translate harness runs end-to-end on a real BIDS layout."""
    _write_minimal_bids_dataset(tmp_path)
    _run_translator_parity(tmp_path)
