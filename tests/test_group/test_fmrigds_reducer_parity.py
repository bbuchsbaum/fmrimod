"""R-oracle parity tests for native :mod:`fmrimod.group` reducers.

These tests compare the native eager reducers against the explicit
``fmrigds-r`` oracle backend. They auto-skip when R/fmrigds is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.group import group_dataset_from_group_data, reduce
from fmrimod.stats import GroupFitRequest, fmrigds_backend_available, group_fit


def _candidate_fmrigds_source() -> str | None:
    env = os.environ.get("FMRIGDS_SOURCE_DIR", "").strip()
    if env and Path(env).exists():
        return env
    candidate = Path.home() / "code" / "fmrigds"
    if candidate.exists():
        return str(candidate)
    return None


def _skip_if_no_fmrigds() -> dict[str, str]:
    source = _candidate_fmrigds_source()
    ok, reason = fmrigds_backend_available(fmrigds_source=source)
    if not ok:
        pytest.skip(f"fmrigds unavailable: {reason}")
    opts: dict[str, str] = {}
    if source is not None:
        opts["fmrigds_source"] = source
    return opts


def _make_csv_group_data() -> Any:
    rows: list[dict[str, Any]] = []
    for roi, offset in (("r1", 0.0), ("r2", 0.08)):
        for subject, beta, se in (
            ("s1", 0.20, 0.10),
            ("s2", 0.10, 0.20),
            ("s3", 0.30, 0.10),
            ("s4", 0.25, 0.10),
            ("s5", 0.15, 0.15),
        ):
            rows.append(
                {
                    "subject": subject,
                    "roi": roi,
                    "beta": beta + offset,
                    "se": se,
                }
            )
    df = pd.DataFrame(rows)
    return group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
    )


@pytest.mark.cross_test
@pytest.mark.parametrize(
    ("native_method", "effects", "tau2"),
    [
        ("meta:fe", "fixed", "pm"),
        ("meta:re", "random", "dl"),
    ],
)
def test_native_meta_reducers_match_fmrigds_oracle(
    native_method: str,
    effects: Literal["fixed", "random"],
    tau2: Literal["pm", "dl"],
) -> None:
    backend_opts = _skip_if_no_fmrigds()
    gd = _make_csv_group_data()
    ds = group_dataset_from_group_data(gd)

    native = reduce(ds, method=native_method)
    oracle = group_fit(
        GroupFitRequest(
            data=gd,
            model="meta",
            effects=effects,
            tau2=tau2,
            backend="fmrigds-r",
            backend_options=backend_opts,
        )
    )

    np.testing.assert_allclose(
        native.assay("beta_g")[:, 0, 0], oracle.estimate[:, 0], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        native.assay("se_g")[:, 0, 0], oracle.se[:, 0], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        native.assay("z_g")[:, 0, 0], oracle.statistic[:, 0], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        native.assay("p_g")[:, 0, 0], oracle.p[:, 0], rtol=1e-6, atol=1e-8
    )
    if native_method == "meta:re":
        assert oracle.tau2 is not None
        np.testing.assert_allclose(
            native.assay("tau2")[:, 0, 0],
            np.asarray(oracle.tau2, dtype=np.float64).ravel(),
            rtol=1e-6,
            atol=1e-8,
        )
