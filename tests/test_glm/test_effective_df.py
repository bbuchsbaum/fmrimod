"""Wave 3: effective residual df is n - rank, not n - rank - ar_order."""

from __future__ import annotations

from fmrimod.glm.effective_df import effective_df


def test_effective_df_does_not_subtract_ar_order() -> None:
    """fmrireg 0.2.0 live path reports n - rank.

    Cheap pass: still subtracting ``ar_order`` (the dead helper's old
    approximation).
    """
    assert effective_df(100, rank=10) == 90.0
    assert effective_df(100, rank=10, ar_order=5) == 90.0
    assert effective_df(100, rank=10, ar_order=5) != 85.0
    assert effective_df(100, rank=10, soft_subspace_rank=2) == 88.0
    assert effective_df(5, rank=10) == 1.0
