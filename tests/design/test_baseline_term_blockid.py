"""BaselineTerm block subsetting (inventory v2 DES-17, R fmridesign 95042d4).

Each requested block contributes its active columns once: the union of the
blocks' columns, in the term's own column order. A term shared by all blocks
(a global intercept) is not repeated per block, and a term without block
structure cannot be subset by block.
"""

import numpy as np
import pytest

from fmrimod.baseline import baseline_model
from fmrimod.baseline.baseline_term import BaselineTerm
from fmrimod.design.design_matrix import design_matrix
from fmrimod.sampling import SamplingFrame


def _model(**kwargs):
    sf = SamplingFrame(blocklens=[10, 10, 10], tr=2.0)
    return baseline_model(sframe=sf, basis="poly", degree=1, **kwargs)


def test_global_intercept_is_returned_once_for_several_blocks():
    term = _model(intercept="global").terms["block"]
    sub = term.get_block_matrix([1, 2, 3])
    assert list(sub.columns) == ["constant_global"]
    assert sub.shape == (30, 1)
    sub2 = term.get_block_matrix([3, 1])
    assert sub2.shape == (20, 1)


def test_runwise_drift_unions_block_columns_in_term_order():
    term = _model(intercept="global").terms["drift"]
    sub = term.get_block_matrix([3, 1])
    assert list(sub.columns) == ["base_poly1_block_1", "base_poly1_block_3"]
    assert not sub.columns.duplicated().any()


def test_design_matrix_blockid_has_no_duplicate_columns():
    bm = _model(intercept="global")
    X = np.asarray(design_matrix(bm, blockid=[1, 2, 3]))
    assert X.shape == (30, 4)
    np.testing.assert_array_equal(X, np.asarray(bm.design_matrix))


def test_term_without_block_structure_cannot_be_subset():
    term = BaselineTerm("plain", np.ones((4, 2)), colind=[], rowind=[])
    assert term.get_block_matrix().shape == (4, 2)
    with pytest.raises(ValueError, match="no block structure"):
        term.get_block_matrix(1)
