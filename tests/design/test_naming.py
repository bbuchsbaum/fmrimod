"""Tests for naming utilities."""

import pytest
from fmrimod.naming import (
    zeropad,
    sanitize,
    sanitize_level,
    basis_suffix,
    feature_suffix,
    make_unique_tags,
    level_token,
    continuous_token,
    make_cond_tag,
    add_basis_suffix,
    make_column_names,
    make_unique_colnames,
    shortnames,
    longnames,
    translate_legacy_pattern,
)


class TestZeroPad:
    """Test zero padding function."""
    
    def test_basic_padding(self):
        """Test basic zero padding."""
        assert zeropad(1, 10) == "01"
        assert zeropad(10, 10) == "10"
        assert zeropad(1, 100) == "001"
        assert zeropad(100, 100) == "100"
        
    def test_single_item(self):
        """Test padding with single item."""
        assert zeropad(1, 1) == "1"
        
    def test_zero_total(self):
        """Test padding with zero total."""
        assert zeropad(1, 0) == "1"


class TestSanitize:
    """Test name sanitization."""
    
    def test_valid_names(self):
        """Test that valid names are unchanged."""
        assert sanitize("valid_name") == "valid_name"
        assert sanitize("ValidName123") == "ValidName123"
        assert sanitize("with.dot") == "with.dot"
        
    def test_invalid_chars(self):
        """Test replacement of invalid characters."""
        assert sanitize("name with spaces") == "name_with_spaces"
        assert sanitize("name-with-dashes") == "name_with_dashes"
        assert sanitize("name@symbol") == "name_symbol"
        
    def test_leading_digit(self):
        """Test handling of leading digits."""
        assert sanitize("123name") == "X123name"
        assert sanitize("_name") == "X_name"
        
    def test_no_dots(self):
        """Test dot replacement when not allowed."""
        assert sanitize("with.dot", allow_dot=False) == "with_dot"
        assert sanitize("multiple...dots", allow_dot=False) == "multiple_dots"
        
    def test_empty_result(self):
        """Test handling of empty result."""
        assert sanitize("") == "X"
        assert sanitize("@#$%") == "X____"


class TestSanitizeLevel:
    """Test factor level sanitization."""
    
    def test_numeric_levels(self):
        """Test numeric level handling."""
        assert sanitize_level("123") == "123"
        assert sanitize_level("01") == "01"
        
    def test_string_levels(self):
        """Test string level handling."""
        assert sanitize_level("level A") == "level_A"
        assert sanitize_level("level-B") == "level_B"


class TestBasisSuffix:
    """Test basis suffix generation."""
    
    def test_single_basis(self):
        """Test suffix for single basis function."""
        assert basis_suffix(1, 1) == "_b1"
        
    def test_multiple_basis(self):
        """Test suffix for multiple basis functions."""
        assert basis_suffix(1, 10) == "_b01"
        assert basis_suffix(10, 10) == "_b10"
        assert basis_suffix(1, 100) == "_b001"
        assert basis_suffix(100, 100) == "_b100"


class TestFeatureSuffix:
    """Test feature suffix generation."""
    
    def test_single_feature(self):
        """Test suffix for single feature."""
        assert feature_suffix(1, 1) == "f1"
        
    def test_multiple_features(self):
        """Test suffix for multiple features."""
        assert feature_suffix(1, 10) == "f01"
        assert feature_suffix(10, 10) == "f10"
        assert feature_suffix(1, 100) == "f001"
        assert feature_suffix(100, 100) == "f100"


class TestMakeUniqueTags:
    """Test unique tag generation."""
    
    def test_no_duplicates(self):
        """Test with no duplicate tags."""
        tags = ["tag1", "tag2", "tag3"]
        assert make_unique_tags(tags) == tags
        
    def test_with_duplicates(self):
        """Test with duplicate tags."""
        tags = ["tag", "tag", "other", "tag"]
        assert make_unique_tags(tags) == ["tag", "tag#1", "other", "tag#2"]


class TestLevelToken:
    """Test factor level token creation."""
    
    def test_basic_token(self):
        """Test basic level token."""
        assert level_token("condition", "A") == "condition.A"
        assert level_token("group", "control") == "group.control"
        
    def test_sanitization(self):
        """Test that names are sanitized."""
        assert level_token("my var", "level 1") == "my_var.level_1"


class TestContinuousToken:
    """Test continuous variable token."""
    
    def test_basic_token(self):
        """Test basic continuous token."""
        assert continuous_token("response_time") == "response_time"
        assert continuous_token("RT.ms") == "RT.ms"
        
    def test_sanitization(self):
        """Test that names are sanitized."""
        assert continuous_token("response time") == "response_time"


class TestMakeCondTag:
    """Test condition tag creation."""
    
    def test_single_token(self):
        """Test single token."""
        assert make_cond_tag(["token"]) == "token"
        
    def test_multiple_tokens(self):
        """Test multiple tokens."""
        assert make_cond_tag(["cond", "A", "resp", "fast"]) == "cond_A_resp_fast"


class TestAddBasisSuffix:
    """Test adding basis suffixes."""
    
    def test_single_basis(self):
        """Test with single basis function."""
        tags = ["cond_A", "cond_B"]
        assert add_basis_suffix(tags, 1) == tags
        
    def test_multiple_basis(self):
        """Test with multiple basis functions.

        Condition-major ordering matches the categorical convolver's
        per-level hstack output, so the column name at index k describes
        the realised column at index k.
        """
        tags = ["cond_A", "cond_B"]
        result = add_basis_suffix(tags, 2)
        assert result == [
            "cond_A_b01", "cond_A_b02",
            "cond_B_b01", "cond_B_b02"
        ]


class TestMakeColumnNames:
    """Test column name generation."""
    
    def test_identity_term(self):
        """Test identity term (no term tag)."""
        assert make_column_names(None, ["intercept"]) == ["intercept"]
        assert make_column_names(None, ["block1", "block2"]) == ["block1", "block2"]
        
    def test_with_term_tag(self):
        """Test with term tag."""
        assert make_column_names("cond", ["A", "B"]) == ["cond_A", "cond_B"]
        
    def test_with_basis(self):
        """Test with basis functions.

        Condition-major: within each condition the basis suffixes appear
        in order before moving on to the next condition. Matches the
        realised column order produced by the categorical convolver.
        """
        result = make_column_names("poly_rt", ["f1", "f2"], nb=2)
        assert result == [
            "poly_rt_f1_b01", "poly_rt_f1_b02",
            "poly_rt_f2_b01", "poly_rt_f2_b02"
        ]


class TestMakeUniqueColnames:
    """Test unique column name generation."""
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        names = ["col1", "col2", "col3"]
        assert make_unique_colnames(names) == names
        
    def test_with_duplicates(self):
        """Test with duplicates."""
        names = ["col", "col", "other", "col"]
        assert make_unique_colnames(names) == ["col", "col.1", "other", "col.2"]


class TestShortnames:
    """Test short name generation."""
    
    def test_basic_shortening(self):
        """Test basic shortening."""
        long = ["condition_A", "condition_B", "response_time"]
        short = shortnames(long)
        # Should extract first letters or abbreviate
        assert len(short) == len(long)
        assert all(len(s) <= len(l) for s, l in zip(short, long))
        
    def test_with_acronym(self):
        """Test with provided acronym."""
        long = ["condition_A", "condition_B", "response_time"]
        short = shortnames(long, acronym="C")
        # Should use provided acronym
        assert any("C" in s for s in short)
        
    def test_empty_list(self):
        """Test with empty list."""
        assert shortnames([]) == []


class TestLongnames:
    """Test longnames function."""

    def test_longnames_alias(self):
        """Test that longnames is an alias for make_column_names."""
        # Should produce same results as make_column_names
        assert longnames("term", ["A", "B"]) == make_column_names("term", ["A", "B"])
        assert longnames(None, ["intercept"]) == make_column_names(None, ["intercept"])
        assert longnames("poly", ["f1"], nb=3) == make_column_names("poly", ["f1"], nb=3)


class TestTranslateLegacyPattern:
    """Test legacy pattern translation."""

    def test_translate_legacy_pattern_bracket(self):
        """Test bracket notation translation."""
        assert translate_legacy_pattern("condition[A]") == "condition.A"

    def test_translate_legacy_pattern_basis(self):
        """Test basis notation translation."""
        assert translate_legacy_pattern("term:basis[2]") == "term_b2"
        assert translate_legacy_pattern("term:basis[2]$") == "term_b2$"

    def test_translate_legacy_pattern_colon(self):
        """Test colon to underscore translation."""
        assert translate_legacy_pattern("fac1:fac2") == "fac1_fac2"

    def test_translate_legacy_pattern_combined(self):
        """Test combined transformations."""
        assert translate_legacy_pattern("fac1:fac2[level]") == "fac1_fac2.level"

    def test_translate_legacy_pattern_empty(self):
        """Test empty string."""
        assert translate_legacy_pattern("") == ""

    def test_translate_legacy_pattern_no_change(self):
        """Test string that needs no changes."""
        assert translate_legacy_pattern("simple_name") == "simple_name"

    def test_translate_legacy_pattern_multiple_brackets(self):
        """Test multiple bracket notations."""
        assert translate_legacy_pattern("A[x]:B[y]") == "A.x_B.y"
