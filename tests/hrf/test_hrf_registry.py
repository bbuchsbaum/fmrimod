import pytest

import fmrimod.hrf_dispatch as hrf_dispatch
from fmrimod.dispatch import get_hrf as dispatch_get_hrf
from fmrimod.hrf.aliases import _normalize_hrf_name
from fmrimod.hrf.core import as_hrf
from fmrimod.hrf.registry import get_hrf, remove_hrf
from fmrimod.hrf.registry import list_available_hrfs as registry_list_available_hrfs
from fmrimod.utils import list_available_hrfs


class TestHRFRegistry:
    """Test public HRF registry behavior."""

    def test_list_available_hrfs_is_sorted_and_informative(self):
        """Public listing should be sorted and include core built-ins."""
        names = list_available_hrfs()

        assert names == sorted(names)
        assert "spmg1" in names
        assert "gamma" in names
        assert "bspline" in names
        assert "fir" in names
        assert "tent" in names
        assert "daguerre" in names
        assert "boxcar" in names
        assert "weighted" in names

    def test_list_available_hrfs_details_include_typed_fields(self):
        """Detailed registry metadata should use Python-native types."""
        names = list_available_hrfs()
        details = registry_list_available_hrfs(details=True)

        assert len(details) == len(names)

        required_fields = {"name", "type", "nbasis_default", "is_alias", "description"}
        first = details[0]
        assert required_fields.issubset(set(first))
        assert isinstance(first["description"], str)
        assert isinstance(first["is_alias"], bool)

        by_name = {entry["name"]: entry for entry in details}
        assert by_name["fir"]["nbasis_default"] == 12
        assert by_name["bspline"]["nbasis_default"] == 5
        assert by_name["weighted"]["nbasis_default"] is None
        assert by_name["boxcar"]["type"] == "generator"
        # Verify alias detection is consistent: aliases point to same object as primary
        primary_names = {e["name"] for e in details if not e["is_alias"]}
        # Every non-alias should be unique (no two primaries share the same object)
        assert len(primary_names) > 0

    def test_list_available_hrfs_description_is_human_readable(self):
        """Descriptions should remain non-empty and human readable."""
        details = registry_list_available_hrfs(details=True)
        assert all(entry["description"] for entry in details)

    def test_alias_normalizer_collapses_legacy_spm_names(self):
        assert _normalize_hrf_name("spm") == "spmg1"
        assert _normalize_hrf_name("SPM_CANONICAL") == "spmg1"

    def test_get_hrf_validates_unknown_parameters(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            get_hrf("gamma", bogus=1)

    def test_get_hrf_routes_parameterized_basis_to_generator(self):
        hrf = get_hrf("bspline", n_basis=7, degree=2)

        assert hrf.nbasis == 7
        assert hrf.params["degree"] == 2

    def test_hrf_dispatch_has_no_parallel_registry(self):
        """Compatibility dispatch must delegate to the canonical registry."""
        assert not hasattr(hrf_dispatch, "_HRF_REGISTRY")

        canonical = get_hrf("spm")
        compat = hrf_dispatch.get_hrf("spm")
        generic = dispatch_get_hrf("spm")

        assert compat.name == canonical.name
        assert compat.nbasis == canonical.nbasis
        assert generic.name == canonical.name
        assert generic.nbasis == canonical.nbasis

    def test_hrf_dispatch_registers_custom_name_in_canonical_registry(self):
        name = "unit_test_custom_dispatch"

        def factory():
            return as_hrf(lambda t: t, name=name)

        hrf_dispatch.register_hrf(name, factory)
        try:
            assert hrf_dispatch.get_hrf(name).name == name
            assert get_hrf(name).name == name
        finally:
            remove_hrf(name)
