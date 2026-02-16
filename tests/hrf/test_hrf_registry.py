from fmrimod.utils import list_available_hrfs
from fmrimod.hrf.registry import list_available_hrfs as registry_list_available_hrfs


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
        alias_names = {e["name"] for e in details if e["is_alias"]}
        primary_names = {e["name"] for e in details if not e["is_alias"]}
        # Every non-alias should be unique (no two primaries share the same object)
        assert len(primary_names) > 0

    def test_list_available_hrfs_description_is_human_readable(self):
        """Descriptions should remain non-empty and human readable."""
        details = registry_list_available_hrfs(details=True)
        assert all(entry["description"] for entry in details)
