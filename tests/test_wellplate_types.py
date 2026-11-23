"""
The plate types the server offers come from seaconfig. A plate added there reaches the UI only
once the seaconfig pin in pyproject.toml is moved to a release that has it.
"""


def test_the_ibidi_6_well_plate_is_offered(mock_core):
    offered = {p.Model_id for p in mock_core.get_hardware_capabilities().wellplate_types}
    assert "ibidi-6-80637" in offered
