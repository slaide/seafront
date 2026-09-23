"""
The protocol list, and the files in it that cannot be read.

A broken protocol file is still a file the operator saved. It must stay visible in the list,
with the reason it cannot be loaded, rather than drop out as though it had been deleted - and
it must not take the readable ones down with it.
"""

import pytest

from seafront.config.basics import GlobalConfigHandler
from seafront.config.default_protocol import generate_default_protocol


@pytest.fixture
def protocol_dir(mock_core, temp_config, monkeypatch):
    "the protocol directory, inside the test's tmp dir rather than the real ~/seafront"
    tmp_path, _ = temp_config
    monkeypatch.setattr(GlobalConfigHandler, "_seafront_home", tmp_path)
    generate_default_protocol("test-mock-microscope")
    return GlobalConfigHandler.home_acquisition_config_dir()


def test_unreadable_protocols_are_listed_with_the_reason(mock_core, protocol_dir):
    (protocol_dir / "broken-syntax.json").write_text("{ this is not json5")
    (protocol_dir / "wrong-shape.json").write_text('{"project_name": 3}')

    listing = mock_core.get_config_list()

    unreadable = {entry.filename: entry.error for entry in listing.unreadable}
    assert set(unreadable) == {"broken-syntax.json", "wrong-shape.json"}
    assert all(error for error in unreadable.values())
    assert "ValidationError" in unreadable["wrong-shape.json"]

    # the readable one is still there
    assert [entry.filename for entry in listing.configs] == ["default.json"]


def test_no_unreadable_protocols_means_an_empty_list(mock_core, protocol_dir):
    listing = mock_core.get_config_list()
    assert listing.unreadable == []
    assert [entry.filename for entry in listing.configs] == ["default.json"]
