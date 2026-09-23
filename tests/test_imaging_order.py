"""
The imaging order (imaging.order) decides which channel is imaged first, and it is the same
order everywhere channels are taken one after another: a protocol run and a progressive snap.

- wavelength_order: the wavelength in the channel name, highest first; brightfield last
- z_order: lowest z offset first
- protocol_order: as listed
- a value that is none of these is refused, not treated as one of them
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_protocol_geometry import _channel, make_protocol  # noqa: F401 (fixture)

from seafront.config.basics import sort_channels_by_imaging_order
from seafront.config.handles import ImagingConfig
from seafront.config.registry import ConfigRegistry


def named(name: str, z_offset_um: float = 0.0):
    channel = _channel(handle=name.lower().replace(" ", "_"), z_offset_um=z_offset_um)
    return channel.model_copy(update={"name": name})


CHANNELS = [
    named("DAPI 405 nm"),
    named("BF LED matrix full"),
    named("Cy5 638 nm"),
    named("GFP 488 nm"),
]


def names(channels) -> list[str]:
    return [c.name for c in channels]


def test_wavelength_order_is_longest_wavelength_first_and_brightfield_last():
    assert names(sort_channels_by_imaging_order(CHANNELS, "wavelength_order")) == [
        "Cy5 638 nm",
        "GFP 488 nm",
        "DAPI 405 nm",
        "BF LED matrix full",
    ]


def test_z_order_is_lowest_offset_first():
    channels = [named("a", 5.0), named("b", -2.0), named("c", 0.0)]
    assert names(sort_channels_by_imaging_order(channels, "z_order")) == ["b", "c", "a"]


def test_protocol_order_keeps_the_list():
    assert names(sort_channels_by_imaging_order(CHANNELS, "protocol_order")) == names(CHANNELS)


def test_an_unknown_order_is_refused():
    with pytest.raises(ValueError, match="sideways"):
        sort_channels_by_imaging_order(CHANNELS, "sideways")  # type: ignore[arg-type]


def test_a_run_images_channels_in_the_same_order(make_protocol):  # noqa: F811
    ConfigRegistry.set_value(ImagingConfig.ORDER, "wavelength_order")
    protocol = make_protocol(channels=CHANNELS)
    assert [info.channel.name for info in protocol.iter_channels()] == names(
        sort_channels_by_imaging_order(CHANNELS, "wavelength_order")
    )
