"""Published Z travel remains in motor coordinates when stage calibration changes."""

from seafront.config.handles import CalibrationConfig
from seafront.config.registry import ConfigRegistry
from seafront.hardware.limits import HardwareLimits


def test_capabilities_publish_motor_travel_without_applying_calibration_twice(mock_core):
    limits = mock_core.get_hardware_capabilities().hardware_limits
    assert limits.stage_z_travel_mm is not None
    assert limits.stage_z_travel_mm.min == 0
    assert limits.stage_z_travel_mm.max == 6.7

    ConfigRegistry.set_value(CalibrationConfig.OFFSET_Z_MM, 0.5)
    assert mock_core.get_hardware_capabilities().hardware_limits.stage_z_travel_mm == limits.stage_z_travel_mm


def test_adapters_without_z_travel_can_still_publish_other_capabilities(mock_scope):
    existing = mock_scope.get_hardware_limits().model_dump(exclude={"stage_z_travel_mm"})
    assert HardwareLimits.model_validate(existing).stage_z_travel_mm is None
