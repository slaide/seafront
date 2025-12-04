"""
Mock microscope implementation for testing and development.

This module provides a MockMicroscope class that behaves like a real microscope
but doesn't communicate with any actual hardware.

Features:
- Realistic movement simulation at 4cm/s speed based on actual distance
- Realistic imaging delays (exposure time + 10ms overhead)
- Concurrent usage protection (prevents operations during streaming)
- Synthetic image generation with proper exposure time and gain scaling
- Channel-specific image patterns (brightfield vs fluorescence)
- Loading position simulation at coordinates (0, 0, 0)

Environment Variables:
- MOCK_NO_DELAYS=1: Disable realistic delays for faster testing (default: delays enabled)
"""

import asyncio
import os
import time
import typing as tp
from contextlib import contextmanager

import json5
import numpy as np
import seaconfig as sc
from pydantic import PrivateAttr

from seafront.config.basics import ChannelConfig, ConfigItem, FilterConfig, GlobalConfigHandler, ImagingOrder
from seafront.config.handles import ImagingConfig, LaserAutofocusConfig
from seafront.hardware.adapter import AdapterState, Position
from seafront.hardware.camera import Camera, HardwareLimitValue, MockCamera
from seafront.hardware.firmware_config import (
    get_firmware_config,
)
from seafront.hardware.illumination import IlluminationController
from seafront.hardware.microscope import HardwareLimits, Microscope, microscope_exclusive
from seafront.hardware.squid import Locked
from seafront.logger import logger
from seafront.server import commands as cmd

from seafront.config.handles import ProtocolConfig
from seafront.hardware.forbidden_areas import ForbiddenAreaList

class MockMicroscope(Microscope):
    """
    Mock microscope implementation for testing and development.

    This implementation:
    - Simulates all hardware operations without actual hardware
    - Returns instantly from all operations
    - Generates synthetic image data
    - Maintains realistic state transitions
    """

    illumination_controller: IlluminationController
    main_camera: Locked[Camera]

    # Mock hardware state
    _current_position: Position = PrivateAttr(default_factory=lambda: Position(x_pos_mm=0.0, y_pos_mm=0.0, z_pos_mm=0.0))
    _is_homed: bool = PrivateAttr(default=False)
    _streaming_channel: ChannelConfig | None = PrivateAttr(default=None)
    _streaming_acquisition_config: sc.AcquisitionChannelConfig | None = PrivateAttr(default=None)
    _latest_images: dict[str, cmd.ImageStoreEntry] = PrivateAttr(default_factory=dict)

    # Per-axis movement parameters - loaded from firmware profile
    _max_velocity_x_mm_s: float = PrivateAttr()
    _max_velocity_y_mm_s: float = PrivateAttr()
    _max_velocity_z_mm_s: float = PrivateAttr()
    _max_velocity_w_mm_s: float = PrivateAttr()

    _acceleration_x_mm_s2: float = PrivateAttr()
    _acceleration_y_mm_s2: float = PrivateAttr()
    _acceleration_z_mm_s2: float = PrivateAttr()
    _acceleration_w_mm_s2: float = PrivateAttr()

    _stabilization_time_x_s: float = PrivateAttr()
    _stabilization_time_y_s: float = PrivateAttr()
    _stabilization_time_z_s: float = PrivateAttr()
    _stabilization_time_w_s: float = PrivateAttr()

    def model_post_init(self, __context: tp.Any) -> None:
        """Initialize movement parameters from firmware profile"""
        # Load firmware configuration based on SEAFRONT_FIRMWARE_PROFILE environment variable
        firmware_config = get_firmware_config()

        # Set movement parameters from firmware profile
        self._max_velocity_x_mm_s = firmware_config.MAX_VELOCITY_X_mm
        self._max_velocity_y_mm_s = firmware_config.MAX_VELOCITY_Y_mm
        self._max_velocity_z_mm_s = firmware_config.MAX_VELOCITY_Z_mm
        self._max_velocity_w_mm_s = firmware_config.MAX_VELOCITY_W_mm

        self._acceleration_x_mm_s2 = firmware_config.MAX_ACCELERATION_X_mm
        self._acceleration_y_mm_s2 = firmware_config.MAX_ACCELERATION_Y_mm
        self._acceleration_z_mm_s2 = firmware_config.MAX_ACCELERATION_Z_mm
        self._acceleration_w_mm_s2 = firmware_config.MAX_ACCELERATION_W_mm

        self._stabilization_time_x_s = firmware_config.SCAN_STABILIZATION_TIME_MS_X / 1000.0
        self._stabilization_time_y_s = firmware_config.SCAN_STABILIZATION_TIME_MS_Y / 1000.0
        self._stabilization_time_z_s = firmware_config.SCAN_STABILIZATION_TIME_MS_Z / 1000.0
        # W axis doesn't have stabilization time in firmware config - use default 50ms
        self._stabilization_time_w_s = 0.05

        self.is_in_loading_position = False

        logger.info(f"Mock microscope initialized with firmware profile parameters - "
                   f"max velocities: X={self._max_velocity_x_mm_s}mm/s, Y={self._max_velocity_y_mm_s}mm/s, Z={self._max_velocity_z_mm_s}mm/s, W={self._max_velocity_w_mm_s}mm/s")

    @property
    def _realistic_delays_enabled(self) -> bool:
        """Check if realistic delays should be simulated (default: True, disable with MOCK_NO_DELAYS=1)"""
        return os.environ.get("MOCK_NO_DELAYS", "0").lower() not in ("1", "true", "yes")

    def is_position_forbidden(self, x_mm: float, y_mm: float) -> tuple[bool, str]:
        """
        Check if a position is forbidden for movement.

        Args:
            x_mm: X coordinate in mm
            y_mm: Y coordinate in mm

        Returns:
            Tuple of (is_forbidden, error_message). error_message is empty if position is allowed.
        """

        g_config = GlobalConfigHandler.get_dict()
        forbidden_areas_entry = g_config.get(ProtocolConfig.FORBIDDEN_AREAS.value)

        # If no forbidden areas config is found, allow the movement
        if forbidden_areas_entry is None:
            return False, ""

        forbidden_areas_str = forbidden_areas_entry.value
        if not isinstance(forbidden_areas_str, str):
            logger.warning("forbidden_areas entry is not a string, allowing movement")
            return False, ""

        data = json5.loads(forbidden_areas_str)
        forbidden_areas = ForbiddenAreaList.model_validate({"areas": data})

        # Check if position is forbidden
        is_forbidden, conflicting_area = forbidden_areas.is_position_forbidden(x_mm, y_mm)

        if is_forbidden and conflicting_area is not None:
            reason_text = f" ({conflicting_area.reason})" if conflicting_area.reason else ""
            error_msg = f"Movement to ({x_mm:.1f}, {y_mm:.1f}) mm is forbidden - conflicts with area '{conflicting_area.name}'{reason_text}"
            return True, error_msg

        return False, ""

    def _calculate_axis_movement_time(self, distance_mm: float, max_vel_mm_s: float, accel_mm_s2: float) -> tuple[float, float]:
        """
        Calculate movement time and max velocity reached for a single axis.
        
        Returns:
            (total_time_s, actual_max_vel_mm_s)
        """
        if abs(distance_mm) < 0.001:  # No movement
            return 0.0, 0.0

        distance = abs(distance_mm)

        # Time to reach max velocity
        accel_time = max_vel_mm_s / accel_mm_s2

        # Distance covered during acceleration
        accel_distance = 0.5 * accel_mm_s2 * accel_time**2

        # If movement is short, we never reach max velocity
        if distance <= 2 * accel_distance:
            # Triangle profile - accelerate then decelerate
            accel_time_actual = (distance / accel_mm_s2) ** 0.5
            total_time = 2 * accel_time_actual
            actual_max_vel = accel_mm_s2 * accel_time_actual
        else:
            # Trapezoid profile - accelerate, constant velocity, decelerate
            const_vel_distance = distance - 2 * accel_distance
            const_vel_time = const_vel_distance / max_vel_mm_s
            total_time = 2 * accel_time + const_vel_time
            actual_max_vel = max_vel_mm_s

        return total_time, actual_max_vel

    async def _simulate_gradual_movement(self, target_x_mm: float | None = None, target_y_mm: float | None = None, target_z_mm: float | None = None) -> None:
        """Simulate realistic movement with per-axis acceleration, max speed, and stabilization time"""
        # Get current real positions
        start_x = self._pos_x_measured_to_real(self._current_position.x_pos_mm)
        start_y = self._pos_y_measured_to_real(self._current_position.y_pos_mm)
        start_z = self._pos_z_measured_to_real(self._current_position.z_pos_mm)

        # Use current position if target not specified
        final_x = target_x_mm if target_x_mm is not None else start_x
        final_y = target_y_mm if target_y_mm is not None else start_y
        final_z = target_z_mm if target_z_mm is not None else start_z

        # If delays disabled, move instantly to final position
        if not self._realistic_delays_enabled:
            self._current_position.x_pos_mm = self._pos_x_real_to_measured(final_x)
            self._current_position.y_pos_mm = self._pos_y_real_to_measured(final_y)
            self._current_position.z_pos_mm = self._pos_z_real_to_measured(final_z)
            return

        # Calculate per-axis distances
        dx = final_x - start_x
        dy = final_y - start_y
        dz = final_z - start_z

        # Calculate movement time for each axis
        time_x, max_vel_x = self._calculate_axis_movement_time(dx, self._max_velocity_x_mm_s, self._acceleration_x_mm_s2)
        time_y, max_vel_y = self._calculate_axis_movement_time(dy, self._max_velocity_y_mm_s, self._acceleration_y_mm_s2)
        time_z, max_vel_z = self._calculate_axis_movement_time(dz, self._max_velocity_z_mm_s, self._acceleration_z_mm_s2)

        # Total movement time is the longest axis
        movement_time_s = max(time_x, time_y, time_z)

        # Determine stabilization time (max of moved axes)
        stabilization_time_s = 0.0
        if abs(dx) > 0.001:
            stabilization_time_s = max(stabilization_time_s, self._stabilization_time_x_s)
        if abs(dy) > 0.001:
            stabilization_time_s = max(stabilization_time_s, self._stabilization_time_y_s)
        if abs(dz) > 0.001:
            stabilization_time_s = max(stabilization_time_s, self._stabilization_time_z_s)

        total_distance_mm = (dx**2 + dy**2 + dz**2)**0.5
        logger.debug(f"Mock movement: {total_distance_mm:.1f}mm distance, {movement_time_s:.3f}s movement + {stabilization_time_s:.3f}s settling")
        logger.debug(f"  X: {dx:.1f}mm @ {max_vel_x:.1f}mm/s, Y: {dy:.1f}mm @ {max_vel_y:.1f}mm/s, Z: {dz:.1f}mm @ {max_vel_z:.1f}mm/s")

        if movement_time_s < 0.001:  # No significant movement
            return

        # Update position gradually during movement using simple linear interpolation
        # (Full trapezoidal profile would be complex - this gives realistic timing)
        update_interval_s = 0.01  # Update every 10ms for smooth movement
        num_steps = max(1, int(movement_time_s / update_interval_s))

        for step in range(num_steps + 1):
            # Calculate progress (0.0 to 1.0)
            progress = min((step + 1) / num_steps, 1.0) if num_steps > 0 else 1.0

            # Interpolate position for each axis
            current_x = start_x + dx * progress
            current_y = start_y + dy * progress
            current_z = start_z + dz * progress

            # Update measured position
            self._current_position.x_pos_mm = self._pos_x_real_to_measured(current_x)
            self._current_position.y_pos_mm = self._pos_y_real_to_measured(current_y)
            self._current_position.z_pos_mm = self._pos_z_real_to_measured(current_z)

            # Wait for next update (except last step)
            if step < num_steps:
                await asyncio.sleep(update_interval_s)

        # Ensure final position is exact
        self._current_position.x_pos_mm = self._pos_x_real_to_measured(final_x)
        self._current_position.y_pos_mm = self._pos_y_real_to_measured(final_y)
        self._current_position.z_pos_mm = self._pos_z_real_to_measured(final_z)

        # Wait for stabilization time
        if stabilization_time_s > 0:
            await asyncio.sleep(stabilization_time_s)

    async def _delay_for_imaging(self, exposure_time_ms: float) -> None:
        """Simulate realistic imaging delay (exposure time + overhead)"""
        if self._realistic_delays_enabled:
            # Imaging takes exposure time + 10ms overhead (readout, processing, etc.)
            total_time_s = (exposure_time_ms + 10.0) / 1000.0
            await asyncio.sleep(total_time_s)

    @contextmanager
    def lock(self, blocking: bool = True, reason: str = "unknown") -> tp.Iterator[tp.Self | None]:
        """Mock lock - always succeeds immediately."""
        if self._lock.acquire(blocking=blocking):
            self._lock_reasons.append(reason)
            try:
                yield self
            finally:
                self._lock_reasons.pop()
                self._lock.release()
        else:
            yield None

    @classmethod
    def make(cls) -> "MockMicroscope":
        """Create a mock microscope instance from configuration."""
        g_dict = GlobalConfigHandler.get_dict()

        logger.info("Creating mock microscope instance")

        # Parse channels from configuration
        channels_json = g_dict["imaging.channels"].strvalue
        channels_data = json5.loads(channels_json)
        if not isinstance(channels_data, list):
            raise ValueError("Invalid channels configuration: expected list")
        channel_configs = []
        for ch in channels_data:
            if not isinstance(ch, dict):
                raise ValueError("Each channel configuration must be a dictionary")
            channel_configs.append(ChannelConfig(**ch))

        # Parse filters from configuration (may not exist in global config)
        filter_configs = []
        if "filter.wheel.configuration" in g_dict:
            filters_json = g_dict["filter.wheel.configuration"].strvalue
            filters_data = json5.loads(filters_json)
            if not isinstance(filters_data, list):
                raise ValueError("Invalid filters configuration: expected list")
            for f in filters_data:
                if not isinstance(f, dict):
                    raise ValueError("Each filter configuration must be a dictionary")
                filter_configs.append(FilterConfig(**f))

        # Create illumination controller
        illumination_controller = IlluminationController(channel_configs)

        # Create mock camera
        mock_camera = MockCamera()

        return cls(
            channels=channel_configs,
            filters=filter_configs,
            illumination_controller=illumination_controller,
            main_camera=Locked(mock_camera),
        )

    @microscope_exclusive
    def open_connections(self) -> None:
        """Mock connection opening - always succeeds."""
        logger.info("Mock microscope: opening connections")
        self.main_camera.value.open("main")
        self.is_connected = True

    def close(self) -> None:
        """Mock connection closing."""
        logger.info("Mock microscope: closing connections")

        # Ensure streaming is stopped (safety measure - should be done via ChannelStreamEnd)
        self.main_camera.value.stop_streaming()

        # Clean up streaming state
        self._streaming_channel = None
        self._streaming_acquisition_config = None
        self.stream_callback = None
        self.is_connected = False

    async def home(self,x=True,y=True,z=True) -> None:
        """
        homing sequence.

        x,y,z: home in that axis
        """
        logger.info("Mock microscope: performing homing sequence")

        await self._simulate_gradual_movement(
            0 if x else None,
            0 if y else None,
            0 if z else None,
        )

        await self._simulate_gradual_movement(
            20 if x else None,
            20 if y else None,
            1 if z else None,
        )

        self._is_homed = True

        logger.info("Mock microscope: homing completed")

    async def get_current_state(self) -> AdapterState:
        """Get current mock microscope state."""
        # Apply calibration offsets to position
        calibrated_pos = Position(
            x_pos_mm=self._pos_x_measured_to_real(self._current_position.x_pos_mm),
            y_pos_mm=self._pos_y_measured_to_real(self._current_position.y_pos_mm),
            z_pos_mm=self._pos_z_measured_to_real(self._current_position.z_pos_mm)
        )

        self.last_state = AdapterState(
            stage_position=calibrated_pos,
            is_in_loading_position=self.is_in_loading_position,
        )

        return self.last_state

    @property
    def calibrated_stage_position(self) -> tuple[float, float, float]:
        """Get calibrated stage offset from configuration."""
        g_dict = GlobalConfigHandler.get_dict()

        x_offset = g_dict["calibration.offset.x_mm"].floatvalue
        y_offset = g_dict["calibration.offset.y_mm"].floatvalue
        z_offset = g_dict["calibration.offset.z_mm"].floatvalue

        return (x_offset, y_offset, z_offset)

    def _pos_x_measured_to_real(self, x_mm: float) -> float:
        """Convert measured X position to real position."""
        return x_mm + self.calibrated_stage_position[0]

    def _pos_y_measured_to_real(self, y_mm: float) -> float:
        """Convert measured Y position to real position."""
        return y_mm + self.calibrated_stage_position[1]

    def _pos_z_measured_to_real(self, z_mm: float) -> float:
        """Convert measured Z position to real position."""
        return z_mm + self.calibrated_stage_position[2]

    def _pos_x_real_to_measured(self, x_mm: float) -> float:
        """Convert real X position to measured position."""
        return x_mm - self.calibrated_stage_position[0]

    def _pos_y_real_to_measured(self, y_mm: float) -> float:
        """Convert real Y position to measured position."""
        return y_mm - self.calibrated_stage_position[1]

    def _pos_z_real_to_measured(self, z_mm: float) -> float:
        """Convert real Z position to measured position."""
        return z_mm - self.calibrated_stage_position[2]

    def _sort_channels_by_imaging_order(self, channels: list, imaging_order: ImagingOrder) -> list:
        """
        Sort channels according to the specified imaging order.
        
        Args:
            channels: List of enabled AcquisitionChannelConfig objects
            imaging_order: Sorting strategy to use
            
        Returns:
            Sorted list of channels
        """
        if imaging_order == "z_order":
            # Sort by z_offset_um (lowest to highest z coordinate)
            return sorted(channels, key=lambda ch: ch.z_offset_um)

        elif imaging_order == "wavelength_order":
            # Sort by wavelength (highest to lowest), then brightfield last
            def wavelength_sort_key(ch):
                # Extract wavelength from channel name if possible
                import re
                wavelength_match = re.search(r'(\d+)\s*nm', ch.name, re.IGNORECASE)
                if wavelength_match:
                    wavelength = int(wavelength_match.group(1))
                    # Higher wavelengths first (descending order)
                    return (0, -wavelength)
                elif 'brightfield' in ch.name.lower() or 'bf' in ch.name.lower():
                    # Brightfield comes last
                    return (1, 0)
                else:
                    # Unknown wavelength, put in middle
                    return (0, -500)  # Assume ~500nm for unknown

            return sorted(channels, key=wavelength_sort_key)

        elif imaging_order == "protocol_order":
            # Keep original order from config file
            return channels

        else:
            # Default to protocol order for unknown values
            logger.warning(f"Unknown imaging order '{imaging_order}', using protocol_order")
            return channels

    async def snap_selected_channels(self, config_file: sc.AcquisitionConfig) -> cmd.BasicSuccessResponse:
        """Mock channel snapping."""
        logger.info("Mock microscope: snapping selected channels")

        enabled_channels = [ch for ch in config_file.channels if ch.enabled]

        # Get imaging order from machine config and sort channels
        g_config = GlobalConfigHandler.get_dict()
        imaging_order = g_config.get(ImagingConfig.ORDER.value, "protocol_order")
        if isinstance(imaging_order, str):
            imaging_order_value = imaging_order
        else:
            imaging_order_value = imaging_order.strvalue if hasattr(imaging_order, 'strvalue') else "protocol_order"

        # Sort channels according to configured imaging order
        enabled_channels = self._sort_channels_by_imaging_order(enabled_channels, tp.cast(ImagingOrder, imaging_order_value))

        for channel in enabled_channels:
            # Generate synthetic image via camera
            synthetic_img = self.main_camera.value.snap(channel)

            # Create image store entry
            image_entry = cmd.ImageStoreEntry(
                pixel_format="mono16",
                info=cmd.ImageStoreInfo(
                    channel=channel,
                    width_px=synthetic_img.shape[1],
                    height_px=synthetic_img.shape[0],
                    timestamp=time.time(),
                    position=cmd.SitePosition(
                        well_name="mock_well",
                        site_x=0,
                        site_y=0,
                        site_z=0,
                        x_offset_mm=0.0,
                        y_offset_mm=0.0,
                        z_offset_mm=0.0,
                        position=self._current_position
                    ),
                    storage_path=None
                )
            )
            image_entry._img = synthetic_img

            self._latest_images[channel.handle] = image_entry

        logger.info(f"Mock microscope: snapped {len(enabled_channels)} channels")
        return cmd.BasicSuccessResponse()

    def get_hardware_limits(self) -> HardwareLimits:
        """
        Get mock hardware limits for testing.
        
        Returns realistic limits that simulate various hardware capabilities.
        """
        # Create mock limits as HardwareLimitValue objects for type safety
        exposure_limits = HardwareLimitValue(min=0.05, max=10000, step=0.01)  # 50 microseconds to 10 seconds
        gain_limits = HardwareLimitValue(min=-5, max=30, step=0.1)  # Typical scientific camera range
        focus_offset_limits = HardwareLimitValue(min=-500, max=500, step=0.05)  # Larger range for testing
        fluorescence_power_limits = HardwareLimitValue(min=5, max=100, step=0.1)  # Same as SQUID
        brightfield_power_limits = HardwareLimitValue(min=3, max=100, step=0.1)  # Same as SQUID
        generic_power_limits = HardwareLimitValue(min=5, max=100, step=0.1)  # Use fluorescence as default
        z_planes_limits = HardwareLimitValue(min=1, max=200, step=1)  # Smaller max for testing
        z_spacing_limits = HardwareLimitValue(min=0.01, max=2000, step=0.01)  # Higher precision for testing

        # Return properly typed HardwareLimits object
        return HardwareLimits(
            imaging_exposure_time_ms=exposure_limits.to_dict(),
            imaging_analog_gain_db=gain_limits.to_dict(),
            imaging_focus_offset_um=focus_offset_limits.to_dict(),
            imaging_illum_perc=generic_power_limits.to_dict(),
            imaging_illum_perc_fluorescence=fluorescence_power_limits.to_dict(),
            imaging_illum_perc_brightfield=brightfield_power_limits.to_dict(),
            imaging_number_z_planes=z_planes_limits.to_dict(),
            imaging_delta_z_um=z_spacing_limits.to_dict(),
        )

    def _validate_imaging_parameters(self, **params) -> None:
        """
        Validate imaging parameters against hardware limits.
        
        Args:
            **params: Imaging parameters to validate. Supported keys:
                - exposure_time_ms: float
                - analog_gain: float  
                - z_offset_um: float
                - illum_perc: float
                - num_z_planes: int
                - delta_z_um: float
                - channel_handle: str (used to determine fluorescence vs brightfield limits)
        
        Raises:
            HTTPException: If any parameter is outside valid range
        """
        if not params:
            return

        # Get current hardware limits
        limits = self.get_hardware_limits()

        # Validate exposure time
        if 'exposure_time_ms' in params:
            self._validate_parameter_range(
                params['exposure_time_ms'],
                limits.imaging_exposure_time_ms,
                'exposure_time_ms'
            )

        # Validate analog gain
        if 'analog_gain' in params:
            self._validate_parameter_range(
                params['analog_gain'],
                limits.imaging_analog_gain_db,
                'analog_gain'
            )

        # Validate focus offset
        if 'z_offset_um' in params:
            self._validate_parameter_range(
                params['z_offset_um'],
                limits.imaging_focus_offset_um,
                'z_offset_um'
            )

        # Validate illumination power (check channel type for appropriate limits)
        if 'illum_perc' in params:
            channel_handle = params.get('channel_handle', '')
            if channel_handle.startswith('bfled'):
                limit_dict = limits.imaging_illum_perc_brightfield
                param_name = 'illum_perc (brightfield)'
            elif channel_handle.startswith('f'):
                limit_dict = limits.imaging_illum_perc_fluorescence
                param_name = 'illum_perc (fluorescence)'
            else:
                limit_dict = limits.imaging_illum_perc
                param_name = 'illum_perc'

            self._validate_parameter_range(
                params['illum_perc'],
                limit_dict,
                param_name
            )

        # Validate number of Z planes
        if 'num_z_planes' in params:
            self._validate_parameter_range(
                params['num_z_planes'],
                limits.imaging_number_z_planes,
                'num_z_planes'
            )

        # Validate Z spacing
        if 'delta_z_um' in params:
            self._validate_parameter_range(
                params['delta_z_um'],
                limits.imaging_delta_z_um,
                'delta_z_um'
            )

    def _validate_parameter_range(self, value: float, limit_dict: dict[str, float | int], param_name: str) -> None:
        """
        Validate a single parameter against its limits.
        
        Args:
            value: Value to validate
            limit_dict: Dictionary with 'min', 'max', 'step' keys
            param_name: Parameter name for error messages
        
        Raises:
            HTTPException: If value is outside valid range or doesn't match step
        """
        min_val = limit_dict.get('min')
        max_val = limit_dict.get('max')
        step_val = limit_dict.get('step')

        if min_val is not None and value < min_val:
            cmd.error_internal(
                detail=f"{param_name} value {value} is below minimum {min_val}"
            )

        if max_val is not None and value > max_val:
            cmd.error_internal(
                detail=f"{param_name} value {value} is above maximum {max_val}"
            )

        # Validate step increment (optional - some parameters may not have step validation)
        if step_val is not None and step_val > 0:
            if min_val is not None:
                # Don't validate step compliance - just snap to nearest valid step
                # This avoids floating-point precision issues with modulo operations
                offset = value - min_val
                snapped_offset = round(offset / step_val) * step_val
                snapped_value = min_val + snapped_offset
                # We could warn or adjust the value here if needed, but for now just accept it
                # The caller should use: usedval = round((val - min_val) / step) * step + min_val

    def _validate_acquisition_config(self, config: sc.AcquisitionConfig) -> None:
        """
        Validate all channels in an acquisition configuration.
        
        Args:
            config: Acquisition configuration to validate
            
        Raises:
            HTTPException: If any channel has invalid parameters
        """
        for channel in config.channels:
            if channel.enabled:
                try:
                    self._validate_imaging_parameters(
                        exposure_time_ms=channel.exposure_time_ms,
                        analog_gain=channel.analog_gain,
                        z_offset_um=channel.z_offset_um,
                        illum_perc=channel.illum_perc,
                        num_z_planes=channel.num_z_planes,
                        delta_z_um=channel.delta_z_um,
                        channel_handle=channel.handle
                    )
                except Exception as e:
                    # Re-raise with channel context
                    cmd.error_internal(
                        detail=f"Channel '{channel.handle}' validation failed: {str(e).split('detail=')[-1].strip('\"')}"
                    )

    def validate_command(self, command: cmd.BaseCommand[tp.Any]) -> None:
        """
        Validate a command against hardware limits before execution.
        
        Args:
            command: Command object to validate
            
        Raises:
            HTTPException: If any parameter is outside valid range
        """
        if isinstance(command, cmd.AutofocusSnap):
            self._validate_imaging_parameters(
                exposure_time_ms=command.exposure_time_ms,
                analog_gain=command.analog_gain
            )

        elif isinstance(command, (cmd.ChannelSnapshot, cmd.ChannelStreamBegin)):
            self._validate_imaging_parameters(
                exposure_time_ms=command.channel.exposure_time_ms,
                analog_gain=command.channel.analog_gain,
                z_offset_um=command.channel.z_offset_um,
                illum_perc=command.channel.illum_perc,
                num_z_planes=command.channel.num_z_planes,
                delta_z_um=command.channel.delta_z_um,
                channel_handle=command.channel.handle
            )

        # Other commands don't require validation (movement, connection, etc.)

    def validate_channel_for_acquisition(self, channel: sc.AcquisitionChannelConfig) -> None:
        """
        Validate that a channel can be acquired with current microscope configuration.

        For MockMicroscope, validates that if filters are configured, the channel
        has a filter selected.

        Args:
            channel: The channel configuration to validate

        Raises:
            Calls error_internal() if validation fails
        """
        if self.filters and channel.enabled:
            if channel.filter_handle is None or channel.filter_handle == '':
                cmd.error_internal(
                    detail=f"Channel '{channel.name}' has no filter selected, but filter wheel is available. "
                    "Please select a filter for all enabled channels."
                )

    def extend_machine_config(self, config_items: list[ConfigItem]) -> None:
        """
        Extend machine configuration with microscope-specific options.

        For MockMicroscope, this is a no-op since there are no actual hardware
        capabilities to query. Mock microscope doesn't have real cameras that
        can report supported formats.

        Args:
            config_items: List of ConfigItem objects to modify in-place
        """
        # Mock microscope has no real hardware, so no extension needed
        pass

    async def execute[T](self, command: cmd.BaseCommand[T]) -> T:
        """Execute mock commands."""

        # Validate command against hardware limits first
        self.validate_command(command)

        if isinstance(command, cmd.EstablishHardwareConnection):
            self.open_connections()
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.LoadingPositionEnter):
            logger.info("Mock microscope: entering loading position")

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="Cannot enter loading position while camera is streaming - stop streaming first")

            # Simulate gradual movement to loading position (0, 0, 0)
            await self._simulate_gradual_movement(target_x_mm=0.0, target_y_mm=0.0, target_z_mm=0.0)

            self.is_in_loading_position = True
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.LoadingPositionLeave):
            logger.info("Mock microscope: leaving loading position")

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="Cannot leave loading position while camera is streaming - stop streaming first")

            # Move to center of plate (127.8mm width, ~80mm height)
            plate_center_x = 127.8 / 2.0  # 63.9mm
            plate_center_y = 80.0 / 2.0   # 40.0mm
            current_z = self._pos_z_measured_to_real(self._current_position.z_pos_mm)

            await self._simulate_gradual_movement(target_x_mm=plate_center_x, target_y_mm=plate_center_y, target_z_mm=current_z)
            self.is_in_loading_position = False
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.MoveTo):
            logger.info(f"Mock microscope: moving to ({command.x_mm}, {command.y_mm}, {command.z_mm})")

            # Check forbidden areas if we have complete X,Y coordinates
            if command.x_mm is not None and command.y_mm is not None:
                is_forbidden, error_message = cmd.positionIsForbidden(
                    command.x_mm, command.y_mm
                )
                if is_forbidden:
                    cmd.error_internal(detail=error_message)

            # Calculate target position (use current position for unspecified axes)
            target_x = command.x_mm if command.x_mm is not None else self._pos_x_measured_to_real(self._current_position.x_pos_mm)
            target_y = command.y_mm if command.y_mm is not None else self._pos_y_measured_to_real(self._current_position.y_pos_mm)
            target_z = command.z_mm if command.z_mm is not None else self._pos_z_measured_to_real(self._current_position.z_pos_mm)

            # Simulate gradual movement with real-time position updates
            await self._simulate_gradual_movement(target_x_mm=target_x, target_y_mm=target_y, target_z_mm=target_z)
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.MoveBy):
            logger.info(f"Mock microscope: moving {command.axis} by {command.distance_mm}mm")

            # Calculate target position after relative movement
            current_x = self._pos_x_measured_to_real(self._current_position.x_pos_mm)
            current_y = self._pos_y_measured_to_real(self._current_position.y_pos_mm)
            current_z = self._pos_z_measured_to_real(self._current_position.z_pos_mm)

            if command.axis == "x":
                target_x = current_x + command.distance_mm
                target_y = current_y
                target_z = current_z
            elif command.axis == "y":
                target_x = current_x
                target_y = current_y + command.distance_mm
                target_z = current_z
            elif command.axis == "z":
                target_x = current_x
                target_y = current_y
                target_z = current_z + command.distance_mm
            else:
                target_x = current_x
                target_y = current_y
                target_z = current_z

            # Check forbidden areas for X,Y movement
            is_forbidden, error_message = cmd.positionIsForbidden(
                target_x, target_y
            )
            if is_forbidden:
                cmd.error_internal(detail=error_message)

            # Simulate gradual movement with real-time position updates
            await self._simulate_gradual_movement(target_x_mm=target_x, target_y_mm=target_y, target_z_mm=target_z)
            return cmd.MoveByResult(axis=command.axis, moved_by_mm=command.distance_mm)  # type: ignore

        elif isinstance(command, cmd.MoveToWell):
            logger.info(f"Mock microscope: moving to well {command.well_name}")

            # Calculate well center position (offset + half well size)
            well_x = command.plate_type.get_well_offset_x(command.well_name) + command.plate_type.Well_size_x_mm / 2
            well_y = command.plate_type.get_well_offset_y(command.well_name) + command.plate_type.Well_size_y_mm / 2

            # Check if well center position is in forbidden area
            is_forbidden, error_message = cmd.positionIsForbidden(
                well_x, well_y
            )
            if is_forbidden:
                cmd.error_internal(detail=f"Well {command.well_name} center position is in forbidden area - {error_message}")

            # Current Z position doesn't change for well moves
            current_z = self._pos_z_measured_to_real(self._current_position.z_pos_mm)

            # Simulate gradual movement with real-time position updates
            await self._simulate_gradual_movement(target_x_mm=well_x, target_y_mm=well_y, target_z_mm=current_z)
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.ChannelSnapshot):
            logger.info(f"Mock microscope: snapping channel {command.channel.handle}")

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="Cannot take snapshot while camera is streaming")

            # Validate channel configuration for acquisition
            self.validate_channel_for_acquisition(command.channel)

            # Simulate imaging delay based on exposure time
            await self._delay_for_imaging(command.channel.exposure_time_ms)

            # Generate synthetic image via camera
            synthetic_img = self.main_camera.value.snap(command.channel)

            result = cmd.ImageAcquiredResponse()
            result._img = synthetic_img
            result._cambits = 16
            return result  # type: ignore

        elif isinstance(command, cmd.MC_getLastPosition):
            # Convert adapter.Position to microcontroller.Position
            from seafront.hardware.firmware_config import get_firmware_config
            from seafront.hardware.microcontroller import Position as McPosition

            firmware_config = get_firmware_config()

            # Convert mm to micro-steps
            x_usteps = int(self._current_position.x_pos_mm / firmware_config.mm_per_ustep_x)
            y_usteps = int(self._current_position.y_pos_mm / firmware_config.mm_per_ustep_y)
            z_usteps = int(self._current_position.z_pos_mm / firmware_config.mm_per_ustep_z)

            return McPosition(x_usteps=x_usteps, y_usteps=y_usteps, z_usteps=z_usteps)  # type: ignore

        elif isinstance(command, cmd.IlluminationEndAll):
            logger.info("Mock microscope: turning off all illumination")
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.AutofocusLaserWarmup):
            logger.info("Mock microscope: warming up autofocus laser")
            await asyncio.sleep(0.1)  # Simulate warmup time
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.AutofocusMeasureDisplacement):
            logger.info("Mock microscope: measuring autofocus displacement")
            # Mock measurement - return zero displacement
            return cmd.AutofocusMeasureDisplacementResult(displacement_um=0.0)  # type: ignore

        elif isinstance(command, cmd.AutofocusSnap):
            logger.info("Mock microscope: taking autofocus snapshot")

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="Cannot capture autofocus image while main camera is streaming")

            # Simulate imaging delay (autofocus typically uses shorter exposure ~5ms)
            await self._delay_for_imaging(5.0)

            # Generate mock autofocus image (smaller, grayscale)
            synthetic_img = np.random.randint(0, 65536, (256, 256), dtype=np.uint16)
            result = cmd.AutofocusSnapResult(width_px=256, height_px=256)
            result._img = synthetic_img
            return result  # type: ignore

        elif isinstance(command, cmd.ChannelStreamBegin):
            logger.info(f"Mock microscope: starting stream for channel {command.channel.handle}")

            # Check if already streaming
            if self.stream_callback is not None:
                cmd.error_internal("Streaming already active - stop current stream before starting a new one")

            # Validate channel configuration for acquisition
            self.validate_channel_for_acquisition(command.channel)

            # Find matching channel config
            matching_configs = [ch for ch in self.channels if ch.handle == command.channel.handle]
            if not matching_configs:
                cmd.error_internal(f"No channel config found for handle: {command.channel.handle}")

            self._streaming_channel = matching_configs[0]
            self._streaming_acquisition_config = command.channel

            # Create wrapper to forward camera images to microscope's stream callback
            def streaming_callback_wrapper(img: np.ndarray) -> None:
                if self.stream_callback is not None:
                    self.stream_callback(img)

            # Delegate streaming to camera (camera manages its own thread)
            self.main_camera.value.start_streaming(command.channel, callback=streaming_callback_wrapper)

            return cmd.StreamingStartedResponse(channel=command.channel)  # type: ignore

        elif isinstance(command, cmd.ChannelStreamEnd):
            logger.info("Mock microscope: ending channel stream")

            # Stop camera streaming (camera manages its own thread cleanup)
            self.main_camera.value.stop_streaming()

            # Clean up streaming state
            self._streaming_channel = None
            self._streaming_acquisition_config = None
            self.stream_callback = None

            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.LaserAutofocusCalibrate):
            logger.info("Mock microscope: calibrating laser autofocus")
            # Read z_span from machine config
            conf_z_span_item = LaserAutofocusConfig.CALIBRATION_Z_SPAN_MM.value_item
            assert conf_z_span_item is not None
            z_span = conf_z_span_item.floatvalue
            logger.debug(f"Using z_span_mm={z_span} for autofocus calibration")
            await asyncio.sleep(0.2)  # Simulate calibration time
            # Mock calibration result using current position
            current_state = await self.get_current_state()
            calibration_data = cmd.LaserAutofocusCalibrationData(
                um_per_px=1.0,
                x_reference=0.0,
                calibration_position=current_state.stage_position
            )
            return cmd.LaserAutofocusCalibrationResponse(calibration_data=calibration_data)  # type: ignore

        elif isinstance(command, cmd.AutofocusApproachTargetDisplacement):
            logger.info(
                f"Mock microscope: approaching target displacement "
                f"(target={command.target_offset_um}um, max_reps={command.max_num_reps})"
            )
            # Mock implementation: simulate reaching the target with minimal moves
            # In real hardware, this would iteratively measure and move until reaching the target

            # Simulate the approach process
            await asyncio.sleep(0.1)

            # For mock: simulate that we reach the target threshold quickly
            # Typical behavior: 1-2 compensating moves needed
            num_compensating_moves = 1
            uncompensated_offset_mm = 0.0  # Successfully reached target
            reached_threshold = True  # Converged to target

            return cmd.AutofocusApproachTargetDisplacementResult(
                num_compensating_moves=num_compensating_moves,
                uncompensated_offset_mm=uncompensated_offset_mm,
                reached_threshold=reached_threshold,
            )  # type: ignore

        elif isinstance(command, cmd. Home):
            await self.home(command.x,command.y,command.z)

            return cmd.BasicSuccessResponse() # type: ignore

        else:
            logger.warning(f"Mock microscope: unhandled command type {type(command)}")
            # Return a basic success response for unhandled commands
            return cmd.BasicSuccessResponse()  # type: ignore
