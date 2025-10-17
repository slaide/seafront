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
import threading
import time
import typing as tp
from contextlib import contextmanager

import json5
import numpy as np
import seaconfig as sc
from pydantic import PrivateAttr

from seafront.config.basics import ChannelConfig, FilterConfig, GlobalConfigHandler, ImagingOrder
from seafront.config.handles import ImagingConfig
from seafront.hardware.adapter import AdapterState, CoreState, Position
from seafront.hardware.camera import HardwareLimitValue
from seafront.hardware.firmware_config import (
    get_firmware_config,
)
from seafront.hardware.illumination import IlluminationController
from seafront.hardware.microscope import HardwareLimits, Microscope, microscope_exclusive
from seafront.logger import logger
from seafront.server import commands as cmd


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

    # Mock hardware state
    _current_position: Position = PrivateAttr(default_factory=lambda: Position(x_pos_mm=0.0, y_pos_mm=0.0, z_pos_mm=0.0))
    _is_homed: bool = PrivateAttr(default=False)
    _streaming_channel: ChannelConfig | None = PrivateAttr(default=None)
    _streaming_acquisition_config: sc.AcquisitionChannelConfig | None = PrivateAttr(default=None)
    _streaming_thread: threading.Thread | None = PrivateAttr(default=None)
    _streaming_stop_event: threading.Event = PrivateAttr(default_factory=threading.Event)
    _streaming_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
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
        from seafront.config.handles import ProtocolConfig
        from seafront.hardware.forbidden_areas import ForbiddenAreaList

        g_config = GlobalConfigHandler.get_dict()
        forbidden_areas_entry = g_config.get(ProtocolConfig.FORBIDDEN_AREAS.value)

        # If no forbidden areas config is found, allow the movement
        if forbidden_areas_entry is None:
            return False, ""

        forbidden_areas_str = forbidden_areas_entry.value
        if not isinstance(forbidden_areas_str, str):
            logger.warning("forbidden_areas entry is not a string, allowing movement")
            return False, ""

        try:
            forbidden_areas = ForbiddenAreaList.from_json_string(forbidden_areas_str)
        except ValueError as e:
            logger.warning(f"Invalid forbidden areas configuration: {e}, allowing movement")
            return False, ""

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
    def lock(self, blocking: bool = True) -> tp.Iterator[tp.Self | None]:
        """Mock lock - always succeeds immediately."""
        if self._lock.acquire(blocking=blocking):
            try:
                yield self
            finally:
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

        return cls(
            channels=channel_configs,
            filters=filter_configs,
            illumination_controller=illumination_controller,
        )

    @microscope_exclusive
    def open_connections(self) -> None:
        """Mock connection opening - always succeeds."""
        logger.info("Mock microscope: opening connections")
        self.is_connected = True
        self.state = CoreState.Idle

    def close(self) -> None:
        """Mock connection closing."""
        logger.info("Mock microscope: closing connections")

        # Stop background thread
        if self._streaming_thread is not None:
            self._streaming_stop_event.set()
            self._streaming_thread.join(timeout=1.0)
            self._streaming_thread = None

        # Clean up streaming state
        self._streaming_channel = None
        self._streaming_acquisition_config = None
        self.stream_callback = None
        self.is_connected = False
        self.state = CoreState.Idle

    async def home(self) -> None:
        """Mock homing sequence."""
        logger.info("Mock microscope: performing homing sequence")

        # Simulate homing time
        await asyncio.sleep(0.1)

        self._is_homed = True
        self._current_position = Position(x_pos_mm=0.0, y_pos_mm=0.0, z_pos_mm=0.0)
        self.is_in_loading_position = False

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
            state=self.state,
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

    def _background_streaming_thread(self) -> None:
        """Background thread that continuously generates streaming images and calls callbacks."""
        frame_rate = 6  # FPS - realistic microscope streaming rate
        frame_interval = 1.0 / frame_rate
        frame_number = 0

        logger.info("Mock microscope: background streaming thread started")

        while not self._streaming_stop_event.is_set():
            if self._streaming_channel is None or self._streaming_acquisition_config is None:
                time.sleep(0.1)
                continue

            # Generate synthetic image with frame-based variation and acquisition parameters
            synthetic_img = self._generate_synthetic_image(
                self._streaming_channel,
                frame_number=frame_number,
                exposure_time_ms=self._streaming_acquisition_config.exposure_time_ms,
                analog_gain=self._streaming_acquisition_config.analog_gain
            )

            # Call stream_callback directly (like camera hardware does)
            with self._streaming_lock:
                current_callback = self.stream_callback

            if current_callback is not None:
                try:
                    # This is the key: call callback for EVERY frame generated
                    should_stop = current_callback(synthetic_img)

                    if should_stop:
                        logger.info("Mock microscope: stream callback requested stop")
                        break

                except Exception as e:
                    logger.error(f"Mock microscope: error in stream callback: {e}")
                    break

            frame_number += 1

            # Wait for next frame or until stop event is set
            if self._streaming_stop_event.wait(frame_interval):
                break

        logger.info("Mock microscope: background streaming thread stopped")

    def _generate_synthetic_image(self, channel: ChannelConfig, width: int = 1024, height: int = 1024, frame_number: int = 0, exposure_time_ms: float = 50.0, analog_gain: float = 1.0) -> np.ndarray:
        """Generate a synthetic image for testing."""
        # Use frame number to create variation between frames
        np.random.seed((frame_number * 73 + hash(channel.handle)) % 2**31)

        # Add time-based noise variation
        time_factor = np.sin(frame_number * 0.1) * 0.2 + 1.0  # Varies between 0.8 and 1.2

        # Calculate exposure and gain factors
        # Exposure: linear relationship, normalized to 50ms baseline
        exposure_factor = exposure_time_ms / 50.0

        # Analog gain: convert from dB to linear scale
        # 0 dB = 1x, 10 dB = 10x, 20 dB = 100x
        gain_factor = 10.0 ** (analog_gain / 10.0)

        # Create different patterns based on channel type
        if channel.handle.startswith('bfled'):
            # Brightfield: create cellular-like structures with variation
            # Start with low background that scales with exposure
            base_intensity = int(2000 * time_factor * exposure_factor)  # Much lower background
            base = np.random.normal(base_intensity, 500, (height, width)).astype(np.uint16)

            # Add some circular "cells" that slightly move/change
            y, x = np.ogrid[:height, :width]
            num_cells = 20 + int(5 * np.sin(frame_number * 0.05))  # Cell count varies
            for i in range(num_cells):
                # Cells have slight movement over time
                base_x = 100 + (i * 100) % (width - 100)
                base_y = 100 + ((i * 73) % (height - 100))
                drift_x = 10 * np.sin(frame_number * 0.02 + i)
                drift_y = 10 * np.cos(frame_number * 0.03 + i)

                center_x = int(base_x + drift_x)
                center_y = int(base_y + drift_y)
                radius = 25 + int(10 * np.sin(frame_number * 0.04 + i))  # Size varies

                if 0 <= center_x < width and 0 <= center_y < height:
                    mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
                    # Signal from cells scales with exposure time (more photons collected)
                    cell_signal = int(8000 * time_factor * exposure_factor * (0.8 + 0.4 * np.random.random()))
                    # Add signal to existing background (avoid saturation)
                    base[mask] += np.clip(cell_signal + np.random.normal(0, 500, int(np.sum(mask))), 0, 50000).astype(np.uint16)

        else:
            # Fluorescence: create spot-like structures with blinking/movement
            # Very low background for fluorescence that scales with exposure
            base_intensity = int(300 * time_factor * exposure_factor)  # Much lower background
            base = np.random.normal(base_intensity, 100, (height, width)).astype(np.uint16)

            # Add bright fluorescent spots that blink and move
            y, x = np.ogrid[:height, :width]
            num_spots = 25 + int(10 * np.sin(frame_number * 0.08))  # Number of spots varies
            for i in range(num_spots):
                # Some spots blink on/off
                blink_factor = max(0, np.sin(frame_number * 0.1 + i * 0.5))
                if blink_factor < 0.2:  # Skip spots that are "off"
                    continue

                # More irregular and spread out positioning using different prime numbers
                base_x = (i * 347 + hash(channel.handle) * 191) % width
                base_y = (i * 431 + hash(channel.handle) * 239) % height
                drift_x = 8 * np.sin(frame_number * 0.03 + i * 0.7)
                drift_y = 8 * np.cos(frame_number * 0.04 + i * 0.7)

                center_x = int(base_x + drift_x) % width
                center_y = int(base_y + drift_y) % height
                radius = 15 + int(10 * np.sin(frame_number * 0.06 + i))  # Much larger spots, similar to cells

                mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
                # Fluorescent signal scales with exposure time (more photons collected)
                spot_signal = int(12000 * time_factor * exposure_factor * blink_factor * (0.7 + 0.6 * np.random.random()))
                # Add signal to existing background (avoid saturation)
                base[mask] += np.clip(spot_signal + np.random.normal(0, 1000, int(np.sum(mask))), 0, 40000).astype(np.uint16)

        # Apply analog gain to entire image (background + signal + noise)
        # This simulates the camera's analog amplification stage
        final_image = base.astype(np.float32) * gain_factor

        return np.clip(final_image, 0, 65535).astype(np.uint16)

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
            # Find matching channel config
            matching_configs = [ch for ch in self.channels if ch.handle == channel.handle]
            if not matching_configs:
                logger.warning(f"No channel config found for handle: {channel.handle}")
                continue

            channel_config = matching_configs[0]

            # Generate synthetic image
            frame_num = int(time.time() * 10 + hash(channel.handle)) % 10000
            synthetic_img = self._generate_synthetic_image(
                channel_config,
                frame_number=frame_num,
                exposure_time_ms=channel.exposure_time_ms,
                analog_gain=channel.analog_gain
            )

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

        elif isinstance(command, cmd.ChannelSnapSelection):
            self._validate_acquisition_config(command.config_file)

        # Other commands don't require validation (movement, connection, etc.)

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
                cmd.error_internal(detail="already streaming")

            # Simulate gradual movement to loading position (0, 0, 0)
            await self._simulate_gradual_movement(target_x_mm=0.0, target_y_mm=0.0, target_z_mm=0.0)

            self.is_in_loading_position = True
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.LoadingPositionLeave):
            logger.info("Mock microscope: leaving loading position")

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="already streaming")

            # Move to center of plate (127.8mm width, ~80mm height)
            plate_center_x = 127.8 / 2.0  # 63.9mm
            plate_center_y = 80.0 / 2.0   # 40.0mm
            current_z = self._pos_z_measured_to_real(self._current_position.z_pos_mm)

            await self._simulate_gradual_movement(target_x_mm=plate_center_x, target_y_mm=plate_center_y, target_z_mm=current_z)
            self.is_in_loading_position = False
            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.MoveTo):
            logger.info(f"Mock microscope: moving to ({command.x_mm}, {command.y_mm}, {command.z_mm})")

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="already streaming")

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

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="already streaming")

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

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="already streaming")

            # Check if well is forbidden
            plate = command.plate_type
            if cmd.wellIsForbidden(command.well_name, plate):
                cmd.error_internal(detail="well is forbidden")

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
                cmd.error_internal(detail="already streaming")

            # Simulate imaging delay based on exposure time
            await self._delay_for_imaging(command.channel.exposure_time_ms)

            # Find matching channel config
            matching_configs = [ch for ch in self.channels if ch.handle == command.channel.handle]
            if not matching_configs:
                raise ValueError(f"No channel config found for handle: {command.channel.handle}")

            channel_config = matching_configs[0]
            # For snapshots, use current time to create unique images
            frame_num = int(time.time() * 10) % 10000  # Use time-based frame number
            synthetic_img = self._generate_synthetic_image(
                channel_config,
                frame_number=frame_num,
                exposure_time_ms=command.channel.exposure_time_ms,
                analog_gain=command.channel.analog_gain
            )

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
                cmd.error_internal(detail="already streaming")

            # Simulate imaging delay (autofocus typically uses shorter exposure ~5ms)
            await self._delay_for_imaging(5.0)

            # Generate mock autofocus image (smaller, grayscale)
            synthetic_img = np.random.randint(0, 65536, (256, 256), dtype=np.uint16)
            result = cmd.AutofocusSnapResult(width_px=256, height_px=256)
            result._img = synthetic_img
            return result  # type: ignore

        elif isinstance(command, cmd.ChannelSnapSelection):
            logger.info("Mock microscope: snapping selected channels")

            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="already streaming")

            channel_handles: list[str] = []
            channel_images: dict[str, np.ndarray] = {}

            for channel in command.config_file.channels:
                if not channel.enabled:
                    continue

                # Create individual ChannelSnapshot command for each enabled channel
                cmd_snap = cmd.ChannelSnapshot(
                    channel=channel,
                    machine_config=command.config_file.machine_config or [],
                )

                # Execute the individual channel snapshot (reuses existing logic)
                res = await self.execute(cmd_snap)

                # Store the generated image
                channel_images[channel.handle] = res._img
                channel_handles.append(channel.handle)

            logger.info(f"Mock microscope: snapped {len(channel_handles)} channels")

            # Create result with generated images
            result = cmd.ChannelSnapSelectionResult(channel_handles=channel_handles)
            result._images = channel_images
            return result  # type: ignore

        elif isinstance(command, cmd.ChannelStreamBegin):
            logger.info(f"Mock microscope: starting stream for channel {command.channel.handle}")

            # Check if already streaming
            if self.stream_callback is not None and self._streaming_thread is not None:
                cmd.error_internal("already streaming")

            # Find matching channel config
            matching_configs = [ch for ch in self.channels if ch.handle == command.channel.handle]
            if not matching_configs:
                cmd.error_internal(f"No channel config found for handle: {command.channel.handle}")

            self._streaming_channel = matching_configs[0]
            self._streaming_acquisition_config = command.channel

            # Clear stop event and start background thread (mimics camera hardware)
            self._streaming_stop_event.clear()
            self._streaming_thread = threading.Thread(target=self._background_streaming_thread, daemon=True)
            self._streaming_thread.start()

            return cmd.StreamingStartedResponse(channel=command.channel)  # type: ignore

        elif isinstance(command, cmd.ChannelStreamEnd):
            logger.info("Mock microscope: ending channel stream")

            # Stop background thread
            if self._streaming_thread is not None:
                self._streaming_stop_event.set()
                self._streaming_thread.join(timeout=1.0)  # Wait up to 1 second
                self._streaming_thread = None

            # Clean up streaming state
            self._streaming_channel = None
            self._streaming_acquisition_config = None
            self.stream_callback = None

            return cmd.BasicSuccessResponse()  # type: ignore

        elif isinstance(command, cmd.LaserAutofocusCalibrate):
            logger.info("Mock microscope: calibrating laser autofocus")
            await asyncio.sleep(0.2)  # Simulate calibration time
            # Mock calibration result using current position
            current_state = await self.get_current_state()
            calibration_data = cmd.LaserAutofocusCalibrationData(
                um_per_px=1.0,
                x_reference=0.0,
                calibration_position=current_state.stage_position
            )
            return cmd.LaserAutofocusCalibrationResponse(calibration_data=calibration_data)  # type: ignore

        else:
            logger.warning(f"Mock microscope: unhandled command type {type(command)}")
            # Return a basic success response for unhandled commands
            return cmd.BasicSuccessResponse()  # type: ignore
