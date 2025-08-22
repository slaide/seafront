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
import json5
import os
import threading
import time
import typing as tp
from contextlib import contextmanager

import numpy as np
import seaconfig as sc
from pydantic import Field, PrivateAttr

from seafront.config.basics import ChannelConfig, FilterConfig, GlobalConfigHandler
from seafront.hardware.adapter import AdapterState, CoreState, Position
from seafront.hardware.illumination import IlluminationController
from seafront.hardware.microscope import Microscope, microscope_exclusive
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
    
    @property
    def _realistic_delays_enabled(self) -> bool:
        """Check if realistic delays should be simulated (default: True, disable with MOCK_NO_DELAYS=1)"""
        return os.environ.get("MOCK_NO_DELAYS", "0").lower() not in ("1", "true", "yes")
    
    async def _simulate_gradual_movement(self, target_x_mm: float | None = None, target_y_mm: float | None = None, target_z_mm: float | None = None) -> None:
        """Simulate realistic movement with gradual position updates (4cm/s)"""
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
        
        # Calculate 3D distance
        dx = final_x - start_x
        dy = final_y - start_y
        dz = final_z - start_z
        distance_mm = (dx**2 + dy**2 + dz**2)**0.5
        
        # Convert to cm and calculate time at 4cm/s
        distance_cm = distance_mm / 10.0
        movement_time_s = distance_cm / 4.0  # 4 cm/s
        
        # Add minimum time for acceleration/deceleration (50ms)
        total_time_s = max(0.05, movement_time_s)
        
        logger.debug(f"Mock movement: {distance_mm:.1f}mm distance, {total_time_s:.3f}s duration")
        
        # Update position gradually during movement
        update_interval_s = 0.02  # Update every 20ms for smooth movement
        num_steps = max(1, int(total_time_s / update_interval_s))
        
        for step in range(num_steps):
            # Calculate interpolation factor (0.0 to 1.0)
            progress = (step + 1) / num_steps
            
            # Interpolate position
            current_x = start_x + dx * progress
            current_y = start_y + dy * progress  
            current_z = start_z + dz * progress
            
            # Update measured position (this is what get_current_state() sees)
            self._current_position.x_pos_mm = self._pos_x_real_to_measured(current_x)
            self._current_position.y_pos_mm = self._pos_y_real_to_measured(current_y)
            self._current_position.z_pos_mm = self._pos_z_real_to_measured(current_z)
            
            # Wait for next update (or finish if last step)
            if step < num_steps - 1:
                await asyncio.sleep(update_interval_s)
            else:
                # Final step - sleep remaining time
                remaining_time = total_time_s - (step * update_interval_s)
                if remaining_time > 0:
                    await asyncio.sleep(remaining_time)
    
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
        channels_json = g_dict["channels"].strvalue
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
        if "filters" in g_dict:
            filters_json = g_dict["filters"].strvalue
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
        
        x_offset = g_dict["calibration_offset_x_mm"].floatvalue
        y_offset = g_dict["calibration_offset_y_mm"].floatvalue  
        z_offset = g_dict["calibration_offset_z_mm"].floatvalue
        
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
                    base[mask] += np.clip(cell_signal + np.random.normal(0, 500, np.sum(mask)), 0, 50000).astype(np.uint16)
                
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
                base[mask] += np.clip(spot_signal + np.random.normal(0, 1000, np.sum(mask)), 0, 40000).astype(np.uint16)
        
        # Apply analog gain to entire image (background + signal + noise)
        # This simulates the camera's analog amplification stage
        final_image = base.astype(np.float32) * gain_factor
        
        return np.clip(final_image, 0, 65535).astype(np.uint16)
    
    async def snap_selected_channels(self, config_file: sc.AcquisitionConfig) -> cmd.BasicSuccessResponse:
        """Mock channel snapping."""
        logger.info("Mock microscope: snapping selected channels")
        
        enabled_channels = [ch for ch in config_file.channels if ch.enabled]
        
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
    
    async def execute[T](self, command: cmd.BaseCommand[T]) -> T:
        """Execute mock commands."""
        
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
            
            # Simulate gradual movement with real-time position updates
            await self._simulate_gradual_movement(target_x_mm=target_x, target_y_mm=target_y, target_z_mm=target_z)
            return cmd.MoveByResult(axis=command.axis, moved_by_mm=command.distance_mm)  # type: ignore
            
        elif isinstance(command, cmd.MoveToWell):
            logger.info(f"Mock microscope: moving to well {command.well_name}")
            
            # Check if streaming is active
            if self.stream_callback is not None:
                cmd.error_internal(detail="already streaming")
            
            # Calculate well center position (offset + half well size)
            well_x = command.plate_type.get_well_offset_x(command.well_name) + command.plate_type.Well_size_x_mm / 2
            well_y = command.plate_type.get_well_offset_y(command.well_name) + command.plate_type.Well_size_y_mm / 2
            
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
            from seafront.hardware.microcontroller import Position as McPosition, FirmwareDefinitions
            
            # Convert mm to micro-steps
            x_usteps = int(self._current_position.x_pos_mm / FirmwareDefinitions.mm_per_ustep_x())
            y_usteps = int(self._current_position.y_pos_mm / FirmwareDefinitions.mm_per_ustep_y())  
            z_usteps = int(self._current_position.z_pos_mm / FirmwareDefinitions.mm_per_ustep_z())
            
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
            # Mock calibration result
            calibration_data = cmd.LaserAutofocusCalibrationData(
                um_per_px=1.0,
                x_reference=0.0,
                calibration_position=Position(x_pos_mm=0.0, y_pos_mm=0.0, z_pos_mm=0.0)
            )
            return cmd.LaserAutofocusCalibrationResponse(calibration_data=calibration_data)  # type: ignore
            
        else:
            logger.warning(f"Mock microscope: unhandled command type {type(command)}")
            # Return a basic success response for unhandled commands
            return cmd.BasicSuccessResponse()  # type: ignore