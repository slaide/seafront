
import math

import numpy as np

from seafront.config.basics import ChannelConfig
from seafront.logger import logger


class IlluminationController:
    """
    Illumination controller that handles intensity calibration for different light sources.
    
    This provides calibrated power control using channel configuration data,
    converting requested power percentages to proper DAC values using lookup tables.
    """

    def __init__(self, channels: list[ChannelConfig]):
        """
        Args:
            channels: List of channel configurations from the microscope config
        """
        self.channels = {ch.handle: ch for ch in channels}
        self.intensity_luts: dict[str, dict[str, np.ndarray]] = {}
        self.max_power: dict[str, float] = {}
        self.broken_calibration: dict[str, str] = {}
        "channels that asked for calibration and could not have it, and why"

        self._build_calibration_data()

    def _build_calibration_data(self):
        """Build calibration lookup tables from channel configuration data"""
        for handle, channel in self.channels.items():
            if not channel.use_power_calibration:
                continue

            if channel.power_calibration is None:
                # asked for calibration and supplied none: the same mistake as supplying a
                # broken one, and it must not fall through to uncalibrated output either
                self.broken_calibration[handle] = "use_power_calibration is set but no calibration data was given"
                logger.error(f"Channel {handle}: {self.broken_calibration[handle]}")
                continue

            try:
                # Validate calibration data
                channel.validate_calibration()

                cal = channel.power_calibration

                # Store max power for this channel
                self.max_power[handle] = max(cal.optical_power_mw)

                # Create normalized power values (0-100%)
                power_values = np.array(cal.optical_power_mw)
                max_power = self.max_power[handle]
                normalized_power = power_values / max_power * 100

                # DAC values are already in range 0-100
                dac_percent = np.array(cal.dac_percent)

                self.intensity_luts[handle] = {
                    "power_percent": normalized_power,
                    "dac_percent": dac_percent,
                }

            except Exception as e:
                # Remembered rather than forgotten. A channel with
                # use_power_calibration=True is one where somebody decided the requested
                # percentage must NOT go to the hardware directly; falling back to doing
                # exactly that, quietly, is the opposite of what the configuration asks
                # for. The channel refuses when it is used instead, so a miscalibrated
                # laser stops itself rather than the whole microscope.
                logger.error(f"Failed to load calibration for channel {handle}: {e}")
                self.broken_calibration[handle] = str(e)

    def _apply_lut(self, channel_handle: str, intensity_percent: float) -> float:
        """Convert desired power percentage to DAC value (0-100) using LUT."""
        if channel_handle not in self.intensity_luts:
            # No calibration available, use linear scaling
            return intensity_percent

        lut = self.intensity_luts[channel_handle]
        # Ensure intensity is within bounds
        intensity_percent = np.clip(intensity_percent, 0, 100)

        # Interpolate to get DAC value
        dac_percent = np.interp(intensity_percent, lut["power_percent"], lut["dac_percent"])

        # Ensure DAC value is in range 0-100
        return float(np.clip(dac_percent, 0, 100))

    def get_calibrated_intensity(self, channel_handle: str, requested_intensity: float) -> float:
        """
        Get the calibrated intensity for a channel.
        
        Args:
            channel_handle: Channel handle (e.g., "bfledfull", "fluo405")
            requested_intensity: Requested intensity as percentage (0-100)
            
        Returns:
            Calibrated intensity as percentage (0-100) to send to DAC
        """
        if not math.isfinite(requested_intensity):
            # clipping leaves nan alone, and nan is not a value to drive a light source with
            raise ValueError(
                f"Channel {channel_handle}: requested intensity {requested_intensity} "
                "is not a finite number"
            )

        if channel_handle in self.broken_calibration:
            raise ValueError(
                f"Channel {channel_handle} is configured to use power calibration, but its "
                f"calibration could not be loaded ({self.broken_calibration[channel_handle]}). "
                "Refusing to drive it with an uncalibrated value."
            )

        if channel_handle in self.intensity_luts:
            calibrated = self._apply_lut(channel_handle, requested_intensity)
            logger.debug(f"Channel {channel_handle}: {requested_intensity}% -> {calibrated:.1f}% (calibrated)")
            return calibrated
        else:
            logger.debug(f"Channel {channel_handle}: no calibration found, using linear scaling")
            return requested_intensity

    def get_max_power(self, channel_handle: str) -> float | None:
        """Get the maximum power in mW for a channel, if calibration is available."""
        return self.max_power.get(channel_handle)
