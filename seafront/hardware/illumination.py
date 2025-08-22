import numpy as np
import typing as tp
from typing import Dict

from seafront.config.basics import ChannelConfig, PowerCalibration
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
        self.intensity_luts: Dict[str, Dict[str, np.ndarray]] = {}
        self.max_power: Dict[str, float] = {}
        
        self._build_calibration_data()
    
    def _build_calibration_data(self):
        """Build calibration lookup tables from channel configuration data"""
        for handle, channel in self.channels.items():
            if channel.use_power_calibration and channel.power_calibration is not None:
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
                    
                    logger.info(f"Loaded intensity calibration for channel {handle}, max power: {max_power:.2f} mW")
                    
                except Exception as e:
                    logger.error(f"Failed to load calibration for channel {handle}: {e}")
                    # Don't add to LUTs, will fall back to linear scaling
    
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
        if channel_handle in self.intensity_luts:
            calibrated = self._apply_lut(channel_handle, requested_intensity)
            logger.debug(f"Channel {channel_handle}: {requested_intensity}% -> {calibrated:.1f}% (calibrated)")
            return calibrated
        else:
            logger.debug(f"Channel {channel_handle}: no calibration found, using linear scaling")
            return requested_intensity
    
    def get_max_power(self, channel_handle: str) -> tp.Optional[float]:
        """Get the maximum power in mW for a channel, if calibration is available."""
        return self.max_power.get(channel_handle)