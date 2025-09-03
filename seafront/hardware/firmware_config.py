"""
Pydantic-based firmware configuration system for seafront.

This module provides configurable firmware definitions, replacing the static 
FirmwareDefinitions class with a flexible Pydantic model system that can be
selected via the SEAFRONT_FIRMWARE_PROFILE environment variable.
"""

import os
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, computed_field


class FirmwareConfig(BaseModel):
    """
    Pydantic model for firmware configuration parameters.
    
    This replaces the static FirmwareDefinitions class with a flexible configuration
    system that supports different hardware configurations via environment variables.
    """
    model_config = ConfigDict(
        frozen=False,  # Allow modification after instantiation
        extra="forbid",  # Don't allow extra fields
        validate_assignment=True,  # Validate when assigning new values
    )
    
    # Basic packet configuration
    READ_PACKET_LENGTH: int = Field(default=24, description="Length of read packets")
    COMMAND_PACKET_LENGTH: int = Field(default=8, description="Length of command packets")
    
    # Axis constants - these should remain constant
    AXIS_X: int = Field(default=0, description="X axis identifier")
    AXIS_Y: int = Field(default=1, description="Y axis identifier") 
    AXIS_Z: int = Field(default=2, description="Z axis identifier")
    AXIS_THETA: int = Field(default=3, description="Theta axis identifier")
    AXIS_XY: int = Field(default=4, description="XY combined axis identifier")
    AXIS_W: int = Field(default=5, description="W (filter wheel) axis identifier")
    
    # Screw pitch configurations (mm per revolution) - hand-tuned values
    SCREW_PITCH_X_MM: float = Field(default=2.54, description="X axis screw pitch in mm")
    SCREW_PITCH_Y_MM: float = Field(default=2.54, description="Y axis screw pitch in mm")
    SCREW_PITCH_Z_MM: float = Field(default=0.3, description="Z axis screw pitch in mm")
    SCREW_PITCH_W_MM: float = Field(default=1.0, description="W axis screw pitch in mm")
    
    # Microstepping configuration - hand-tuned values
    MICROSTEPPING_DEFAULT_X: int = Field(default=256, description="X axis microstepping")
    MICROSTEPPING_DEFAULT_Y: int = Field(default=256, description="Y axis microstepping")
    MICROSTEPPING_DEFAULT_Z: int = Field(default=256, description="Z axis microstepping")
    MICROSTEPPING_DEFAULT_THETA: int = Field(default=256, description="Theta axis microstepping")
    MICROSTEPPING_DEFAULT_W: int = Field(default=64, description="W axis microstepping")
    
    # Encoder configuration
    USE_ENCODER_X: bool = Field(default=False, description="Use encoder for X axis")
    USE_ENCODER_Y: bool = Field(default=False, description="Use encoder for Y axis") 
    USE_ENCODER_Z: bool = Field(default=False, description="Use encoder for Z axis")
    USE_ENCODER_THETA: bool = Field(default=False, description="Use encoder for Theta axis")
    
    ENCODER_POS_SIGN_X: int = Field(default=1, description="X encoder position sign")
    ENCODER_POS_SIGN_Y: int = Field(default=1, description="Y encoder position sign")
    ENCODER_POS_SIGN_Z: int = Field(default=1, description="Z encoder position sign")
    ENCODER_POS_SIGN_THETA: int = Field(default=1, description="Theta encoder position sign")
    
    ENCODER_STEP_SIZE_X_MM: float = Field(default=100e-6, description="X encoder step size in mm")
    ENCODER_STEP_SIZE_Y_MM: float = Field(default=100e-6, description="Y encoder step size in mm")
    ENCODER_STEP_SIZE_Z_MM: float = Field(default=100e-6, description="Z encoder step size in mm")
    ENCODER_STEP_SIZE_THETA: float = Field(default=1.0, description="Theta encoder step size")
    
    # Motor configuration - steps per revolution
    FULLSTEPS_PER_REV_X: int = Field(default=200, description="X motor full steps per revolution")
    FULLSTEPS_PER_REV_Y: int = Field(default=200, description="Y motor full steps per revolution")
    FULLSTEPS_PER_REV_Z: int = Field(default=200, description="Z motor full steps per revolution")
    FULLSTEPS_PER_REV_THETA: int = Field(default=200, description="Theta motor full steps per revolution")
    FULLSTEPS_PER_REV_W: int = Field(default=200, description="W motor full steps per revolution")
    
    # Stage movement direction signs - hand-tuned values
    STAGE_MOVEMENT_SIGN_X: int = Field(default=1, description="X stage movement direction sign")
    STAGE_MOVEMENT_SIGN_Y: int = Field(default=1, description="Y stage movement direction sign")
    STAGE_MOVEMENT_SIGN_Z: int = Field(default=-1, description="Z stage movement direction sign")
    STAGE_MOVEMENT_SIGN_THETA: int = Field(default=1, description="Theta stage movement direction sign")
    STAGE_MOVEMENT_SIGN_W: int = Field(default=1, description="W stage movement direction sign")
    
    # Motor current settings (mA) - hand-tuned values
    X_MOTOR_RMS_CURRENT_mA: int = Field(default=1000, description="X motor RMS current in mA")
    Y_MOTOR_RMS_CURRENT_mA: int = Field(default=1000, description="Y motor RMS current in mA") 
    Z_MOTOR_RMS_CURRENT_mA: int = Field(default=500, description="Z motor RMS current in mA")
    W_MOTOR_RMS_CURRENT_mA: int = Field(default=1900, description="W motor RMS current in mA")
    
    # Motor hold current (fraction of RMS current, 0.0-1.0) - hand-tuned values
    X_MOTOR_I_HOLD: float = Field(default=0.25, ge=0.0, le=1.0, description="X motor hold current fraction")
    Y_MOTOR_I_HOLD: float = Field(default=0.25, ge=0.0, le=1.0, description="Y motor hold current fraction")
    Z_MOTOR_I_HOLD: float = Field(default=0.5, ge=0.0, le=1.0, description="Z motor hold current fraction")
    W_MOTOR_I_HOLD: float = Field(default=0.5, ge=0.0, le=1.0, description="W motor hold current fraction")
    
    # Velocity limits (mm/s) - hand-tuned values
    MAX_VELOCITY_X_mm: float = Field(default=40.0, description="X axis max velocity in mm/s")
    MAX_VELOCITY_Y_mm: float = Field(default=40.0, description="Y axis max velocity in mm/s")
    MAX_VELOCITY_Z_mm: float = Field(default=2.0, description="Z axis max velocity in mm/s")
    MAX_VELOCITY_W_mm: float = Field(default=3.19, description="W axis max velocity in mm/s")
    
    # Acceleration limits (mm/s²) - hand-tuned values
    MAX_ACCELERATION_X_mm: float = Field(default=500.0, description="X axis max acceleration in mm/s²")
    MAX_ACCELERATION_Y_mm: float = Field(default=500.0, description="Y axis max acceleration in mm/s²")
    MAX_ACCELERATION_Z_mm: float = Field(default=100.0, description="Z axis max acceleration in mm/s²")
    MAX_ACCELERATION_W_mm: float = Field(default=300.0, description="W axis max acceleration in mm/s²")
    
    # Stabilization times (ms) - hand-tuned values
    SCAN_STABILIZATION_TIME_MS_X: float = Field(default=160.0, description="X axis stabilization time in ms")
    SCAN_STABILIZATION_TIME_MS_Y: float = Field(default=160.0, description="Y axis stabilization time in ms")
    SCAN_STABILIZATION_TIME_MS_Z: float = Field(default=20.0, description="Z axis stabilization time in ms")
    
    # Limit switch polarity (0=ACTIVE_LOW, 1=ACTIVE_HIGH, 2=DISABLED) - hand-tuned values
    X_HOME_SWITCH_POLARITY: int = Field(default=1, description="X home switch polarity")
    Y_HOME_SWITCH_POLARITY: int = Field(default=1, description="Y home switch polarity")
    Z_HOME_SWITCH_POLARITY: int = Field(default=0, description="Z home switch polarity")
    
    # Filter wheel specific configuration - hand-tuned values
    FILTERWHEEL_MAX_INDEX: int = Field(default=8, description="Maximum filter wheel position")
    FILTERWHEEL_MIN_INDEX: int = Field(default=1, description="Minimum filter wheel position") 
    FILTERWHEEL_OFFSET_MM: float = Field(default=0.008, description="Filter wheel offset after homing in mm")
    
    # Computed properties for position sign (usually same as movement sign)
    @computed_field
    @property
    def STAGE_POS_SIGN_X(self) -> int:
        return self.STAGE_MOVEMENT_SIGN_X
        
    @computed_field  
    @property
    def STAGE_POS_SIGN_Y(self) -> int:
        return self.STAGE_MOVEMENT_SIGN_Y
        
    @computed_field
    @property
    def STAGE_POS_SIGN_Z(self) -> int:
        return self.STAGE_MOVEMENT_SIGN_Z
        
    @computed_field
    @property
    def STAGE_POS_SIGN_THETA(self) -> int:
        return self.STAGE_MOVEMENT_SIGN_THETA
    
    # Computed step size calculations
    @computed_field
    @property 
    def mm_per_ustep_x(self) -> float:
        """Calculate mm per microstep for X axis"""
        return self.SCREW_PITCH_X_MM / (self.MICROSTEPPING_DEFAULT_X * self.FULLSTEPS_PER_REV_X)
    
    @computed_field
    @property
    def mm_per_ustep_y(self) -> float:
        """Calculate mm per microstep for Y axis"""
        return self.SCREW_PITCH_Y_MM / (self.MICROSTEPPING_DEFAULT_Y * self.FULLSTEPS_PER_REV_Y)
    
    @computed_field
    @property
    def mm_per_ustep_z(self) -> float:
        """Calculate mm per microstep for Z axis"""
        return self.SCREW_PITCH_Z_MM / (self.MICROSTEPPING_DEFAULT_Z * self.FULLSTEPS_PER_REV_Z)
    
    @computed_field
    @property 
    def mm_per_ustep_w(self) -> float:
        """Calculate mm per microstep for W axis"""
        return self.SCREW_PITCH_W_MM / (self.MICROSTEPPING_DEFAULT_W * self.FULLSTEPS_PER_REV_W)
    
    def mm_to_ustep_x(self, value_mm: float) -> int:
        """Convert mm to microsteps for X axis"""
        if self.USE_ENCODER_X:
            return int(value_mm / (self.ENCODER_POS_SIGN_X * self.ENCODER_STEP_SIZE_X_MM))
        else:
            return int(value_mm / (self.STAGE_POS_SIGN_X * self.mm_per_ustep_x))
    
    def mm_to_ustep_y(self, value_mm: float) -> int:
        """Convert mm to microsteps for Y axis"""
        if self.USE_ENCODER_Y:
            return int(value_mm / (self.ENCODER_POS_SIGN_Y * self.ENCODER_STEP_SIZE_Y_MM))
        else:
            return int(value_mm / (self.STAGE_POS_SIGN_Y * self.mm_per_ustep_y))
    
    def mm_to_ustep_z(self, value_mm: float) -> int:
        """Convert mm to microsteps for Z axis"""
        if self.USE_ENCODER_Z:
            return int(value_mm / (self.ENCODER_POS_SIGN_Z * self.ENCODER_STEP_SIZE_Z_MM))
        else:
            return int(value_mm / (self.STAGE_POS_SIGN_Z * self.mm_per_ustep_z))
    
    def mm_to_ustep_w(self, value_mm: float) -> int:
        """Convert mm to microsteps for W axis"""
        return int(value_mm / (self.STAGE_MOVEMENT_SIGN_W * self.mm_per_ustep_w))


# Available configuration profiles
AVAILABLE_PROFILES = {
    "default": lambda: FirmwareConfig(),  # Use current hand-tuned default values
    
    "hcs_v2": lambda: FirmwareConfig(
        # Only parameters that differ from default
        MICROSTEPPING_DEFAULT_X=32,      # vs 256 default
        MICROSTEPPING_DEFAULT_Y=32,      # vs 256 default  
        MICROSTEPPING_DEFAULT_Z=32,      # vs 256 default
        MAX_VELOCITY_X_mm=30.0,          # vs 40.0 default
        MAX_VELOCITY_Y_mm=30.0,          # vs 40.0 default
        MAX_VELOCITY_Z_mm=3.8,           # vs 2.0 default
        SCAN_STABILIZATION_TIME_MS_X=25.0,  # vs 160.0 default
        SCAN_STABILIZATION_TIME_MS_Y=25.0,  # vs 160.0 default
    ),
    
    "squid_plus": lambda: FirmwareConfig(
        # Only parameters that differ from default
        MICROSTEPPING_DEFAULT_X=16,      # vs 256 default
        MICROSTEPPING_DEFAULT_Y=16,      # vs 256 default
        MICROSTEPPING_DEFAULT_Z=16,      # vs 256 default
        MAX_VELOCITY_X_mm=30.0,          # vs 40.0 default
        MAX_VELOCITY_Y_mm=30.0,          # vs 40.0 default
        MAX_VELOCITY_Z_mm=3.8,           # vs 2.0 default
        SCAN_STABILIZATION_TIME_MS_X=25.0,  # vs 160.0 default
        SCAN_STABILIZATION_TIME_MS_Y=25.0,  # vs 160.0 default
    ),
}

# Global configuration instance
_global_firmware_config: Optional[FirmwareConfig] = None


def get_firmware_config() -> FirmwareConfig:
    """Get the global firmware configuration instance, loading from environment on first call"""
    global _global_firmware_config
    if _global_firmware_config is None:
        # Check for profile selection via environment variable
        profile_name = os.getenv("SEAFRONT_FIRMWARE_PROFILE", "default")
        
        if profile_name not in AVAILABLE_PROFILES:
            available = ", ".join(AVAILABLE_PROFILES.keys())
            raise ValueError(
                f"Unknown firmware profile '{profile_name}' specified in SEAFRONT_FIRMWARE_PROFILE environment variable. "
                f"Available profiles: {available}"
            )
        
        _global_firmware_config = AVAILABLE_PROFILES[profile_name]()
        print(f"Loaded firmware profile: '{profile_name}'")
        
    return _global_firmware_config


def set_firmware_config(config: FirmwareConfig) -> None:
    """Set the global firmware configuration instance"""
    global _global_firmware_config
    _global_firmware_config = config


def get_available_profiles() -> list[str]:
    """Get list of available profile names"""
    return list(AVAILABLE_PROFILES.keys())