# this code is written on top of the firmware specifications in github.com/hongquanli/octopi-research

import asyncio
import dataclasses
import threading
import time
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps

import crc
import numpy as np
import serial
import serial.tools.list_ports
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from seafront.hardware.adapter import Position as AdapterPosition
from seafront.hardware.firmware_config import get_firmware_config, FirmwareConfig
from seafront.logger import logger

# Global firmware configuration instance
firmware_config = get_firmware_config()


@dataclass
class MicrocontrollerTimeoutInfo:
    await_move: bool
    "if the currently awaited command is a move command"
    move_done: bool
    "if a move command is awaited, indicates if the movement has finished"
    target_cmd_id: int
    "id of the awaited command"
    current_cmd_id: int
    "id of the currently processed command on the microcontroller"


@dataclass
class MicrocontrollerTimeout(BaseException):
    info: MicrocontrollerTimeoutInfo | None = None


def intFromPayload(payload, start_index, num_bytes):
    ret = 0
    for i in range(num_bytes):
        ret += payload[start_index + i] << 8 * (num_bytes - 1 - i)

    return ret


def twos_complement(v, num_bytes):
    THRESHOLD = 2 ** (8 * num_bytes)
    v = int(v)
    if v >= 0:
        payload = v
    else:
        payload = THRESHOLD + v  # find two's complement
    return payload


def twos_complement_rev(payload, num_bytes):
    THRESHOLD = 2 ** (8 * num_bytes)
    if payload <= THRESHOLD / 2:
        v = payload
    else:
        v = payload - THRESHOLD
    return v


# alias SerialDeviceInfo for annotations (it's actually a platform
# depedent type that is difficult to get right otherwise)
SerialDeviceInfo = tp.Any


class HOME_OR_ZERO:
    HOME_POSITIVE = 0  # motor moves along the positive direction (MCU coordinates)
    HOME_NEGATIVE = 1  # motor moves along the negative direction (MCU coordinates)  
    ZERO = 2


class LIMIT_CODE:
    X_POSITIVE: int = 0
    X_NEGATIVE: int = 1
    Y_POSITIVE: int = 2
    Y_NEGATIVE: int = 3
    Z_POSITIVE: int = 4
    Z_NEGATIVE: int = 5


class LIMIT_SWITCH_POLARITY:
    ACTIVE_LOW: int = 0
    ACTIVE_HIGH: int = 1
    DISABLED: int = 2


# Global firmware configuration instance - this replaces the old static FirmwareDefinitions class
firmware_config = get_firmware_config()


@dataclass(init=False)
class MicrocontrollerStatusPackage:
    """
    - command ID (1 byte)
    - execution status (1 byte)
    - X pos (4 bytes)
    - Y pos (4 bytes)
    - Z pos (4 bytes)
    - Theta (4 bytes)
    - buttons and switches (1 byte)
    - reserved (4 bytes)
    - CRC (1 byte)
    """

    last_cmd_id: int
    exec_status: int
    """
        COMPLETED_WITHOUT_ERRORS = 0
        IN_PROGRESS = 1
        CMD_CHECKSUM_ERROR = 2
        CMD_INVALID = 3
        CMD_EXECUTION_ERROR = 4
    """

    x_pos_usteps: int
    y_pos_usteps: int
    z_pos_usteps: int
    buttons_and_switches: int
    crc: int

    def __init__(self, packet: bytes | list[int]):
        self.last_cmd_id: int = packet[0]
        self.exec_status: int = packet[1]
        self.x_pos_usteps: int = (
            twos_complement_rev(intFromPayload(packet, 2, 4), 4)
            * firmware_config.STAGE_MOVEMENT_SIGN_X
        )
        self.y_pos_usteps: int = (
            twos_complement_rev(intFromPayload(packet, 6, 4), 4)
            * firmware_config.STAGE_MOVEMENT_SIGN_Y
        )
        self.z_pos_usteps: int = (
            twos_complement_rev(intFromPayload(packet, 10, 4), 4)
            * firmware_config.STAGE_MOVEMENT_SIGN_Z
        )
        # 4 bytes from theta ignored (index 14-17)
        self.buttons_and_switches: int = packet[18]
        # 4 bytes reserved (index 19-22)
        self.crc: int = packet[23]

    @property
    def pos(self) -> "Position":
        return Position(
            x_usteps=self.x_pos_usteps,
            y_usteps=self.y_pos_usteps,
            z_usteps=self.z_pos_usteps,
        )

    def __str__(self):
        s = ", ".join(
            [f"{field.name}={getattr(self, field.name)!r}" for field in dataclasses.fields(self)]
            + [
                f"x_pos_mm={self.pos.x_pos_mm}",
                f"y_pos_mm={self.pos.y_pos_mm}",
                f"z_pos_mm={self.pos.z_pos_mm}",
            ]
        )
        return f"{type(self).__name__}({s})"


class CommandName(int, Enum):
    MOVE_X = 0
    MOVE_Y = 1
    MOVE_Z = 2
    MOVE_THETA = 3
    MOVE_W = 4
    HOME_OR_ZERO = 5
    MOVETO_X = 6
    MOVETO_Y = 7
    MOVETO_Z = 8
    SET_LIM = 9
    TURN_ON_ILLUMINATION = 10
    TURN_OFF_ILLUMINATION = 11
    SET_ILLUMINATION = 12
    SET_ILLUMINATION_LED_MATRIX = 13
    ACK_JOYSTICK_BUTTON_PRESSED = 14
    ANALOG_WRITE_ONBOARD_DAC = 15
    SET_LIM_SWITCH_POLARITY = 20
    CONFIGURE_STEPPER_DRIVER = 21
    SET_MAX_VELOCITY_ACCELERATION = 22
    SET_LEAD_SCREW_PITCH = 23
    SET_OFFSET_VELOCITY = 24
    SEND_HARDWARE_TRIGGER = 30
    SET_STROBE_DELAY = 31
    SET_PIN_LEVEL = 41
    INITFILTERWHEEL = 253
    INITIALIZE = 254
    RESET = 255


class MCU_PINS:
    PWM1 = 5
    PWM2 = 4
    PWM3 = 22
    PWM4 = 3
    PWM5 = 23
    PWM6 = 2
    PWM7 = 1
    PWM9 = 6
    PWM10 = 7
    PWM11 = 8
    PWM12 = 9
    PWM13 = 10
    PWM14 = 15
    PWM15 = 24
    PWM16 = 25
    AF_LASER = 15


class ILLUMINATION_CODE(int, Enum):
    ILLUMINATION_SOURCE_LED_ARRAY_FULL = 0
    ILLUMINATION_SOURCE_LED_ARRAY_LEFT_HALF = 1
    ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_HALF = 2

    ILLUMINATION_SOURCE_LED_ARRAY_LEFTB_RIGHTR = 3
    ILLUMINATION_SOURCE_LED_ARRAY_LOW_NA = 4
    ILLUMINATION_SOURCE_LED_ARRAY_LEFT_DOT = 5
    ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_DOT = 6

    ILLUMINATION_SOURCE_LED_EXTERNAL_FET = 20

    ILLUMINATION_SOURCE_FLUOSLOT11 = 11
    ILLUMINATION_SOURCE_FLUOSLOT12 = 12
    ILLUMINATION_SOURCE_FLUOSLOT13 = 13
    ILLUMINATION_SOURCE_FLUOSLOT14 = 14
    ILLUMINATION_SOURCE_FLUOSLOT15 = 15

    @property
    def is_led_matrix(self) -> bool:
        return self.value <= 6

    @staticmethod
    def from_slot(slot: int) -> "ILLUMINATION_CODE":
        """
        Convert a hardware light source ID to ILLUMINATION_CODE enum for type safety.
        
        The slot parameter corresponds to the physical hardware light source identifier
        that the microcontroller uses to control illumination. This method provides
        type-safe mapping from integer slot IDs to the ILLUMINATION_CODE enum.
        
        Args:
            slot: Hardware light source ID (e.g. 11 for 405nm laser, 0 for LED array full)
            
        Returns:
            ILLUMINATION_CODE enum value corresponding to the hardware slot
            
        Raises:
            ValueError: If slot is not a valid ILLUMINATION_CODE value
        """
        try:
            return ILLUMINATION_CODE(slot)
        except ValueError:
            raise ValueError(f"Invalid illumination source slot: {slot}")


@dataclass
class Position:
    x_usteps: int
    y_usteps: int
    z_usteps: int

    @property
    def x_pos_mm(self):
        return firmware_config.mm_per_ustep_x * self.x_usteps

    @property
    def y_pos_mm(self):
        return firmware_config.mm_per_ustep_y * self.y_usteps

    @property
    def z_pos_mm(self):
        return firmware_config.mm_per_ustep_z * self.z_usteps

    @property
    def pos(self) -> AdapterPosition:
        return AdapterPosition(
            x_pos_mm=self.x_pos_mm,
            y_pos_mm=self.y_pos_mm,
            z_pos_mm=self.z_pos_mm,
        )


class Command:
    def __init__(self):
        self.bytes = bytearray(firmware_config.COMMAND_PACKET_LENGTH)
        self.wait_for_completion = True

        self.is_move_cmd = False

    def __getitem__(self, i: int):
        return self.bytes[i]

    def __setitem__(self, i: int, newval: int):
        self.bytes[i] = newval

    def __len__(self):
        return len(self.bytes)

    def set_id(self, id: int):
        self[0] = id

    @staticmethod
    def reset() -> "Command":
        ret = Command()
        ret[1] = CommandName.RESET.value

        return ret

    @staticmethod
    def initialize() -> "Command":
        ret = Command()
        ret[1] = CommandName.INITIALIZE.value

        return ret

    @staticmethod
    def set_leadscrew_pitch(axis: tp.Literal["x", "y", "z"]) -> "Command":
        ret = Command()
        ret[1] = CommandName.SET_LEAD_SCREW_PITCH.value

        axis_int = None
        screw_pitch_mm = None
        match axis:
            case "x":
                axis_int = firmware_config.AXIS_X
                screw_pitch_mm = firmware_config.SCREW_PITCH_X_MM
            case "y":
                axis_int = firmware_config.AXIS_Y
                screw_pitch_mm = firmware_config.SCREW_PITCH_Y_MM
            case "z":
                axis_int = firmware_config.AXIS_Z
                screw_pitch_mm = firmware_config.SCREW_PITCH_Z_MM
            case _:
                raise RuntimeError("invalid axis " + axis)

        ret[2] = axis_int
        ret[3] = (int(screw_pitch_mm * 1e3) >> 8 * 1) & 0xFF
        ret[4] = (int(screw_pitch_mm * 1e3) >> 8 * 0) & 0xFF

        return ret

    @staticmethod
    def configure_motor_driver(axis: tp.Literal["x", "y", "z"]) -> "Command":
        ret = Command()
        ret[1] = CommandName.CONFIGURE_STEPPER_DRIVER.value

        axis_int = None
        microstepping_default = None
        motor_rms_current = None
        motor_i_hold = None
        match axis:
            case "x":
                axis_int = firmware_config.AXIS_X
                microstepping_default = firmware_config.MICROSTEPPING_DEFAULT_X
                motor_rms_current = firmware_config.X_MOTOR_RMS_CURRENT_mA
                motor_i_hold = firmware_config.X_MOTOR_I_HOLD
            case "y":
                axis_int = firmware_config.AXIS_Y
                microstepping_default = firmware_config.MICROSTEPPING_DEFAULT_Y
                motor_rms_current = firmware_config.Y_MOTOR_RMS_CURRENT_mA
                motor_i_hold = firmware_config.Y_MOTOR_I_HOLD
            case "z":
                axis_int = firmware_config.AXIS_Z
                microstepping_default = firmware_config.MICROSTEPPING_DEFAULT_Z
                motor_rms_current = firmware_config.Z_MOTOR_RMS_CURRENT_mA
                motor_i_hold = firmware_config.Z_MOTOR_I_HOLD
            case _:
                raise RuntimeError("invalid axis " + axis)

        ret[2] = axis_int
        ret[3] = max(min(microstepping_default, 255), 1)
        ret[4] = (motor_rms_current >> 8 * 1) & 0xFF
        ret[5] = (motor_rms_current >> 8 * 0) & 0xFF
        ret[6] = int(motor_i_hold * 255)

        return ret

    @staticmethod
    def set_max_velocity_acceleration(axis: tp.Literal["x", "y", "z"]) -> "Command":
        ret = Command()
        ret[1] = CommandName.SET_MAX_VELOCITY_ACCELERATION.value

        axis_int = None
        max_vel_mm = None
        max_acc_mm = None
        match axis:
            case "x":
                axis_int = firmware_config.AXIS_X
                max_vel_mm = firmware_config.MAX_VELOCITY_X_mm
                max_acc_mm = firmware_config.MAX_ACCELERATION_X_mm
            case "y":
                axis_int = firmware_config.AXIS_Y
                max_vel_mm = firmware_config.MAX_VELOCITY_Y_mm
                max_acc_mm = firmware_config.MAX_ACCELERATION_Y_mm
            case "z":
                axis_int = firmware_config.AXIS_Z
                max_vel_mm = firmware_config.MAX_VELOCITY_Z_mm
                max_acc_mm = firmware_config.MAX_ACCELERATION_Z_mm
            case _:
                raise RuntimeError("invalid axis " + axis)

        ret[2] = axis_int
        ret[3] = (int(max_vel_mm * 100) >> 8 * 1) & 0xFF
        ret[4] = (int(max_vel_mm * 100) >> 8 * 0) & 0xFF
        ret[5] = (int(max_acc_mm * 10) >> 8 * 1) & 0xFF
        ret[6] = (int(max_acc_mm * 10) >> 8 * 0) & 0xFF

        return ret

    @staticmethod
    def set_limit_switch_polarity(axis: tp.Literal["x", "y", "z"]) -> "Command":
        ret = Command()

        ret[1] = CommandName.SET_LIM_SWITCH_POLARITY.value
        match axis:
            case "x":
                ret[2] = firmware_config.AXIS_X
                ret[3] = firmware_config.X_HOME_SWITCH_POLARITY
            case "y":
                ret[2] = firmware_config.AXIS_Y
                ret[3] = firmware_config.Y_HOME_SWITCH_POLARITY
            case "z":
                ret[2] = firmware_config.AXIS_Z
                ret[3] = firmware_config.Z_HOME_SWITCH_POLARITY

        return ret

    @staticmethod
    def configure_actuators() -> list["Command"]:
        rets = []

        # set lead screw pitch
        rets.append(Command.set_leadscrew_pitch("x"))
        rets.append(Command.set_leadscrew_pitch("y"))
        rets.append(Command.set_leadscrew_pitch("z"))

        # stepper driver (microstepping,rms current and I_hold)
        rets.append(Command.configure_motor_driver("x"))
        rets.append(Command.configure_motor_driver("y"))
        rets.append(Command.configure_motor_driver("z"))

        # max velocity and acceleration
        rets.append(Command.set_max_velocity_acceleration("x"))
        rets.append(Command.set_max_velocity_acceleration("y"))
        rets.append(Command.set_max_velocity_acceleration("z"))

        # homing direction
        rets.append(Command.set_limit_switch_polarity("x"))
        rets.append(Command.set_limit_switch_polarity("y"))
        rets.append(Command.set_limit_switch_polarity("z"))

        return rets

    @staticmethod
    def home(direction: tp.Literal["x", "y", "z", "w"]) -> "Command":
        ret = Command()
        ret[1] = CommandName.HOME_OR_ZERO.value
        match direction:
            case "x":
                ret[2] = firmware_config.AXIS_X
                # "move backward" (1?) if SIGN is 1, "move forward" (0?) if SIGN is -1
                ret[3] = int((firmware_config.STAGE_MOVEMENT_SIGN_X + 1) / 2)
            case "y":
                ret[2] = firmware_config.AXIS_Y
                # "move backward" (1?) if SIGN is 1, "move forward" (0?) if SIGN is -1
                ret[3] = int((firmware_config.STAGE_MOVEMENT_SIGN_Y + 1) / 2)
            case "z":
                ret[2] = firmware_config.AXIS_Z
                # "move backward" (1?) if SIGN is 1, "move forward" (0?) if SIGN is -1
                ret[3] = int((firmware_config.STAGE_MOVEMENT_SIGN_Z + 1) / 2)
            case "w":
                ret[2] = firmware_config.AXIS_W
                # Use default W homing direction based on STAGE_MOVEMENT_SIGN_W
                ret[3] = int((firmware_config.STAGE_MOVEMENT_SIGN_W + 1) / 2)
            case _:
                raise RuntimeError("invalid direction " + direction)

        ret.is_move_cmd = True

        return ret

    @staticmethod
    def move_by_mm(direction: tp.Literal["x", "y", "z"], distance_mm: float) -> list["Command"]:
        num_usteps = None
        command_name = None
        match direction:
            case "x":
                num_usteps = firmware_config.mm_to_ustep_x(distance_mm)
                command_name = CommandName.MOVE_X.value
            case "y":
                num_usteps = firmware_config.mm_to_ustep_y(distance_mm)
                command_name = CommandName.MOVE_Y.value
            case "z":
                num_usteps = firmware_config.mm_to_ustep_z(distance_mm)
                command_name = CommandName.MOVE_Z.value
            case _:
                raise RuntimeError("invalid direction " + direction)

        rets = []
        usteps_remaining = num_usteps
        MAX_USTEPS = 2**31 - 1
        while np.abs(usteps_remaining) > 0:
            partial_usteps = max(min(usteps_remaining, MAX_USTEPS), -MAX_USTEPS)
            usteps_remaining -= partial_usteps

            partial_usteps = twos_complement(partial_usteps, 4)

            ret = Command()

            ret.is_move_cmd = True

            ret[1] = command_name
            ret[2] = (partial_usteps >> (8 * 3)) & 0xFF
            ret[3] = (partial_usteps >> (8 * 2)) & 0xFF
            ret[4] = (partial_usteps >> (8 * 1)) & 0xFF
            ret[5] = (partial_usteps >> (8 * 0)) & 0xFF

            rets.append(ret)

        return rets

    @staticmethod
    def set_limit_mm(
        axis: tp.Literal["x", "y", "z"],
        coord: float,
        direction: tp.Literal["lower", "upper"],
    ) -> "Command":
        ret = Command()
        ret[1] = CommandName.SET_LIM.value

        match (axis, direction):
            case ("x", "upper"):
                usteps = firmware_config.mm_to_ustep_x(coord)
                limit_code = (
                    LIMIT_CODE.X_POSITIVE
                    if firmware_config.STAGE_MOVEMENT_SIGN_X > 0
                    else LIMIT_CODE.X_NEGATIVE
                )
            case ("x", "lower"):
                usteps = firmware_config.mm_to_ustep_x(coord)
                limit_code = (
                    LIMIT_CODE.X_POSITIVE
                    if firmware_config.STAGE_MOVEMENT_SIGN_X < 0
                    else LIMIT_CODE.X_NEGATIVE
                )
            case ("y", "upper"):
                usteps = firmware_config.mm_to_ustep_y(coord)
                limit_code = (
                    LIMIT_CODE.Y_POSITIVE
                    if firmware_config.STAGE_MOVEMENT_SIGN_Y > 0
                    else LIMIT_CODE.Y_NEGATIVE
                )
            case ("y", "lower"):
                usteps = firmware_config.mm_to_ustep_y(coord)
                limit_code = (
                    LIMIT_CODE.Y_POSITIVE
                    if firmware_config.STAGE_MOVEMENT_SIGN_Y < 0
                    else LIMIT_CODE.Y_NEGATIVE
                )
            case ("z", "upper"):
                usteps = firmware_config.mm_to_ustep_z(coord)
                limit_code = (
                    LIMIT_CODE.Z_POSITIVE
                    if firmware_config.STAGE_MOVEMENT_SIGN_Z > 0
                    else LIMIT_CODE.Z_NEGATIVE
                )
            case ("z", "lower"):
                usteps = firmware_config.mm_to_ustep_z(coord)
                limit_code = (
                    LIMIT_CODE.Z_POSITIVE
                    if firmware_config.STAGE_MOVEMENT_SIGN_Z < 0
                    else LIMIT_CODE.Z_NEGATIVE
                )
            case _:
                raise ValueError(f"unsupported axis {axis}")

        ret[2] = limit_code
        payload = twos_complement(usteps, 4)
        ret[3] = (payload >> 8 * 3) & 0xFF
        ret[4] = (payload >> 8 * 2) & 0xFF
        ret[5] = (payload >> 8 * 1) & 0xFF
        ret[6] = (payload >> 8 * 0) & 0xFF
        return ret

    @staticmethod
    def set_zero(axis: tp.Literal["x", "y", "z"]) -> "Command":
        """
        set current position on axis to 0
        """

        ret = Command()
        ret[1] = CommandName.HOME_OR_ZERO.value

        match axis:
            case "x":
                ret[2] = firmware_config.AXIS_X
            case "y":
                ret[2] = firmware_config.AXIS_Y
            case "z":
                ret[2] = firmware_config.AXIS_Z
            case _:
                raise RuntimeError("invalid axis " + axis)

        ret[3] = HOME_OR_ZERO.ZERO

        return ret

    @staticmethod
    def move_to_mm(axis: tp.Literal["x", "y", "z"], coord_mm: float) -> "Command":
        """
        move to z is currently TODO, and potentially dangerous. this command does not indicate completion, ever! (but it does move to the target position..)
        """

        ret = Command()

        axis_cmd = None
        move_usteps = None

        match axis:
            case "x":
                axis_cmd = CommandName.MOVETO_X.value
                move_usteps = firmware_config.mm_to_ustep_x(coord_mm)
            case "y":
                axis_cmd = CommandName.MOVETO_Y.value
                move_usteps = firmware_config.mm_to_ustep_y(coord_mm)
            case "z":
                axis_cmd = CommandName.MOVETO_Z.value
                move_usteps = firmware_config.mm_to_ustep_z(coord_mm)
            case _:
                raise RuntimeError("invalid axis " + axis)

        move_usteps = twos_complement(move_usteps, 4)

        ret[1] = axis_cmd
        ret[2] = (move_usteps >> (8 * 3)) & 0xFF
        ret[3] = (move_usteps >> (8 * 2)) & 0xFF
        ret[4] = (move_usteps >> (8 * 1)) & 0xFF
        ret[5] = (move_usteps >> (8 * 0)) & 0xFF

        ret.is_move_cmd = True

        return ret

    @staticmethod
    def illumination_begin(
        illumination_source: ILLUMINATION_CODE,
        intensity_percent: float,
        led_color_r: float = 1.0,
        led_color_g: float = 1.0,
        led_color_b: float = 1.0,
    ) -> list["Command"]:
        """
        turn on illumination source, and set intensity

        intensity_percent: should be in range [0;100] - will be internally clamped to [0;100] for safety either way

        led_color_[r|g|b]: color component used for matrix led, must be in range (and will be clamped to) [0;1]
        """

        cmds = []

        INTENSITY_INT_MAX = 0xFFFF
        intensity_int = int(intensity_percent / 100 * INTENSITY_INT_MAX)
        intensity_clamped = max(min(intensity_int, INTENSITY_INT_MAX), 0)

        cmd = Command()
        if illumination_source.is_led_matrix:
            cmd[1] = CommandName.SET_ILLUMINATION_LED_MATRIX.value
            cmd[2] = illumination_source.value
            # clamp rgb*255 to [0;255]
            cmd[3] = max(0, min(int(led_color_r * 255), 255))
            cmd[4] = max(0, min(int(led_color_g * 255), 255))
            cmd[5] = max(0, min(int(led_color_b * 255), 255))
        else:
            cmd[1] = CommandName.SET_ILLUMINATION.value
            cmd[2] = illumination_source.value
            cmd[3] = (intensity_clamped >> 8 * 1) & 0xFF
            cmd[4] = (intensity_clamped >> 8 * 0) & 0xFF
        cmds.append(cmd)

        cmd = Command()
        cmd[1] = CommandName.TURN_ON_ILLUMINATION.value
        cmds.append(cmd)

        return cmds

    @staticmethod
    def illumination_end(
        illumination_source: ILLUMINATION_CODE | None = None,
    ) -> list["Command"]:
        cmds = []
        if illumination_source is not None:
            cmd = Command()
            cmd[1] = CommandName.SET_ILLUMINATION.value
            cmd[2] = illumination_source.value
            # other bytes indicating strength/color are zero
            cmds.append(cmd)

        cmd = Command()
        cmd[1] = CommandName.TURN_OFF_ILLUMINATION.value
        cmds.append(cmd)

        return cmds

    @staticmethod
    def set_pin_level(pin: int, level: int) -> "Command":
        """
        sets pin level (the interpretation of this is up to the firmware)

        pin: pin number
        level: 0 or 1
        """

        assert level in [0, 1], f"invalid level {level}"

        cmd = Command()
        cmd[1] = CommandName.SET_PIN_LEVEL.value
        cmd[2] = pin
        cmd[3] = level
        return cmd

    @staticmethod
    def af_laser_illum_begin() -> "Command":
        return Command.set_pin_level(pin=MCU_PINS.AF_LASER, level=1)

    @staticmethod
    def af_laser_illum_end() -> "Command":
        return Command.set_pin_level(pin=MCU_PINS.AF_LASER, level=0)
    
    @staticmethod
    def filter_wheel_init() -> "Command":
        cmd = Command()
        cmd[1] = CommandName.INITFILTERWHEEL.value
        return cmd

    @staticmethod
    def move_w_usteps(usteps: int) -> list["Command"]:
        """
        Move filter wheel by the specified number of microsteps
        Similar to move_by_mm but for W axis (filter wheel)
        """
        rets = []
        usteps_remaining = usteps
        MAX_USTEPS = 2**31 - 1
        while np.abs(usteps_remaining) > 0:
            partial_usteps = max(min(usteps_remaining, MAX_USTEPS), -MAX_USTEPS)
            usteps_remaining -= partial_usteps

            partial_usteps = twos_complement(partial_usteps, 4)
            # partial_usteps=10

            ret = Command()
            ret.is_move_cmd = False

            ret[1] = CommandName.MOVE_W.value
            ret[2] = (partial_usteps >> (8 * 3)) & 0xFF
            ret[3] = (partial_usteps >> (8 * 2)) & 0xFF
            ret[4] = (partial_usteps >> (8 * 1)) & 0xFF
            ret[5] = (partial_usteps >> (8 * 0)) & 0xFF

            rets.append(ret)

        return rets


def microcontroller_exclusive(f):
    "ensure exclusive access to certain code sections"

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return f(self, *args, **kwargs)

    return wrapper


class Microcontroller(BaseModel):
    device_info: SerialDeviceInfo

    handle: serial.Serial | None = None
    illum: threading.RLock = Field(default_factory=threading.RLock)
    """ lock on illumination control """

    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    """ lock on hardware control """

    crc_calculator: crc.CrcCalculator = Field(
        default_factory=lambda: crc.CrcCalculator(crc.Crc8.CCITT, table_based=True)
    )

    last_command_id: int = -1
    """ because this is increment BEFORE it is returned, init to -1 -> first id assigned is 0 """

    terminate_reading_received_packet_thread: bool = False
    last_position: Position = Field(default_factory=lambda: Position(0, 0, 0))
    
    # Filter wheel state
    filter_wheel_position: int = Field(default=firmware_config.FILTERWHEEL_MIN_INDEX)

    baudrate: int = 2_000_000

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @contextmanager
    def locked(self, blocking: bool = True) -> "tp.Iterator[tp.Self|None]":
        "convenience function to lock self for multiple function calls. yields none is lock cannot be acquired"
        if self._lock.acquire(blocking=blocking):
            try:
                yield self
            finally:
                self._lock.release()
        else:
            yield None

    async def _wait_until_cmd_is_finished(
        self,
        cmd: "Command",
        _ignore_cmd_id: bool = False,
        timeout_s: float = 5.0,
    ):
        """
        _ignore_cmd_id is only for use with RESET

        may throw MicrocontrollerTimeout
        """
        last_position_x = 0
        last_position_y = 0
        last_position_z = 0
        timeoutinfo: None | MicrocontrollerTimeoutInfo = None

        async def read_packet(packet: MicrocontrollerStatusPackage) -> bool:
            nonlocal last_position_x
            nonlocal last_position_y
            nonlocal last_position_z
            nonlocal timeoutinfo

            latest_position_x = packet.x_pos_usteps
            latest_position_y = packet.y_pos_usteps
            latest_position_z = packet.z_pos_usteps

            timeoutinfo = MicrocontrollerTimeoutInfo(
                await_move=cmd.is_move_cmd,
                move_done=False,
                target_cmd_id=cmd[0],
                current_cmd_id=packet.last_cmd_id,
            )

            if cmd.is_move_cmd:
                # wait until stage has stopped moving (which the microcontroller has no direct knowledge of)
                movement_in_progress = (
                    last_position_x != latest_position_x
                    or last_position_y != latest_position_y
                    or last_position_z != latest_position_z
                )

                timeoutinfo.move_done = movement_in_progress

                if movement_in_progress:
                    last_position_x = latest_position_x
                    last_position_y = latest_position_y
                    last_position_z = latest_position_z
                    return False

            cmd_is_completed = (
                _ignore_cmd_id or (packet.last_cmd_id == cmd[0])
            ) and packet.exec_status == 0

            if cmd.is_move_cmd:
                # moves in Z sometimes do not indicate completion, even when moving to target position is done
                # so for those cases we fall back to checking if
                # 1) command packet has been executed by microcontroller (last command id matches)
                # 2) stage has stopped moving (checked above)

                cmd_name = CommandName(cmd[1])
                match cmd_name:
                    case CommandName.MOVETO_Z:
                        cmd_is_completed = packet.last_cmd_id == cmd[0]
                    case CommandName.MOVE_Z:
                        cmd_is_completed = packet.last_cmd_id == cmd[0]

            return cmd_is_completed

        if cmd.is_move_cmd:
            # wait short time for move commands to have engaged the motors to start moving
            # (moving takes way longer than this short delay, so we are waiting this time anyway)
            await asyncio.sleep(5e-3)

        try:
            await self._read_packets(until=read_packet, timeout_s=timeout_s)
        except MicrocontrollerTimeout as mte:
            logger.debug("_read_packets timed out")
            if mte.info is None:
                mte.info = timeoutinfo
            elif timeoutinfo is not None:
                logger.debug(f"did not overwrite timeoutinfo {mte.info} with {timeoutinfo}")
            raise mte

    async def _read_packets(
        self,
        until: tp.Callable[[MicrocontrollerStatusPackage], bool]
        | tp.Callable[[MicrocontrollerStatusPackage], tp.Coroutine[None, None, bool]],
        timeout_s: float = 5.0,
        MICROCONTROLLER_PACKET_RETRY_DELAY_S=0.4e-3,
    ):
        """
        read packets until 'until' returns True

        this is the central waiting function. other functions with wait-like behaviour rely on this function.

        will throw MicrocontrollerTimeout if 'until' condition has not been met within 'timeout_s'.

        params:
            until:
                called on each received package, must return True to continue this loop (can be sync or async, hence weird signature)
        """

        start_time = time.time()

        last_packet: MicrocontrollerStatusPackage | None = None

        self.terminate_reading_received_packet_thread = False
        while not self.terminate_reading_received_packet_thread:
            # wait to connect and receive data

            # time out at loop iteration start
            if (wait_time_s := (time.time() - start_time)) > timeout_s:
                logger.critical(
                    f"timed out after {wait_time_s:.3f} with {timeout_s=} {last_packet=}"
                )
                raise MicrocontrollerTimeout(info=None)

            if self.handle is None:
                logger.warning("self.handle is None ")
                await asyncio.sleep(MICROCONTROLLER_PACKET_RETRY_DELAY_S)
                continue

            if self.handle.in_waiting < firmware_config.READ_PACKET_LENGTH:
                await asyncio.sleep(MICROCONTROLLER_PACKET_RETRY_DELAY_S)
                continue

            # skip all bytes except those in the last package, since parsing takes too long
            # TODO make this fast enough to check all packet command ids, to not skip over the
            # result of a command that finishes quickly
            num_bytes_to_skip = (
                self.handle.in_waiting // firmware_config.READ_PACKET_LENGTH
            ) - 1
            if num_bytes_to_skip > 0:
                self.handle.read(num_bytes_to_skip * firmware_config.READ_PACKET_LENGTH)

            packet = MicrocontrollerStatusPackage(
                self.handle.read(firmware_config.READ_PACKET_LENGTH)
            )

            # save current position as last known position
            self.last_position = packet.pos

            until_res = until(packet)
            # the type checker does not understand inspect.iscoroutine, so we isinstance a bool
            if isinstance(until_res, bool):
                should_terminate: bool = until_res
            else:
                should_terminate: bool = await until_res

            self.terminate_reading_received_packet_thread = should_terminate

    async def get_last_position(self) -> Position:
        """
        get last known position of the stage

        this function works even if another command is currently running
        """

        if self._lock.acquire(blocking=False):
            try:
                # do not wait to read a package if there is currently no connection
                if self.handle is not None:
                    # internally updates the last known position
                    await self._read_packets(lambda p: True)

            except Exception as e:
                print(f"got error? {e}")
                raise e
            finally:
                self._lock.release()

        return self.last_position

    def _get_next_cmd_id(self) -> int:
        """
        generate id for next command, and return it

        just increment last command id by 1, and wrap to 0 after 255
        """

        self.last_command_id = (self.last_command_id + 1) % 256
        return self.last_command_id

    @microcontroller_exclusive
    async def send_cmd(self, cmd_in: tp.Union["Command", list["Command"]]):
        "send command for execution. waits for command to complete if command type requires awaiting."
        if isinstance(cmd_in, list):
            cmds = cmd_in
        else:
            cmds = [cmd_in]

        for cmd in cmds:
            # generate command id
            cmd.set_id(self._get_next_cmd_id())
            cmd[-1] = self.crc_calculator.calculate_checksum(cmd.bytes[:-1])

            # keep track of illumination lock
            acquired_illum_lock = False
            match cmd[1]:
                case CommandName.TURN_ON_ILLUMINATION.value:
                    # overlapping illumination can cause all sorts of issues
                    if not self.illum.acquire(blocking=False):
                        raise RuntimeError("illumination already on")

                    acquired_illum_lock = True

                case CommandName.TURN_OFF_ILLUMINATION.value:
                    # illumination is in undefined state upon startup
                    # and turning off does not damage anything, so we allow this
                    # (e.g. on startup commands can be issued to ensure all illumination is off)
                    if not self.illum.acquire(blocking=False):
                        # if we cannot acquire illumination, it is in use by another thread
                        raise RuntimeError("illumination currently controlled by another thread!")

                    acquired_illum_lock = True

            try:
                if self.handle is None:
                    raise RuntimeError("mc handle is None")

                self.handle.write(cmd.bytes)
                if cmd.wait_for_completion:
                    # make more than one attempt at completing a command
                    NUM_CMD_REPEATS_MAX = 5

                    for cmd_repeat_attempt in range(NUM_CMD_REPEATS_MAX):
                        logger.debug(
                            f"awaiting {CommandName(cmd[1])} the {cmd_repeat_attempt}th time"
                        )
                        try:
                            await self._wait_until_cmd_is_finished(
                                cmd,
                                # allow more time for a move command to finish
                                timeout_s=5.0 if cmd.is_move_cmd else 1.0,
                                # reset command also resets the command id to zero, so we need to ignore the command id then
                                _ignore_cmd_id=cmd[1] == Command.reset()[1],
                            )
                        except MicrocontrollerTimeout as mte:
                            logger.warning(
                                f"microcontroller timed out while waiting for a command to finish with info: {mte.info}"
                            )

                        break
            finally:
                if acquired_illum_lock:
                    self.illum.release()

    @microcontroller_exclusive
    def open(self):
        "open connection to device"
        self.handle = serial.Serial(self.device_info.device, self.baudrate)

    @microcontroller_exclusive
    def close(self):
        "close connection to device"

        logger.debug("microcontroller - closing")

        if self.handle is not None:
            self.handle.close()
            self.handle = None

        logger.debug("microcontroller - closed")

    @microcontroller_exclusive
    async def filter_wheel_set_position(self, position: int):
        """
        Set the filter wheel to the specified position.
        Position should be between FILTERWHEEL_MIN_INDEX and FILTERWHEEL_MAX_INDEX.
        This is the main public interface for filter wheel control.
        """
        if not (firmware_config.FILTERWHEEL_MIN_INDEX <= position <= firmware_config.FILTERWHEEL_MAX_INDEX):
            raise ValueError(f"Position {position} out of range [{firmware_config.FILTERWHEEL_MIN_INDEX}, {firmware_config.FILTERWHEEL_MAX_INDEX}]")
            
        if position != self.filter_wheel_position:
            # Calculate movement needed
            delta_positions = position - self.filter_wheel_position
            distance_per_position = firmware_config.SCREW_PITCH_W_MM / (
                firmware_config.FILTERWHEEL_MAX_INDEX - firmware_config.FILTERWHEEL_MIN_INDEX + 1
            )
            distance_mm = delta_positions * distance_per_position
            
            # Convert to microsteps and move
            usteps = firmware_config.mm_to_ustep_w(distance_mm)
            move_commands = Command.move_w_usteps(usteps)
            
            await self.send_cmd(move_commands)
            
            # Update internal position tracking
            self.filter_wheel_position = position

    @microcontroller_exclusive  
    async def filter_wheel_init(self):
        """Initialize the filter wheel"""
        await self.send_cmd(Command.filter_wheel_init())
        
    @microcontroller_exclusive
    async def filter_wheel_configure_actuator(self):
        """Configure the filter wheel (W axis) motor parameters before homing"""
        # Configure W axis leadscrew pitch
        cmd = Command()
        cmd[1] = CommandName.SET_LEAD_SCREW_PITCH.value
        cmd[2] = firmware_config.AXIS_W
        cmd[3] = (int(firmware_config.SCREW_PITCH_W_MM * 1e3) >> 8 * 1) & 0xFF
        cmd[4] = (int(firmware_config.SCREW_PITCH_W_MM * 1e3) >> 8 * 0) & 0xFF
        await self.send_cmd(cmd)

        # Configure W axis motor driver (microstepping and current)
        cmd = Command()
        cmd[1] = CommandName.CONFIGURE_STEPPER_DRIVER.value
        cmd[2] = firmware_config.AXIS_W
        cmd[3] = firmware_config.MICROSTEPPING_DEFAULT_W
        cmd[4] = (firmware_config.W_MOTOR_RMS_CURRENT_mA >> 8 * 1) & 0xFF
        cmd[5] = (firmware_config.W_MOTOR_RMS_CURRENT_mA >> 8 * 0) & 0xFF
        cmd[6] = int(firmware_config.W_MOTOR_I_HOLD * 255)
        await self.send_cmd(cmd)

        # Configure W axis max velocity and acceleration
        cmd = Command()
        cmd[1] = CommandName.SET_MAX_VELOCITY_ACCELERATION.value
        cmd[2] = firmware_config.AXIS_W
        cmd[3] = (int(firmware_config.MAX_VELOCITY_W_mm * 100) >> 8 * 1) & 0xFF
        cmd[4] = (int(firmware_config.MAX_VELOCITY_W_mm * 100) >> 8 * 0) & 0xFF
        cmd[5] = (int(firmware_config.MAX_ACCELERATION_W_mm * 10) >> 8 * 1) & 0xFF
        cmd[6] = (int(firmware_config.MAX_ACCELERATION_W_mm * 10) >> 8 * 0) & 0xFF
        await self.send_cmd(cmd)

    @microcontroller_exclusive
    async def filter_wheel_home(self):
        """
        Home the filter wheel to establish reference position.
        
        This performs the complete homing sequence similar to Squid:
        1. Home the W axis using limit switches
        2. Apply small offset to move away from limit 
        3. Reset position tracking to minimum index
        
        The send_cmd() method automatically waits for completion with appropriate timeout.
        """
        # Home the W axis - this moves to the physical limit switch
        # The home command is marked as a move command so send_cmd will wait for completion
        await self.send_cmd(Command.home("w"))
        
        # Apply small offset to move away from the limit switch 
        # This matches SQUID_FILTERWHEEL_OFFSET from the original Squid code
        offset_usteps = firmware_config.mm_to_ustep_w(firmware_config.FILTERWHEEL_OFFSET_MM)
        move_commands = Command.move_w_usteps(offset_usteps)
        await self.send_cmd(move_commands)
        
        # Reset position tracking to minimum index (position 1)
        self.filter_wheel_position = firmware_config.FILTERWHEEL_MIN_INDEX
        
    def filter_wheel_get_position(self) -> int:
        """Get the current filter wheel position (non-blocking)"""
        return self.filter_wheel_position

    @staticmethod
    def get_all() -> list["Microcontroller"]:
        "get all available devices"

        ret = []
        for p in serial.tools.list_ports.comports():
            if p.description == "Arduino Due":
                device_info = p
            elif p.manufacturer == "Teensyduino":
                device_info = p
            else:
                # we dont care about other devices
                continue

            ret.append(Microcontroller(device_info=device_info))

        return ret
