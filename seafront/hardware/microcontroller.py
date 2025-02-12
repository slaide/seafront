# this code is written on top of the firmware specifications in github.com/hongquanli/octopi-research

import time
import typing as tp
from enum import Enum
import threading
import dataclasses
from dataclasses import dataclass

import serial
import serial.tools.list_ports
import crc
import numpy as np

from .adapter import Position as AdapterPosition

def intFromPayload(payload,start_index,num_bytes):
    ret=0
    for i in range(num_bytes):
        ret+=payload[start_index+i]<<8*(num_bytes-1-i)

    return ret

def twos_complement(v,num_bytes):
    THRESHOLD=2**(8*num_bytes)
    v=int(v)
    if v >= 0:
        payload = v
    else:
        payload = THRESHOLD + v # find two's complement
    return payload

def twos_complement_rev(payload,num_bytes):
    THRESHOLD=2**(8*num_bytes)
    if payload <= THRESHOLD/2:
        v = payload
    else:
        v = payload - THRESHOLD
    return v

# alias SerialDeviceInfo for annotations (it's actually a platform
# depedent type that is difficult to get right otherwise)
SerialDeviceInfo=tp.Any

class HOME_OR_ZERO:
    ZERO = 2

class LIMIT_CODE:
    X_POSITIVE:int = 0
    X_NEGATIVE:int = 1
    Y_POSITIVE:int = 2
    Y_NEGATIVE:int = 3
    Z_POSITIVE:int = 4
    Z_NEGATIVE:int = 5

class LIMIT_SWITCH_POLARITY:
    ACTIVE_LOW:int = 0
    ACTIVE_HIGH:int = 1
    DISABLED:int = 2

class FirmwareDefinitions:
    READ_PACKET_LENGTH=24
    # 1 byte cmd id, 1 byte cmd name, 6 bytes for command arguments
    COMMAND_PACKET_LENGTH=8

    AXIS_X:int = 0
    AXIS_Y:int = 1
    AXIS_Z:int = 2
    AXIS_THETA:int = 3
    AXIS_XY:int = 4

    # TODO
    SCREW_PITCH_X_MM:float = 2.54
    SCREW_PITCH_Y_MM:float = 2.54
    # SCREW_PITCH_Z_MM=0.012*25.4 was written here at some point, not sure why.
    # the motor makes _the_ weird noise during homing when set to the latter term (=0.3048), instead of 0.3
    SCREW_PITCH_Z_MM:float = 0.3

    MICROSTEPPING_DEFAULT_X:int = 256
    MICROSTEPPING_DEFAULT_Y:int = 256
    MICROSTEPPING_DEFAULT_Z:int = 256
    MICROSTEPPING_DEFAULT_THETA:int = 256

    USE_ENCODER_X:bool = False
    USE_ENCODER_Y:bool = False
    USE_ENCODER_Z:bool = False
    USE_ENCODER_THETA:bool = False

    ENCODER_POS_SIGN_X:int = 1
    ENCODER_POS_SIGN_Y:int = 1
    ENCODER_POS_SIGN_Z:int = 1
    ENCODER_POS_SIGN_THETA:int = 1

    ENCODER_STEP_SIZE_X_MM:float = 100e-6
    ENCODER_STEP_SIZE_Y_MM:float = 100e-6
    ENCODER_STEP_SIZE_Z_MM:float = 100e-6
    ENCODER_STEP_SIZE_THETA:float = 1.0

    FULLSTEPS_PER_REV_X:int = 200
    FULLSTEPS_PER_REV_Y:int = 200
    FULLSTEPS_PER_REV_Z:int = 200
    FULLSTEPS_PER_REV_THETA:int = 200

    STAGE_MOVEMENT_SIGN_X:int = 1
    STAGE_MOVEMENT_SIGN_Y:int = 1
    STAGE_MOVEMENT_SIGN_Z:int = -1
    STAGE_MOVEMENT_SIGN_THETA:int = 1

    STAGE_POS_SIGN_X:int = STAGE_MOVEMENT_SIGN_X
    STAGE_POS_SIGN_Y:int = STAGE_MOVEMENT_SIGN_Y
    STAGE_POS_SIGN_Z:int = STAGE_MOVEMENT_SIGN_Z
    STAGE_POS_SIGN_THETA:int = STAGE_MOVEMENT_SIGN_THETA

    X_MOTOR_RMS_CURRENT_mA:int = 1000
    Y_MOTOR_RMS_CURRENT_mA:int = 1000
    Z_MOTOR_RMS_CURRENT_mA:int = 500

    # these 3 values must be in range [0.0;1.0]
    X_MOTOR_I_HOLD:float = 0.25
    Y_MOTOR_I_HOLD:float = 0.25
    Z_MOTOR_I_HOLD:float = 0.5

    MAX_VELOCITY_X_mm:float = 40.0
    MAX_VELOCITY_Y_mm:float = 40.0
    MAX_VELOCITY_Z_mm:float = 2.0

    MAX_ACCELERATION_X_mm:float = 500.0
    MAX_ACCELERATION_Y_mm:float = 500.0
    MAX_ACCELERATION_Z_mm:float = 100.0

    # end of actuator specific configurations

    SCAN_STABILIZATION_TIME_MS_X:float = 160.0
    SCAN_STABILIZATION_TIME_MS_Y:float = 160.0
    SCAN_STABILIZATION_TIME_MS_Z:float = 20.0

    # limit switch
    X_HOME_SWITCH_POLARITY:int = LIMIT_SWITCH_POLARITY.ACTIVE_HIGH
    Y_HOME_SWITCH_POLARITY:int = LIMIT_SWITCH_POLARITY.ACTIVE_HIGH
    Z_HOME_SWITCH_POLARITY:int = LIMIT_SWITCH_POLARITY.ACTIVE_LOW

    @staticmethod
    def mm_per_ustep_x()->float:
        return FirmwareDefinitions.SCREW_PITCH_X_MM/(FirmwareDefinitions.MICROSTEPPING_DEFAULT_X*FirmwareDefinitions.FULLSTEPS_PER_REV_X)
    @staticmethod
    def mm_per_ustep_y()->float:
        return FirmwareDefinitions.SCREW_PITCH_Y_MM/(FirmwareDefinitions.MICROSTEPPING_DEFAULT_Y*FirmwareDefinitions.FULLSTEPS_PER_REV_Y)
    @staticmethod
    def mm_per_ustep_z()->float:
        return FirmwareDefinitions.SCREW_PITCH_Z_MM/(FirmwareDefinitions.MICROSTEPPING_DEFAULT_Z*FirmwareDefinitions.FULLSTEPS_PER_REV_Z)
    
    @staticmethod
    def mm_to_ustep_x(value_mm:float)->int:
        if FirmwareDefinitions.USE_ENCODER_X:
            return int(value_mm/(FirmwareDefinitions.ENCODER_POS_SIGN_X*FirmwareDefinitions.ENCODER_STEP_SIZE_X_MM))
        else:
            return int(value_mm/(FirmwareDefinitions.STAGE_POS_SIGN_X*FirmwareDefinitions.mm_per_ustep_x()))
    @staticmethod
    def mm_to_ustep_y(value_mm:float)->int: 
        if FirmwareDefinitions.USE_ENCODER_Y:
            return int(value_mm/(FirmwareDefinitions.ENCODER_POS_SIGN_Y*FirmwareDefinitions.ENCODER_STEP_SIZE_Y_MM))
        else:
            return int(value_mm/(FirmwareDefinitions.STAGE_POS_SIGN_Y*FirmwareDefinitions.mm_per_ustep_y()))
    @staticmethod
    def mm_to_ustep_z(value_mm:float)->int:
        if FirmwareDefinitions.USE_ENCODER_Z:
            return int(value_mm/(FirmwareDefinitions.ENCODER_POS_SIGN_Z*FirmwareDefinitions.ENCODER_STEP_SIZE_Z_MM))
        else:
            return int(value_mm/(FirmwareDefinitions.STAGE_POS_SIGN_Z*FirmwareDefinitions.mm_per_ustep_z()))

@dataclass(init=False)
class MicrocontrollerStatusPackage:
    '''
    - command ID (1 byte)
    - execution status (1 byte)
    - X pos (4 bytes)
    - Y pos (4 bytes)
    - Z pos (4 bytes)
    - Theta (4 bytes)
    - buttons and switches (1 byte)
    - reserved (4 bytes)
    - CRC (1 byte)
    '''

    last_cmd_id:int
    exec_status:int
    x_pos_usteps:int
    y_pos_usteps:int
    z_pos_usteps:int
    buttons_and_switches:int
    crc:int

    def __init__(self,packet):
        self.last_cmd_id:int=packet[0]
        self.exec_status:int=packet[1]
        self.x_pos_usteps:int=twos_complement_rev(intFromPayload(packet,2,4),4)*FirmwareDefinitions.STAGE_MOVEMENT_SIGN_X
        self.y_pos_usteps:int=twos_complement_rev(intFromPayload(packet,6,4),4)*FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Y
        self.z_pos_usteps:int=twos_complement_rev(intFromPayload(packet,10,4),4)*FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Z
        # 4 bytes from theta ignored (index 14-17)
        self.buttons_and_switches:int=packet[18]
        # 4 bytes reserved (index 19-22)
        self.crc:int=packet[23]

    @property
    def pos(self)->"Position":
        return Position(x_usteps=self.x_pos_usteps,y_usteps=self.y_pos_usteps,z_usteps=self.z_pos_usteps)

    def __str__(self):
        s=", ".join(
            [f"{field.name}={getattr(self,field.name)!r}" for field in dataclasses.fields(self)]
            +[f"x_pos_mm={self.pos.x_pos_mm}", f"y_pos_mm={self.pos.y_pos_mm}", f"z_pos_mm={self.pos.z_pos_mm}"])
        return f"{type(self).__name__}({s})"

class CommandName(int,Enum):
    MOVE_X = 0
    MOVE_Y = 1
    MOVE_Z = 2
    MOVE_THETA = 3
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

class ILLUMINATION_CODE(int,Enum):
    ILLUMINATION_SOURCE_LED_ARRAY_FULL = 0
    ILLUMINATION_SOURCE_LED_ARRAY_LEFT_HALF = 1
    ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_HALF = 2

    ILLUMINATION_SOURCE_LED_ARRAY_LEFTB_RIGHTR = 3
    ILLUMINATION_SOURCE_LED_ARRAY_LOW_NA = 4
    ILLUMINATION_SOURCE_LED_ARRAY_LEFT_DOT = 5
    ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_DOT = 6

    ILLUMINATION_SOURCE_LED_EXTERNAL_FET = 20

    ILLUMINATION_SOURCE_405NM = 11
    ILLUMINATION_SOURCE_488NM = 12
    ILLUMINATION_SOURCE_638NM = 13
    ILLUMINATION_SOURCE_561NM = 14
    ILLUMINATION_SOURCE_730NM = 15

    @property
    def is_led_matrix(self)->bool:
        return self.value <= 6

    @staticmethod
    def from_handle(handle:str)->"ILLUMINATION_CODE":
        match handle:
            case "fluo405":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM
            case "fluo488":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_488NM
            case "fluo561":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_561NM
            case "fluo638":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_638NM
            case "fluo730":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_730NM
            case "bfledfull":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_FULL
            case "bfledleft":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_LEFT_HALF
            case "bfledright":
                return ILLUMINATION_CODE.ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_HALF
            case _:
                raise ValueError(f"unknown handle {handle}")

@dataclass
class Position:
    x_usteps:int
    y_usteps:int
    z_usteps:int

    @property
    def x_pos_mm(self):
        return FirmwareDefinitions.mm_per_ustep_x()*self.x_usteps
    @property
    def y_pos_mm(self):
        return FirmwareDefinitions.mm_per_ustep_y()*self.y_usteps
    @property
    def z_pos_mm(self):
        return FirmwareDefinitions.mm_per_ustep_z()*self.z_usteps

    @property
    def pos(self)->AdapterPosition:
        return AdapterPosition(
            x_pos_mm=self.x_pos_mm,
            y_pos_mm=self.y_pos_mm,
            z_pos_mm=self.z_pos_mm,
        )

class Command:
    def __init__(self):
        self.bytes=bytearray(FirmwareDefinitions.COMMAND_PACKET_LENGTH)
        self.wait_for_completion=True

        self.is_move_cmd=False
    
    def __getitem__(self,i:int):
        return self.bytes[i]
    def __setitem__(self,i:int,newval:int):
        self.bytes[i]=newval
    def __len__(self):
        return len(self.bytes)
    
    def set_id(self,id:int):
        self[0]=id

    @staticmethod
    def reset()->"Command":
        ret=Command()
        ret[1]=CommandName.RESET.value

        return ret

    @staticmethod
    def initialize()->"Command":
        ret=Command()
        ret[1]=CommandName.INITIALIZE.value

        return ret
    
    @staticmethod
    def set_leadscrew_pitch(axis:tp.Literal["x","y","z"])->"Command":
        ret=Command()
        ret[1]=CommandName.SET_LEAD_SCREW_PITCH.value

        axis_int=None
        screw_pitch_mm=None
        match axis:
            case "x":
                axis_int=FirmwareDefinitions.AXIS_X
                screw_pitch_mm=FirmwareDefinitions.SCREW_PITCH_X_MM
            case "y":
                axis_int=FirmwareDefinitions.AXIS_Y
                screw_pitch_mm=FirmwareDefinitions.SCREW_PITCH_Y_MM
            case "z":
                axis_int=FirmwareDefinitions.AXIS_Z
                screw_pitch_mm=FirmwareDefinitions.SCREW_PITCH_Z_MM
            case _:
                raise RuntimeError("invalid axis "+axis)


        ret[2]=axis_int
        ret[3]=(int(screw_pitch_mm*1e3)>>8*1)&0xFF
        ret[4]=(int(screw_pitch_mm*1e3)>>8*0)&0xFF

        return ret
    
    @staticmethod
    def configure_motor_driver(axis:tp.Literal["x","y","z"])->"Command":
        ret=Command()
        ret[1]=CommandName.CONFIGURE_STEPPER_DRIVER.value

        axis_int=None
        microstepping_default=None
        motor_rms_current=None
        motor_i_hold=None
        match axis:
            case "x":
                axis_int=FirmwareDefinitions.AXIS_X
                microstepping_default=FirmwareDefinitions.MICROSTEPPING_DEFAULT_X
                motor_rms_current=FirmwareDefinitions.X_MOTOR_RMS_CURRENT_mA
                motor_i_hold=FirmwareDefinitions.X_MOTOR_I_HOLD
            case "y":
                axis_int=FirmwareDefinitions.AXIS_Y
                microstepping_default=FirmwareDefinitions.MICROSTEPPING_DEFAULT_Y
                motor_rms_current=FirmwareDefinitions.Y_MOTOR_RMS_CURRENT_mA
                motor_i_hold=FirmwareDefinitions.Y_MOTOR_I_HOLD
            case "z":
                axis_int=FirmwareDefinitions.AXIS_Z
                microstepping_default=FirmwareDefinitions.MICROSTEPPING_DEFAULT_Z
                motor_rms_current=FirmwareDefinitions.Z_MOTOR_RMS_CURRENT_mA
                motor_i_hold=FirmwareDefinitions.Z_MOTOR_I_HOLD
            case _:
                raise RuntimeError("invalid axis "+axis)
            
        
        ret[2]=axis_int
        ret[3]=max(min(microstepping_default,255),1)
        ret[4]=(motor_rms_current>>8*1)&0xFF
        ret[5]=(motor_rms_current>>8*0)&0xFF
        ret[6]=int(motor_i_hold*255)

        return ret
    
    @staticmethod
    def set_max_velocity_acceleration(axis:tp.Literal["x","y","z"])->"Command":
        ret=Command()
        ret[1]=CommandName.SET_MAX_VELOCITY_ACCELERATION.value

        axis_int=None
        max_vel_mm=None
        max_acc_mm=None
        match axis:
            case "x":
                axis_int=FirmwareDefinitions.AXIS_X
                max_vel_mm=FirmwareDefinitions.MAX_VELOCITY_X_mm
                max_acc_mm=FirmwareDefinitions.MAX_ACCELERATION_X_mm
            case "y":
                axis_int=FirmwareDefinitions.AXIS_Y
                max_vel_mm=FirmwareDefinitions.MAX_VELOCITY_Y_mm
                max_acc_mm=FirmwareDefinitions.MAX_ACCELERATION_Y_mm
            case "z":
                axis_int=FirmwareDefinitions.AXIS_Z
                max_vel_mm=FirmwareDefinitions.MAX_VELOCITY_Z_mm
                max_acc_mm=FirmwareDefinitions.MAX_ACCELERATION_Z_mm
            case _:
                raise RuntimeError("invalid axis "+axis)
            
        
        ret[2]=axis_int
        ret[3]=(int(max_vel_mm*100)>>8*1)&0xFF
        ret[4]=(int(max_vel_mm*100)>>8*0)&0xFF
        ret[5]=(int(max_acc_mm*10)>>8*1)&0xFF
        ret[6]=(int(max_acc_mm*10)>>8*0)&0xFF

        return ret
    
    @staticmethod
    def set_limit_switch_polarity(axis:tp.Literal["x","y","z"])->"Command":
        ret=Command()

        ret[1]=CommandName.SET_LIM_SWITCH_POLARITY.value
        match axis:
            case "x":
                ret[2]=FirmwareDefinitions.AXIS_X
                ret[3]=FirmwareDefinitions.X_HOME_SWITCH_POLARITY
            case "y":
                ret[2]=FirmwareDefinitions.AXIS_Y
                ret[3]=FirmwareDefinitions.Y_HOME_SWITCH_POLARITY
            case "z":
                ret[2]=FirmwareDefinitions.AXIS_Z
                ret[3]=FirmwareDefinitions.Z_HOME_SWITCH_POLARITY

        return ret
    
    @staticmethod
    def configure_actuators()->tp.List["Command"]:
        rets=[]

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
    def home(direction:tp.Literal["x","y","z"])->"Command":
        ret = Command()
        ret[1] = CommandName.HOME_OR_ZERO.value
        match direction:
            case "x":
                ret[2] = FirmwareDefinitions.AXIS_X
                # "move backward" (1?) if SIGN is 1, "move forward" (0?) if SIGN is -1
                ret[3] = int((FirmwareDefinitions.STAGE_MOVEMENT_SIGN_X+1)/2)
            case "y":
                ret[2] = FirmwareDefinitions.AXIS_Y
                # "move backward" (1?) if SIGN is 1, "move forward" (0?) if SIGN is -1
                ret[3] = int((FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Y+1)/2)
            case "z":
                ret[2] = FirmwareDefinitions.AXIS_Z
                # "move backward" (1?) if SIGN is 1, "move forward" (0?) if SIGN is -1
                ret[3] = int((FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Z+1)/2)
            case _:
                raise RuntimeError("invalid direction "+direction)

        ret.is_move_cmd=True

        return ret
    
    @staticmethod
    def move_by_mm(direction:tp.Literal["x","y","z"],distance_mm:float)->tp.List["Command"]:
        num_usteps=None
        command_name=None
        match direction:
            case "x":
                num_usteps=FirmwareDefinitions.mm_to_ustep_x(distance_mm)
                command_name=CommandName.MOVE_X.value
            case "y":
                num_usteps=FirmwareDefinitions.mm_to_ustep_y(distance_mm)
                command_name=CommandName.MOVE_Y.value
            case "z":
                num_usteps=FirmwareDefinitions.mm_to_ustep_z(distance_mm)
                command_name=CommandName.MOVE_Z.value
            case _:
                raise RuntimeError("invalid direction "+direction)
            

        rets=[]
        usteps_remaining=num_usteps
        MAX_USTEPS=2**31-1
        while np.abs(usteps_remaining)>0:
            partial_usteps=max(min(usteps_remaining,MAX_USTEPS),-MAX_USTEPS)
            usteps_remaining-=partial_usteps

            partial_usteps=twos_complement(partial_usteps,4)

            ret=Command()

            ret.is_move_cmd=True

            ret[1]=command_name
            ret[2]=(partial_usteps>>(8*3))&0xFF
            ret[3]=(partial_usteps>>(8*2))&0xFF
            ret[4]=(partial_usteps>>(8*1))&0xFF
            ret[5]=(partial_usteps>>(8*0))&0xFF

            rets.append(ret)

        return rets
    
    @staticmethod
    def set_limit_mm(axis:tp.Literal["x","y","z"],coord:float,direction:tp.Literal["lower","upper"])->"Command":
        ret=Command()
        ret[1]=CommandName.SET_LIM.value

        match (axis,direction):
            case ("x","upper"):
                usteps=FirmwareDefinitions.mm_to_ustep_x(coord)
                limit_code=LIMIT_CODE.X_POSITIVE if FirmwareDefinitions.STAGE_MOVEMENT_SIGN_X > 0 else LIMIT_CODE.X_NEGATIVE
            case ("x","lower"):
                usteps=FirmwareDefinitions.mm_to_ustep_x(coord)
                limit_code=LIMIT_CODE.X_POSITIVE if FirmwareDefinitions.STAGE_MOVEMENT_SIGN_X < 0 else LIMIT_CODE.X_NEGATIVE
            case ("y","upper"):
                usteps=FirmwareDefinitions.mm_to_ustep_y(coord)
                limit_code=LIMIT_CODE.Y_POSITIVE if FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Y > 0 else LIMIT_CODE.Y_NEGATIVE
            case ("y","lower"):
                usteps=FirmwareDefinitions.mm_to_ustep_y(coord)
                limit_code=LIMIT_CODE.Y_POSITIVE if FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Y < 0 else LIMIT_CODE.Y_NEGATIVE
            case ("z","upper"):
                usteps=FirmwareDefinitions.mm_to_ustep_z(coord)
                limit_code=LIMIT_CODE.Z_POSITIVE if FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Z > 0 else LIMIT_CODE.Z_NEGATIVE
            case ("z","lower"):
                usteps=FirmwareDefinitions.mm_to_ustep_z(coord)
                limit_code=LIMIT_CODE.Z_POSITIVE if FirmwareDefinitions.STAGE_MOVEMENT_SIGN_Z < 0 else LIMIT_CODE.Z_NEGATIVE
            case _:
                raise ValueError(f"unsupported axis {axis}")
        
        ret[2] = limit_code
        payload = twos_complement(usteps,4)
        ret[3] = (payload>>8*3)&0xFF
        ret[4] = (payload>>8*2)&0xFF
        ret[5] = (payload>>8*1)&0xFF
        ret[6] = (payload>>8*0)&0xFF
        return ret
    
    @staticmethod
    def set_zero(axis:tp.Literal["x","y","z"])->"Command":
        """
            set current position on axis to 0
        """

        ret=Command()
        ret[1]=CommandName.HOME_OR_ZERO.value

        match axis:
            case "x":
                ret[2]=FirmwareDefinitions.AXIS_X
            case "y":
                ret[2]=FirmwareDefinitions.AXIS_Y
            case "z":
                ret[2]=FirmwareDefinitions.AXIS_Z
            case _:
                raise RuntimeError("invalid axis "+axis)
            
        ret[3] = HOME_OR_ZERO.ZERO

        return ret

    @staticmethod
    def move_to_mm(axis:tp.Literal["x","y","z"],coord_mm:float)->"Command":
        """
            move to z is currently TODO, and potentially dangerous. this command does not indicate completion, ever! (but it does move to the target position..)
        """

        ret=Command()

        axis_cmd=None
        move_usteps=None

        match axis:
            case "x":
                axis_cmd=CommandName.MOVETO_X.value
                move_usteps=FirmwareDefinitions.mm_to_ustep_x(coord_mm)
            case "y":
                axis_cmd=CommandName.MOVETO_Y.value
                move_usteps=FirmwareDefinitions.mm_to_ustep_y(coord_mm)
            case "z":
                axis_cmd=CommandName.MOVETO_Z.value
                move_usteps=FirmwareDefinitions.mm_to_ustep_z(coord_mm)
            case _:
                raise RuntimeError("invalid axis "+axis)
            
        move_usteps=twos_complement(move_usteps,4)

        ret[1]=axis_cmd
        ret[2]=(move_usteps>>(8*3))&0xFF
        ret[3]=(move_usteps>>(8*2))&0xFF
        ret[4]=(move_usteps>>(8*1))&0xFF
        ret[5]=(move_usteps>>(8*0))&0xFF

        ret.is_move_cmd=True

        return ret

    @staticmethod
    def illumination_begin(illumination_source:ILLUMINATION_CODE,intensity_percent:float)->tp.List["Command"]:
        """
            turn on illumination source, and set intensity

            intensity_percent: should be in range [0;100] - will be internally clamped to [0;100] for safety either way
        """

        cmds=[]

        INTENSITY_INT_MAX=0xFFFF
        intensity_int=int(intensity_percent/100*INTENSITY_INT_MAX)
        intensity_clamped=max(min(intensity_int,INTENSITY_INT_MAX),0)

        cmd = Command()
        if illumination_source.is_led_matrix:
            cmd[1] = CommandName.SET_ILLUMINATION_LED_MATRIX.value
        else:
            cmd[1] = CommandName.SET_ILLUMINATION.value
        cmd[2] = illumination_source.value
        cmd[3] = (intensity_clamped >>8*1) & 0xFF
        cmd[4] = (intensity_clamped >>8*0) & 0xFF
        cmds.append(cmd)

        cmd = Command()
        cmd[1] = CommandName.TURN_ON_ILLUMINATION.value
        cmds.append(cmd)

        return cmds

    @staticmethod
    def illumination_end(illumination_source:tp.Optional[ILLUMINATION_CODE]=None)->tp.List["Command"]:
        cmds=[]
        if illumination_source is not None:
            cmd = Command()
            cmd[1] = CommandName.SET_ILLUMINATION.value
            cmd[2] = illumination_source.value
            cmd[3] = 0
            cmd[4] = 0
            cmds.append(cmd)
        
        cmd = Command()
        cmd[1] = CommandName.TURN_OFF_ILLUMINATION.value
        cmds.append(cmd)

        return cmds

    @staticmethod
    def set_pin_level(pin:int,level:int)->"Command":
        """
        sets pin level (the interpretation of this is up to the firmware)

        pin: pin number
        level: 0 or 1
        """

        assert level in [0,1], f"invalid level {level}"

        cmd = Command()
        cmd[1] = CommandName.SET_PIN_LEVEL.value
        cmd[2] = pin
        cmd[3] = level
        return cmd

    @staticmethod
    def af_laser_illum_begin()->"Command":
        return Command.set_pin_level(pin=MCU_PINS.AF_LASER,level=1)

    @staticmethod
    def af_laser_illum_end()->"Command":
        return Command.set_pin_level(pin=MCU_PINS.AF_LASER,level=0)

import asyncio

class Microcontroller:
    @staticmethod
    async def _wait_until_cmd_is_finished(
        mc:"Microcontroller",
        cmd:"Command",
        additional_delay:tp.Optional[float]=None
    ):
        last_position_x=0
        last_position_y=0
        last_position_z=0
        init_wait_complete=False

        def read_packet(packet:MicrocontrollerStatusPackage)->bool:
            nonlocal last_position_x
            nonlocal last_position_y
            nonlocal last_position_z
            nonlocal init_wait_complete

            latest_position_x=packet.x_pos_usteps
            latest_position_y=packet.y_pos_usteps
            latest_position_z=packet.z_pos_usteps

            if cmd.is_move_cmd:
                # wait for 5ms for a move command to have engaged the motors
                if not init_wait_complete:
                    time.sleep(5e-3)
                    init_wait_complete=True
                    return False

                # wait until stage has stopped moving (which the microcontroller has no knowledge of)
                if last_position_x!=latest_position_x or last_position_y!=latest_position_y or last_position_z!=latest_position_z:
                    last_position_x=latest_position_x
                    last_position_y=latest_position_y
                    last_position_z=latest_position_z
                    return False

            cmd_is_completed=packet.last_cmd_id==cmd[0] and packet.exec_status==0
            if cmd.is_move_cmd:
                # TODO : moves in Z sometimes to not indicate completion, even when moving to target position is done
                cmd_name=CommandName(cmd[1])
                match cmd_name:
                    case CommandName.MOVETO_Z:
                        cmd_is_completed=packet.last_cmd_id==cmd[0]
                    case CommandName.MOVE_Z:
                        cmd_is_completed=packet.last_cmd_id==cmd[0]

            return cmd_is_completed
        
        await mc._read_packets(read_packet)

        if additional_delay is not None:
            await asyncio.sleep(additional_delay)
    
    def __init__(self,device_info:SerialDeviceInfo):
        self.device_info=device_info
        self.handle=None

        self.lock=threading.Lock()
        """ lock on hardware control """
        self.illum=threading.Lock()
        """ lock on illumination control """

        self.crc_calculator=crc.CrcCalculator(crc.Crc8.CCITT,table_based=True)
        self._last_command_id=-1 # because this is increment BEFORE it is returned, init to -1 -> first id assigned is 0

        self.terminate_reading_received_packet_thread=False

        self.last_position=Position(0,0,0)

    async def _read_packets(self,until:tp.Callable[[MicrocontrollerStatusPackage],bool]):
        """
            read packets until 'until' returns True

            this is the central waiting function. other functions with wait-like behaviour rely on this function.

            params:
                until:
                    called on each received package, must return True to continue this loop
        """

        MICROCONTROLLER_PACKET_RETRY_DELAY=5e-3 # 5ms

        self.terminate_reading_received_packet_thread=False
        while not self.terminate_reading_received_packet_thread:
            # wait to receive data
            assert self.handle is not None
            serial_in_waiting_status=self.handle.in_waiting

            if serial_in_waiting_status==0:
                await asyncio.sleep(MICROCONTROLLER_PACKET_RETRY_DELAY)
                continue
            
            # get rid of old data
            num_bytes_in_rx_buffer = serial_in_waiting_status
            if num_bytes_in_rx_buffer > FirmwareDefinitions.READ_PACKET_LENGTH:
                for i in range(num_bytes_in_rx_buffer-FirmwareDefinitions.READ_PACKET_LENGTH):
                    self.handle.read()

            # if data is incomplete, sleep and try again (extremely rare case)
            if serial_in_waiting_status % FirmwareDefinitions.READ_PACKET_LENGTH != 0:
                await asyncio.sleep(MICROCONTROLLER_PACKET_RETRY_DELAY)
                continue
            
            # read the buffer
            msg=[]
            for i in range(FirmwareDefinitions.READ_PACKET_LENGTH):
                msg.append(ord(self.handle.read()))

            packet=MicrocontrollerStatusPackage(msg)
            # save last position
            self.last_position=packet.pos
            
            self.terminate_reading_received_packet_thread=until(packet)

    async def get_last_position(self)->Position:
        """
        get last known position of the stage

        this function works even if another command is currently running
        """

        if self.lock.acquire(blocking=False):
            def read_one_packet(packet_in:MicrocontrollerStatusPackage)->bool:
                return True
            
            # internally updates the last known position
            await self._read_packets(read_one_packet)

            self.lock.release()

        return self.last_position

    def _get_next_cmd_id(self)->int:
        """
            generate id for next command, and return it

            just increment last command id by 1, and wrap to 0 after 255
        """

        self._last_command_id=(self._last_command_id+1)%256
        return self._last_command_id

    async def send_cmd(self,cmd_in:tp.Union["Command",tp.List["Command"]]):
        "send command for execution. waits for command to complete if command type requires awaiting."
        if isinstance(cmd_in,list):
            cmds=cmd_in
        else:
            cmds=[cmd_in]

        with self.lock:
            for cmd in cmds:
                cmd.set_id(self._get_next_cmd_id())
                cmd[-1] = self.crc_calculator.calculate_checksum(cmd.bytes[:-1])

                match cmd[1]:
                    case CommandName.TURN_ON_ILLUMINATION.value:
                        # overlapping illumination can cause all sorts of issues
                        if self.illum.locked():
                            raise RuntimeError("illumination already on")
                        self.illum.acquire()
                    case CommandName.TURN_OFF_ILLUMINATION.value:
                        # illumination is in undefined state upon startup
                        # and turning off does not damage anything, so we allow this
                        # (e.g. on startup commands can be issued to ensure all illumination is off)
                        if self.illum.locked():
                            self.illum.release()
                
                assert self.handle is not None
                self.handle.write(cmd.bytes)
                if cmd.wait_for_completion:
                    await Microcontroller._wait_until_cmd_is_finished(self,cmd)

    def open(self):
        "open connection to device"
        self.handle=serial.Serial(self.device_info.device,2000000)

    def close(self):
        "close connection to device"
        assert self.handle is not None
        self.handle.close()
        self.handle=None

    @staticmethod
    def get_all()->tp.List["Microcontroller"]:
        "get all available devices"
        ret=[]
        for p in serial.tools.list_ports.comports():
            if p.description=="Arduino Due":
                device_info=p
            elif p.manufacturer=="Teensyduino":
                device_info=p
            else:
                # we dont care about other devices
                continue

            ret.append(Microcontroller(device_info))

        return ret
