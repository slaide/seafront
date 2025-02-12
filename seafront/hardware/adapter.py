from enum import Enum

from pydantic import BaseModel, Field

from ..server import commands as cmd

class CoreState(str,Enum):
    Idle="idle"
    ChannelSnap="channel_snap"
    ChannelStream="channel_stream"
    LoadingPosition="loading_position"
    Moving="moving"

class Position(BaseModel):
    x_pos_mm:float
    y_pos_mm:float
    z_pos_mm:float

    @staticmethod
    def zero()->"Position":
        return Position(x_pos_mm=0,y_pos_mm=0,z_pos_mm=0)

class AdapterState(BaseModel):
    "state of an adapter (like SquidAdapter)"
    state:CoreState
    is_in_loading_position:bool
    stage_position:Position
