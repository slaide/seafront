from pydantic import BaseModel


class Position(BaseModel):
    x_pos_mm:float
    y_pos_mm:float
    z_pos_mm:float

    @staticmethod
    def zero()->"Position":
        return Position(x_pos_mm=0,y_pos_mm=0,z_pos_mm=0)

class AdapterState(BaseModel):
    "state of an adapter (like SquidAdapter)"
    is_in_loading_position:bool
    stage_position:Position
