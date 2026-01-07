from pydantic import BaseModel


class DeviceAlreadyInUseError(Exception):
    """Indicate that a hardware device is already in use by another process."""

    def __init__(self, device_type: str, device_id: str):
        self.device_type = device_type
        self.device_id = device_id
        super().__init__(
            f"{device_type} '{device_id}' is already in use. "
            "Another instance of Seafront may be running with this hardware."
        )


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
