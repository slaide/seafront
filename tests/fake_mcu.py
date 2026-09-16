"""
A fake Teensy on a fake serial port.

The microcontroller layer is the one place in this codebase where a mistake is a physical
event, and it is also the least testable: everything below `TeensyMicrocontroller` is a USB
device. This supplies the device instead - a serial handle with `in_waiting`, `read` and
`write`, backed by something that behaves the way the firmware does: it accepts 8-byte
commands, streams 24-byte status packets at whatever rate the test asks for, and reports the
last command id it saw once it decides that command is done.

What it deliberately makes controllable is everything the driver's correctness depends on:

- whether a command is ever acknowledged (a device that has stopped answering)
- how long acknowledgement takes (slow versus dead)
- how many status packets pile up between reads (the driver skips all but the last)
- whether the position advances (the driver infers "still moving" from position changes)
- whether the CRC is right (the driver reads the byte but never checks it)

Nothing here asserts anything. It is the hardware; the tests are next door.
"""

from __future__ import annotations

import threading

import crc
import serial

from seafront.hardware.firmware_config import get_firmware_config
from seafront.hardware.microcontroller.teensy_microcontroller import CommandName

firmware = get_firmware_config()

PACKET_LENGTH = firmware.READ_PACKET_LENGTH  # 24
COMMAND_LENGTH = firmware.COMMAND_PACKET_LENGTH  # 8

EXEC_COMPLETED = 0
EXEC_IN_PROGRESS = 1
EXEC_CHECKSUM_ERROR = 2
EXEC_INVALID = 3
EXEC_ERROR = 4


def _to_payload(value: int, num_bytes: int = 4) -> list[int]:
    """the two's complement encoding the firmware sends positions in"""
    threshold = 2 ** (8 * num_bytes)
    raw = value if value >= 0 else threshold + value
    return [(raw >> (8 * (num_bytes - 1 - i))) & 0xFF for i in range(num_bytes)]


def device_z_mm(device) -> float:
    """
    the fake's z, in mm, as the driver reads it back

    `device.z` holds microsteps exactly as the wire carried them. The status decoder applies
    the axis sign on the way in, so a test that reads or sets the field directly has to apply
    it too - and z is the axis where that matters, being the only one with sign -1.
    """
    return device.z * firmware.STAGE_POS_SIGN_Z * firmware.mm_per_ustep_z


def set_device_z_mm(device, z_mm: float) -> None:
    "put the fake's stage at this z, in the units the driver will read back"
    device.z = round(z_mm / (firmware.STAGE_POS_SIGN_Z * firmware.mm_per_ustep_z))


def status_packet(
    *,
    last_cmd_id: int = 0,
    exec_status: int = EXEC_COMPLETED,
    x_usteps: int = 0,
    y_usteps: int = 0,
    z_usteps: int = 0,
    buttons: int = 0,
    crc_byte: int | None = None,
) -> bytes:
    """
    one status packet, exactly as the firmware lays it out

    cmd id (1), exec status (1), x/y/z/theta (4 each), buttons (1), reserved (4), crc (1).
    `crc_byte` overrides the checksum, which is how a test sends a corrupt packet.
    """
    body = [
        last_cmd_id,
        exec_status,
        *_to_payload(x_usteps),
        *_to_payload(y_usteps),
        *_to_payload(z_usteps),
        *_to_payload(0),  # theta, ignored by the driver
        buttons,
        0, 0, 0, 0,  # reserved
    ]
    assert len(body) == PACKET_LENGTH - 1, len(body)
    calculator = crc.CrcCalculator(crc.Crc8.CCITT, table_based=True)
    checksum = calculator.calculate_checksum(bytes(body)) if crc_byte is None else crc_byte
    return bytes([*body, checksum])


class FakeSerial(serial.Serial):
    """
    the serial handle the driver talks through

    A `serial.Serial` subclass because the driver's `handle` field is type-checked, but one
    that never runs `Serial.__init__` - that would open a port. Only what the driver touches
    is implemented: `in_waiting`, `read`, `write`, `close`.
    """

    def __init__(self, mcu: FakeMCU) -> None:  # deliberately not calling super().__init__
        self._mcu = mcu
        self.is_open = True
        self.closed_count = 0

    @property
    def in_waiting(self) -> int:
        return self._mcu.bytes_waiting()

    def read(self, size: int) -> bytes:
        return self._mcu.read(size)

    def write(self, data: bytes | bytearray) -> int:
        self._mcu.receive_command(bytes(data))
        return len(data)

    def close(self) -> None:
        self.is_open = False
        self.closed_count += 1

    def reset_input_buffer(self) -> None:
        self._mcu.clear()

    def reset_output_buffer(self) -> None:
        pass


class FakeMCU:
    """
    A microcontroller that answers, or does not, exactly as a test asks.

    The default behaviour is a well-behaved device: every command written is acknowledged on
    the next status packet, with `exec_status` completed. Everything interesting is a
    deviation from that, set through the constructor or the knobs below.
    """

    def __init__(
        self,
        *,
        acknowledge: bool = True,
        packets_before_ack: int = 1,
        position: tuple[int, int, int] = (0, 0, 0),
        moves_by: tuple[int, int, int] = (0, 0, 0),
        exec_status: int = EXEC_COMPLETED,
        corrupt_crc: bool = False,
        honour_moves: bool = True,
    ) -> None:
        self.acknowledge = acknowledge
        "whether a written command is ever reported back as the last executed one"
        self.packets_before_ack = packets_before_ack
        "how many status packets go out still naming the previous command"
        self.moves_by = moves_by
        "how far the reported position advances per packet (a stage still in motion)"
        self.exec_status = exec_status
        self.corrupt_crc = corrupt_crc
        self.honour_moves = honour_moves
        "whether an absolute move updates the position this device reports"

        self.x, self.y, self.z = position
        self.commands_received: list[bytes] = []
        "every command packet the driver wrote, in order"

        self._buffer = bytearray()
        self._lock = threading.Lock()
        # 255, not 0: the driver hands out ids starting at 0, and a device reporting 0 as
        # "nothing executed yet" would be indistinguishable from one acknowledging the very
        # first command. Starting elsewhere keeps a test about acknowledgement honest.
        self._acked_cmd_id = 255
        self._pending: int | None = None
        self._packets_since_command = 0
        self._auto = True

    # -- what the serial handle calls -------------------------------------------------------

    def receive_command(self, data: bytes) -> None:
        with self._lock:
            for offset in range(0, len(data), COMMAND_LENGTH):
                packet = data[offset : offset + COMMAND_LENGTH]
                self.commands_received.append(packet)
                self._pending = packet[0]
                self._packets_since_command = 0
                if self.honour_moves:
                    self._apply_move_locked(packet)

    def _apply_move_locked(self, packet: bytes) -> None:
        """
        move the reported position to where an absolute move asked for

        A real stage travels there over some milliseconds; this arrives immediately, which is
        what a test about ordering or coordinates wants. `moves_by` is the knob for tests that
        need it still moving.

        Stored exactly as the wire carried it, which is what the firmware reports back. The
        status decoder then applies the axis sign on the way in, and `mm_to_ustep_*` applied
        it on the way out, so the two cancel and a move to z=1.0 mm reads back as z=1.0 mm.
        Multiplying by the sign here as well applied it twice: x and y have sign +1 so they
        were unaffected, but z has -1, and every absolute z move read back negated.
        """
        moveto = {
            CommandName.MOVETO_X.value: "x",
            CommandName.MOVETO_Y.value: "y",
            CommandName.MOVETO_Z.value: "z",
        }
        # relative moves land here too: the driver's move_by_mm sends MOVE_*, which is what
        # the laser autofocus calibration sweeps z with, so a fake that only honoured
        # absolute moves reported a stage that never left the spot it started on
        move_by = {
            CommandName.MOVE_X.value: "x",
            CommandName.MOVE_Y.value: "y",
            CommandName.MOVE_Z.value: "z",
        }
        axis = moveto.get(packet[1])
        relative = axis is None
        if relative:
            axis = move_by.get(packet[1])
        if axis is None:
            return
        raw = int.from_bytes(packet[2:6], "big", signed=True)
        setattr(self, axis, (getattr(self, axis) if relative else 0) + raw)

    def bytes_waiting(self) -> int:
        with self._lock:
            if self._auto:
                self._emit_locked()
            return len(self._buffer)

    def read(self, size: int) -> bytes:
        with self._lock:
            data = bytes(self._buffer[:size])
            del self._buffer[:size]
            return data

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    # -- what a test calls ------------------------------------------------------------------

    def queue(self, *packets: bytes) -> None:
        """
        put exact packets in the buffer, and stop generating them

        For tests that care about what is in the buffer rather than about the conversation -
        several packets at once, a corrupt one, a packet naming a command nobody sent.
        """
        with self._lock:
            self._auto = False
            for packet in packets:
                self._buffer.extend(packet)

    def ack(self, cmd_id: int) -> None:
        """report this id as the last one executed, from the next packet onward"""
        with self._lock:
            self._acked_cmd_id = cmd_id
            self._pending = None

    def queued_packets(self) -> int:
        with self._lock:
            return len(self._buffer) // PACKET_LENGTH

    def last_command(self) -> bytes | None:
        with self._lock:
            return self.commands_received[-1] if self.commands_received else None

    def command_ids(self) -> list[int]:
        with self._lock:
            return [packet[0] for packet in self.commands_received]

    # -- internals --------------------------------------------------------------------------

    def _emit_locked(self) -> None:
        """top the buffer up with one packet, the way a device streaming status would"""
        if len(self._buffer) >= PACKET_LENGTH:
            return

        self.x += self.moves_by[0]
        self.y += self.moves_by[1]
        self.z += self.moves_by[2]

        if self._pending is not None:
            self._packets_since_command += 1
            if self.acknowledge and self._packets_since_command >= self.packets_before_ack:
                self._acked_cmd_id = self._pending
                self._pending = None

        self._buffer.extend(
            status_packet(
                last_cmd_id=self._acked_cmd_id,
                exec_status=self.exec_status,
                x_usteps=self.x,
                y_usteps=self.y,
                z_usteps=self.z,
                crc_byte=0xFF if self.corrupt_crc else None,
            )
        )


def make_microcontroller(mcu: FakeMCU | None = None):
    """a TeensyMicrocontroller wired to a fake device instead of a USB port"""
    from seafront.hardware.microcontroller.teensy_microcontroller import TeensyMicrocontroller

    device = mcu or FakeMCU()

    class _DeviceInfo:
        device = "/dev/fake-teensy"
        manufacturer = "fake"
        description = "fake teensy"
        serial_number = "FAKE0001"
        vid = 0x16C0
        pid = 0x0483

    controller = TeensyMicrocontroller(device_info=_DeviceInfo(), handle=FakeSerial(device))
    return controller, device
