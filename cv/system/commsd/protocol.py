from enum import Enum
import struct

import serial

from ..core.logging import logger

class Command(Enum):
  CHECK_STATE = 0x00
  MOVE_ROBOT = 0x01
  CONTROL_SPINNING = 0x02
  AIM_ERROR = 0x03
  CONTROL_SHOOT = 0x04

COMMAND_FORMATS = {
  Command.CHECK_STATE: 'B',
  Command.MOVE_ROBOT: 'ff',
  Command.CONTROL_SPINNING: 'B',
  Command.AIM_ERROR: 'ff',
  Command.CONTROL_SHOOT: 'B',
}

RESPONSE_FORMATS = {
  Command.CHECK_STATE: 'BB',
  Command.MOVE_ROBOT: 'B',
  Command.CONTROL_SPINNING: 'B',
  Command.AIM_ERROR: 'B',
  Command.CONTROL_SHOOT: 'B',
}

class State(Enum):
  GAME_RUNNING = 0x00

class Protocol:
  port: serial.Serial

  def __init__(self, port):
    self.port = port

  def msg(self, command, *args):
    for _ in range(3):
      try:
        self._send(command, *args)

        response_command = int.from_bytes(self._read(1), "big")
        if response_command != command.value:
          return None
        response_length = struct.unpack("B", self._read(1))[0]
        if response_length != struct.calcsize(RESPONSE_FORMATS[command]):
          return None
        response_data = self._read(response_length)
        return struct.unpack(RESPONSE_FORMATS[command], response_data)
      except (serial.SerialTimeoutException, TimeoutError):
        pass

    # if we failed 3 times, we probably timed out
    # flush the input buffer to prevent desync
    self.port.reset_input_buffer()

    logger.error(f"command/response timed out")

  def _send(self, command, *args):
    if command not in COMMAND_FORMATS:
      raise ValueError(f"Invalid command: {command}")
    packed = struct.pack(COMMAND_FORMATS[command], *args)
    length = struct.pack("B", len(packed))
    self.port.write(bytes([command.value]) + length + packed)

  def _read(self, length):
    data = self.port.read(length)
    if len(data) != length:
      raise TimeoutError()
    return data

if __name__ == "__main__":
  import time
  import serial
  port = serial.Serial("/dev/ttyUSB0", 115200)

  protocol = Protocol(port)

  assert protocol.msg(Command.CHECK_STATE, 0x0) == (0x0, 0xff)
  assert protocol.msg(Command.MOVE_ROBOT, 0, 0) == (0xff,)
  assert protocol.msg(Command.CONTROL_SPINNING, 1) == (0xff,)
  assert protocol.msg(Command.AIM_ERROR, 0.0, 0.0) == (0xff,)
  assert protocol.msg(Command.CONTROL_SHOOT, 1) == (0x0,)

  # move in "square"
  protocol.msg(Command.MOVE_ROBOT, 1, 0)
  time.sleep(1)
  protocol.msg(Command.MOVE_ROBOT, 0, 1)
  time.sleep(1)
  protocol.msg(Command.MOVE_ROBOT, -1, 0)
  time.sleep(1)
  protocol.msg(Command.MOVE_ROBOT, 0, -1)
  time.sleep(1)
  protocol.msg(Command.MOVE_ROBOT, 0, 0)

  # turret
  protocol.msg(Command.AIM_ERROR, 0.1, 0.0)
  time.sleep(1)
  protocol.msg(Command.AIM_ERROR, -0.1, 0.0)
  time.sleep(1)
  protocol.msg(Command.AIM_ERROR, 0.0, 0.1)
  time.sleep(1)
  protocol.msg(Command.AIM_ERROR, 0.0, -0.1)
  time.sleep(1)
  protocol.msg(Command.AIM_ERROR, 0.0, 0.0)
