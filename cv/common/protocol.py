from enum import Enum
import struct

class Command(Enum):
  CHECK_STATE = 0x00
  MOVE_ROBOT = 0x01
  CONTROL_SPINNING = 0x02
  AIM_ERROR = 0x03

COMMAND_FORMATS = {
  Command.CHECK_STATE: 'B',
  Command.MOVE_ROBOT: 'ff',
  Command.CONTROL_SPINNING: 'B',
  Command.AIM_ERROR: 'ff',
}

class Protocol:
  def __init__(self, port):
    self.port = port

  def msg(self, command, *args):
    packed = struct.pack(COMMAND_FORMATS[command], *args)
    length = struct.pack("B", len(packed))
    self.port.write(bytes([command.value]), length, packed)
