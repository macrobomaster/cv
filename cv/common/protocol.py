from enum import Enum
import struct

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

class Protocol:
  def __init__(self, port):
    self.port = port

  def msg(self, command, *args):
    # send message
    if command not in COMMAND_FORMATS:
      raise ValueError(f"Invalid command: {command}")
    packed = struct.pack(COMMAND_FORMATS[command], *args)
    length = struct.pack("B", len(packed))
    self.port.write(bytes([command.value]) + length + packed)

    # check response
    response_command = Command(self.port.read(1))
    if response_command != command:
      raise ValueError(f"Unexpected response: {response_command}")
    response_length = struct.unpack("B", self.port.read(1))[0]
    assert response_length == struct.calcsize(RESPONSE_FORMATS[command]), f"Invalid response length: {response_length}"
    response_data = self.port.read(response_length)
    return struct.unpack(RESPONSE_FORMATS[command], response_data)

if __name__ == "__main__":
  import time
  import serial
  port = serial.Serial("/dev/ttyUSB0", 115200)

  protocol = Protocol(port)

  assert protocol.msg(Command.CHECK_STATE, 0x0) == (0x0, 0xff)
  assert protocol.msg(Command.MOVE_ROBOT, 0, 0) == (0xff,)
  assert protocol.msg(Command.CONTROL_SPINNING, 1) == (0x0,)
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
