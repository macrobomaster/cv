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
  RAW_ACCEL = 0x05
  GIMBAL_STATE = 0x08

COMMAND_FORMATS = {
  Command.CHECK_STATE: '<B',
  Command.MOVE_ROBOT: '<ff',
  Command.CONTROL_SPINNING: '<B',
  Command.AIM_ERROR: '<ff',
  Command.CONTROL_SHOOT: '<B',
  Command.RAW_ACCEL: '',
  Command.GIMBAL_STATE: '',
}

RESPONSE_FORMATS = {
  Command.CHECK_STATE: '<BB',
  Command.MOVE_ROBOT: '<B',
  Command.CONTROL_SPINNING: '<B',
  Command.AIM_ERROR: '<B',
  Command.CONTROL_SHOOT: '<B',
  Command.RAW_ACCEL: '<fffL',
  # fused-IMU absolute angles + rates (no encoders): pitch, yaw, pitch_rate, yaw_rate (rad, rad/s)
  Command.GIMBAL_STATE: '<ffff',
}

CRC8_INIT = 0xff
CRC8_POLY = 0x31
# DJI's CRC8 is specified with polynomial 0x31 but computed LSB-first.
CRC8_POLY_REFLECTED = 0x8c
PROTOCOL_RETRIES = 3

class ProtocolTimeout(TimeoutError):
  def __init__(self, stage, expected, data):
    super().__init__(stage)
    self.stage = stage
    self.expected = expected
    self.data = data

  def __str__(self):
    return f"{self.stage}: expected {self.expected}B, got {len(self.data)}B ({hex_bytes(self.data)})"

class State(Enum):
  GAME_RUNNING = 0x00
  TEAM_COLOR = 0x01
  ROBOT_TYPE = 0x02

class Protocol:
  port: serial.Serial

  def __init__(self, port):
    self.port = port

  def msg(self, command, *args):
    request_frame = self._frame(command, *args)
    failures = []
    for attempt in range(1, PROTOCOL_RETRIES + 1):
      try:
        # synchronous protocol: drop stale/desync bytes before each exchange
        self.port.reset_input_buffer()
        self.port.write(request_frame)

        response_tag = self._read(1, "response tag")
        response_length = struct.calcsize(RESPONSE_FORMATS[command])
        response_data = self._read(response_length, "response value")
        response_crc_data = self._read(1, "response crc")
        response_crc = struct.unpack("B", response_crc_data)[0]
        response_frame = response_tag + response_data
        expected_crc = crc8(response_frame)
        if response_crc != expected_crc:
          logger.warning(
            f"protocol crc mismatch command={command_str(command)} args={args} "
            f"request={hex_bytes(request_frame)} expected=0x{expected_crc:02x} "
            f"got=0x{response_crc:02x} frame={hex_bytes(response_frame + response_crc_data)}"
          )
          return None
        if response_tag[0] != command.value:
          logger.warning(
            f"protocol response tag mismatch command={command_str(command)} args={args} "
            f"request={hex_bytes(request_frame)} expected=0x{command.value:02x} "
            f"got=0x{response_tag[0]:02x} frame={hex_bytes(response_frame + response_crc_data)}"
          )
          return None
        return struct.unpack(RESPONSE_FORMATS[command], response_data)
      except serial.SerialTimeoutException as e:
        failures.append(f"attempt {attempt}: write timeout: {e}")
        logger.debug(f"protocol write timeout command={command_str(command)} args={args} request={hex_bytes(request_frame)} attempt={attempt}: {e}")
      except ProtocolTimeout as e:
        failures.append(f"attempt {attempt}: {e}")
        logger.debug(f"protocol read timeout command={command_str(command)} args={args} request={hex_bytes(request_frame)} attempt={attempt}: {e}")

    # if we failed 3 times, we probably timed out
    logger.error(f"protocol timed out command={command_str(command)} args={args} request={hex_bytes(request_frame)}; {'; '.join(failures)}")

  def _frame(self, command, *args):
    if command not in COMMAND_FORMATS:
      raise ValueError(f"Invalid command: {command}")
    value = struct.pack(COMMAND_FORMATS[command], *args)
    tag = bytes([command.value])
    frame = tag + value
    return frame + bytes([crc8(frame)])

  def _read(self, length, stage):
    data = self.port.read(length)
    if len(data) != length:
      raise ProtocolTimeout(stage, length, data)
    return data

def command_str(command):
  return f"{command.name}(0x{command.value:02x})"

def hex_bytes(data):
  return data.hex(" ") if data else "<empty>"

def crc8(data, crc=CRC8_INIT):
  for byte in data:
    crc ^= byte
    for _ in range(8):
      if crc & 0x01:
        crc = ((crc >> 1) ^ CRC8_POLY_REFLECTED) & 0xff
      else:
        crc = (crc >> 1) & 0xff
  return crc

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
