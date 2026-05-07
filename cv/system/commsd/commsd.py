import time
from pathlib import Path

import serial

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from .protocol import Protocol, Command, State

def run():
  kv_put("watchdog", "commsd", time.monotonic())

  if Path("/dev/ttyTHS1").exists():
    device = "/dev/ttyTHS1"
  else:
    device = "/dev/ttyUSB0"
  logger.info(f"using device {device}")
  port = serial.Serial(device, 115200, timeout=1)
  protocol = Protocol(port)

  pub = messaging.Pub(["game_running", "team_color", "robot_type", "raw_accel", "pitch_angle"])
  sub = messaging.Sub(["aim_error", "shoot", "chassis_velocity", "spinning"])

  while True:
    kv_put("watchdog", "commsd", time.monotonic())

    sub.update()

    aim_error = sub["aim_error"]
    shoot = sub["shoot"]
    chassis_velocity = sub["chassis_velocity"]

    if aim_error is not None:
      if sub.updated["aim_error"]:
        x = aim_error["x"]
        y = aim_error["y"]
        protocol.msg(Command.AIM_ERROR, x, y)
      if not sub.alive["aim_error"]:
        protocol.msg(Command.AIM_ERROR, 0.0, 0.0)

    if shoot is not None:
      if sub.updated["shoot"]:
        protocol.msg(Command.CONTROL_SHOOT, 0xff if shoot else 0x00)
      if not sub.alive["shoot"]:
        protocol.msg(Command.CONTROL_SHOOT, 0x00)

    if chassis_velocity is not None:
      if sub.updated["chassis_velocity"]:
        x = chassis_velocity["x"]
        z = chassis_velocity["z"]
        protocol.msg(Command.MOVE_ROBOT, x, z)
      if not sub.alive["chassis_velocity"]:
        protocol.msg(Command.MOVE_ROBOT, 0.0, 0.0)

    # if sub.alive["spinning"]:
    #   protocol.msg(Command.CONTROL_SPINNING, 0x00)
    # else:
    #   protocol.msg(Command.CONTROL_SPINNING, 0xff)

    # game_running = protocol.msg(Command.CHECK_STATE, State.GAME_RUNNING.value)
    # if game_running is not None:
    #   pub.send("game_running", True if game_running[1] == 0x00 else False)
    #
    # team_color = protocol.msg(Command.CHECK_STATE, State.TEAM_COLOR.value)
    # if team_color is not None:
    #   pub.send("team_color", "red" if team_color[1] == 0x00 else "blue")
    #
    # robot_type = protocol.msg(Command.CHECK_STATE, State.ROBOT_TYPE.value)
    # if robot_type is not None:
    #   pub.send("robot_type", "sentry" if robot_type[1] == 0x00 else "standard")
    #
    # raw_accel = protocol.msg(Command.RAW_ACCEL, 0x00)
    # if raw_accel is not None:
    #   pub.send("raw_accel", raw_accel)
    #
    # pitch_angle = protocol.msg(Command.PITCH_ANGLE, 0x00)
    # if pitch_angle is not None:
    #   pub.send("pitch_angle", pitch_angle)
