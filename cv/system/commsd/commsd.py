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

  pub = messaging.Pub(["game_running"])
  sub = messaging.Sub(["aim_error", "shoot", "chassis_velocity"])

  while True:
    kv_put("watchdog", "commsd", time.monotonic())

    sub.update()

    aim_error = sub["aim_error"]
    if aim_error is None: continue
    shoot = sub["shoot"]
    if shoot is None: continue
    chassis_velocity = sub["chassis_velocity"]
    if chassis_velocity is None: continue

    if sub.updated["aim_error"]:
      x = aim_error["x"]
      y = aim_error["y"]
      protocol.msg(Command.AIM_ERROR, x, y)
    if not sub.alive["aim_error"]:
      protocol.msg(Command.AIM_ERROR, 0.0, 0.0)

    if sub.updated["shoot"]:
      logger.info(protocol.msg(Command.CONTROL_SHOOT, 0xff if shoot else 0x00))
    if not sub.alive["shoot"]:
      protocol.msg(Command.CONTROL_SHOOT, 0x00)

    if sub.updated["chassis_velocity"]:
      x = chassis_velocity["x"]
      z = chassis_velocity["z"]
      protocol.msg(Command.MOVE_ROBOT, x, z)
    if not sub.alive["chassis_velocity"]:
      protocol.msg(Command.MOVE_ROBOT, 0.0, 0.0)

    game_running = protocol.msg(Command.CHECK_STATE, State.GAME_RUNNING.value)
    if game_running is not None:
      pub.send("game_running", True if game_running[0] == 0x00 else False)
