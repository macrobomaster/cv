import serial

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from .protocol import Protocol, Command, State

def run():
  port = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
  protocol = Protocol(port)

  pub = messaging.Pub(["game_running"])
  sub = messaging.Sub(["aim_error", "shoot", "chassis_velocity"])

  while True:
    sub.update()

    aim_error = sub["aim_error"]
    if sub.uav["aim_error"]:
      x = aim_error["x"]
      y = aim_error["y"]
      protocol.msg(Command.AIM_ERROR, x, y)
    else:
      protocol.msg(Command.AIM_ERROR, 0.0, 0.0)

    shoot = sub["shoot"]
    if sub.uav["shoot"]:
      protocol.msg(Command.CONTROL_SHOOT, 0xff if shoot else 0x00)
    else:
      protocol.msg(Command.CONTROL_SHOOT, 0x00)

    chassis_velocity = sub["chassis_velocity"]
    if sub.uav["chassis_velocity"]:
      x = chassis_velocity["x"]
      z = chassis_velocity["z"]
      protocol.msg(Command.MOVE_ROBOT, x, z)
    else:
      protocol.msg(Command.MOVE_ROBOT, 0.0, 0.0)

    game_running_msg = None
    game_running = protocol.msg(Command.CHECK_STATE, State.GAME_RUNNING.value)
    if game_running is not None:
      game_running_msg = True if game_running[0] == 0x00 else False
    pub.send("game_running", game_running_msg)
