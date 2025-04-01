from os import wait
import time

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put

class VelocityPlanFollower:
  def __init__(self):
    pass

  def step(self, dt:float) -> tuple[float, float]:
    return 0.0, 0.0

def run():
  pub = messaging.Pub(["aim_error", "chassis_velocity"])
  sub = messaging.Sub(["autoaim"])

  st = time.monotonic()
  while True:
    sub.update(10)

    autoaim = sub["autoaim"]
    if sub.updated["autoaim"] and autoaim is not None:
      colorm = autoaim["colorm"]
      colorp = autoaim["colorp"]
      if colorm != "none" and colorp > 0.6:
        x = (autoaim["xc"] - 256) / 256
        y = (autoaim["yc"] - 128) / 128
        pub.send("aim_error", {"x": x, "y": y})
      else:
        pub.send("aim_error", {"x": 0.0, "y": 0.0})
    else:
      pub.send("aim_error", {"x": 0.0, "y": 0.0})

    # dt = time.monotonic() - st
    # if dt < 1:
    #   pub.send("chassis_velocity", {"x": 1.0, "y": 0.0})
    # elif dt < 2:
    #   pub.send("chassis_velocity", {"x": 0.0, "y": 1.0})
    # elif dt < 3:
    #   pub.send("chassis_velocity", {"x": -1.0, "y": 0.0})
    # elif dt < 4:
    #   pub.send("chassis_velocity", {"x": 0.0, "y": -1.0})
    # elif dt < 5:
    #   pub.send("chassis_velocity", {"x": 0.0, "y": 0.0})
