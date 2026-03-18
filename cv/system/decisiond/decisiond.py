import time, math
from dataclasses import dataclass
from collections import deque

import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ..core.helpers import Debounce, FrequencyKeeper

MAINTAIN_DIST = 2
CHASE_SPEED = 2

@dataclass(frozen=True)
class Waypoint:
  x: float
  z: float
  dt: float

class WaypointFollower:
  def __init__(self, waypoints:list[Waypoint]):
    self.waypoints = waypoints
    self.last_waypoint = Waypoint(0, 0, 0)
    self.cur_waypoint = self.waypoints.pop(0)
    self.dt_elapsed = self.cur_waypoint.dt
    self.elapsed = 0

  def step(self, dt:float) -> tuple[float, float]:
    self.elapsed += dt
    if not self.waypoints and self.elapsed > self.dt_elapsed:
      return 0, 0

    if self.elapsed > self.dt_elapsed:
      self.last_waypoint = self.cur_waypoint
      self.cur_waypoint = self.waypoints.pop(0)
      self.dt_elapsed += self.cur_waypoint.dt

    # compute velocity required to reach the waypoint in the dt
    dx = self.cur_waypoint.x - self.last_waypoint.x
    dz = self.cur_waypoint.z - self.last_waypoint.z
    vx = dx / self.cur_waypoint.dt
    vz = dz / self.cur_waypoint.dt
    return vx, vz

def run():
  pub = messaging.Pub(["aim_error", "aim_angle", "chassis_velocity", "shoot"])
  sub = messaging.Sub(["autoaim", "plate", "game_running", "team_color"], poll="autoaim")

  follower = WaypointFollower([
    Waypoint(6.2, 0, 6),
    Waypoint(6.2, 6.2, 6),
    Waypoint(0, 6.2, 6),
    Waypoint(0, 0, 6),
  ])

  fk = FrequencyKeeper(100)

  st = time.monotonic()
  while True:
    sub.update()

    dt = time.monotonic() - st
    if dt > 10 and dt <= 120:
      vx, vz = follower.step(1/100)
      pub.send("chassis_velocity", {"x": vx, "z": vz})

    fk.step()
