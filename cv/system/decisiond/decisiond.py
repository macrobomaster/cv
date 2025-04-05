import time, math
from dataclasses import dataclass
from collections import deque

import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ..core.helpers import Debounce

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

class AimErrorKF:
  def __init__(self, dt:float=1/100):
    self.dt = dt
    self.reset()

  def predict_and_correct(self, x:float, y:float) -> tuple[float, float]:
    self.km.predict()
    est = self.km.correct(np.array([[x], [y]], dtype=np.float32)).flatten().tolist()
    return est[0], est[1]

  def reset(self):
    self.km = cv2.KalmanFilter(6, 2, 0)
    self.km.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
    self.km.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
    self.km.errorCovPost = np.eye(6, dtype=np.float32)
    transition_matrix = np.eye(6, dtype=np.float32)
    transition_matrix[0, 2] = self.dt
    transition_matrix[1, 3] = self.dt
    transition_matrix[2, 4] = self.dt
    transition_matrix[3, 5] = self.dt
    transition_matrix[0, 4] = 0.5 * self.dt * self.dt
    transition_matrix[1, 5] = 0.5 * self.dt * self.dt
    self.km.transitionMatrix = transition_matrix
    measurement_matrix = np.zeros((2, 6), dtype=np.float32)
    measurement_matrix[0, 0] = 1
    measurement_matrix[1, 1] = 1
    self.km.measurementMatrix = measurement_matrix

class AimErrorSpinCompensator:
  def __init__(self, size:int=100):
    self.size = size
    self.xs = deque(maxlen=size)
    self.maxs = deque(maxlen=size // 10)
    self.mins = deque(maxlen=size // 10)

  def correct(self, x:float) -> float:
    self.xs.append(x)
    if len(self.xs) < self.size:
      return x

    # see if we have clustering of max and min values
    # self.maxs.append(max(self.xs))
    # self.mins.append(min(self.xs))
    # if len(self.maxs) == self.maxs.maxlen and len(self.mins) == self.mins.maxlen:
    #   max_avg = sum(self.maxs) / len(self.maxs)
    #   min_avg = sum(self.mins) / len(self.mins)
    #   # see if all maxs and mins are near their average
    #   for i in range(len(self.maxs)):
    #     if abs(self.maxs[i] - max_avg) > 0.1 or abs(self.mins[i] - min_avg) > 0.1:
    #       return x

    # compute the average of the last size elements
    avg = sum(self.xs) / len(self.xs)
    return avg

class ShootDecision:
  def __init__(self):
    self.window = deque(maxlen=10)

    # only shoot a 3 round burst
    self.burst_start = 0
    self.last_burst = 0

  def step(self, x:float, y:float) -> bool:
    # add distance to the window
    dist = math.sqrt(x*x + y*y)
    self.window.append(dist)

    now = time.monotonic()
    if self.burst_start > 0:
      if now - self.burst_start > 1:
        self.last_burst = now
        self.burst_start = 0
        return False
      else:
        # shoot
        return True
    else:
      if now - self.last_burst > 1:
        if len(self.window) == self.window.maxlen:
          avg = sum(self.window) / len(self.window)
          if avg < 0.2:
            self.burst_start = now
    return False

def run():
  pub = messaging.Pub(["aim_error", "aim_angle", "chassis_velocity", "shoot"])
  sub = messaging.Sub(["autoaim", "plate"], poll="autoaim")

  autoaim_valid_debounce = Debounce(0.1)
  aim_error_kf = AimErrorKF()
  aim_error_spin_comp = AimErrorSpinCompensator()
  shoot_decision = ShootDecision()

  follower = WaypointFollower([
    Waypoint(1, 0, 5),
    Waypoint(1, 1, 5),
    Waypoint(0, 1, 5),
    Waypoint(0, 0, 5),
  ])

  st = time.monotonic()
  while True:
    sub.update()

    autoaim = sub["autoaim"]
    if autoaim is None: continue
    plate = sub["plate"]
    if plate is None: continue

    if sub.updated["autoaim"]:
      if autoaim["valid"]:
        x = (autoaim["xc"] - 256) / 256
        y = (autoaim["yc"] - 128) / 128
        x, y = aim_error_kf.predict_and_correct(x, y)
        # x = aim_error_spin_comp.correct(x)

        shoot = shoot_decision.step(x, y)

        # offset y by some amount relative to the distance to the plate
        y -= 0.1 * plate["dist"]
        y += 0.4

        pub.send("aim_error", {"x": x, "y": y})
        pub.send("shoot", shoot)

        chassis_velocity = {"x": 0.0, "z": 0.0}
        if plate["dist"] >= 1.1:
          chassis_velocity["x"] = min(0.5, max(0, plate["dist"] - 1))
        elif plate["dist"] <= 0.9:
          chassis_velocity["x"] = -max(0.5, 1 - min(1, plate["dist"]))

        pos = plate["pos"]

        # compute angle on xz plane
        angle_x = math.degrees(math.atan2(pos[2], pos[0])) - 90
        # compute angle on yz plane
        angle_y = math.degrees(math.atan2(pos[1], pos[2]))
        pub.send("aim_angle", {"x": angle_x, "y": angle_y})

        if angle_x > 1:
          chassis_velocity["z"] = -min(0.5, abs(angle_x) / 10)
        elif angle_x < -1:
          chassis_velocity["z"] = min(0.5, abs(angle_x) / 10)

        pub.send("chassis_velocity", chassis_velocity)

      if autoaim_valid_debounce.debounce(not autoaim["valid"]):
        aim_error_kf.reset()

    # dt = time.monotonic() - st
    # if dt > 0 and dt <= 120:
    #   vx, vz = follower.step(1/100)
    #   pub.send("chassis_velocity", {"x": vx, "z": vz})
