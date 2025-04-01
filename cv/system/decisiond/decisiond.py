import time

import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put

class VelocityPlanFollower:
  def __init__(self):
    pass

  def step(self, dt:float) -> tuple[float, float]:
    return 0.0, 0.0

class AimErrorKF:
  def __init__(self, dt:float=1/100):
    self.km = cv2.KalmanFilter(6, 2, 0)
    self.km.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
    self.km.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
    self.km.errorCovPost = np.eye(6, dtype=np.float32)
    transition_matrix = np.eye(6, dtype=np.float32)
    transition_matrix[0, 2] = dt
    transition_matrix[1, 3] = dt
    transition_matrix[2, 4] = dt
    transition_matrix[3, 5] = dt
    transition_matrix[0, 4] = 0.5 * dt * dt
    transition_matrix[1, 5] = 0.5 * dt * dt
    self.km.transitionMatrix = transition_matrix
    measurement_matrix = np.zeros((2, 6), dtype=np.float32)
    measurement_matrix[0, 0] = 1
    measurement_matrix[1, 1] = 1
    self.km.measurementMatrix = measurement_matrix

  def predict_and_correct(self, x:float, y:float) -> tuple[float, float]:
    self.km.predict()
    est = self.km.correct(np.array([[x], [y]], dtype=np.float32)).flatten().tolist()
    return est[0], est[1]

def run():
  pub = messaging.Pub(["aim_error", "chassis_velocity"])
  sub = messaging.Sub(["autoaim"])

  aim_error_kf = AimErrorKF()

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
        x, y = aim_error_kf.predict_and_correct(x, y)
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
