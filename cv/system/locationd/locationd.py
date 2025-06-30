import time
from pathlib import Path

import cv2
import numpy as np

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put

class LocationKF:
  def __init__(self, dt:float=1/100):
    self.dt = dt
    self.reset()

  def predict_and_correct(self, x:float, y:float, z:float) -> tuple[float, float, float]:
    self.km.predict()
    est = self.km.correct(np.array([[x], [y], [z]], dtype=np.float32)).flatten().tolist()
    return est[0], est[1], est[2]

  def reset(self):
    self.km = cv2.KalmanFilter(9, 3, 0)
    self.km.processNoiseCov = np.eye(9, dtype=np.float32) * 1e-5
    self.km.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-4
    self.km.errorCovPost = np.eye(9, dtype=np.float32)
    transition_matrix = np.eye(9, dtype=np.float32)
    transition_matrix[0, 3] = self.dt
    transition_matrix[1, 4] = self.dt
    transition_matrix[2, 5] = self.dt
    transition_matrix[3, 6] = self.dt
    transition_matrix[4, 7] = self.dt
    transition_matrix[5, 8] = self.dt
    transition_matrix[0, 6] = 0.5 * self.dt * self.dt
    transition_matrix[1, 7] = 0.5 * self.dt * self.dt
    transition_matrix[2, 8] = 0.5 * self.dt * self.dt
    self.km.transitionMatrix = transition_matrix
    measurement_matrix = np.zeros((3, 9), dtype=np.float32)
    measurement_matrix[0, 0] = 1
    measurement_matrix[1, 1] = 1
    measurement_matrix[2, 2] = 1
    self.km.measurementMatrix = measurement_matrix

def run():
  pub = messaging.Pub(["filtered_pos"])
  sub = messaging.Sub(["game_running", "raw_accel"], poll="raw_accel")

  while True:
    sub.update()

    
