import math

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put

class PlateKF:
  def __init__(self, dt:float=1/100):
    self.km = cv2.KalmanFilter(18, 6, 0)
    self.km.processNoiseCov = np.eye(18, dtype=np.float32) * 1e-5
    self.km.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
    self.km.errorCovPost = np.eye(18, dtype=np.float32)
    dt = 1/100
    transition_matrix = np.eye(18, dtype=np.float32)
    transition_matrix[0, 3] = dt
    transition_matrix[1, 4] = dt
    transition_matrix[2, 5] = dt
    transition_matrix[3, 6] = dt
    transition_matrix[4, 7] = dt
    transition_matrix[5, 8] = dt
    transition_matrix[0, 6] = 0.5 * dt * dt
    transition_matrix[1, 7] = 0.5 * dt * dt
    transition_matrix[2, 8] = 0.5 * dt * dt
    transition_matrix[9, 12] = dt
    transition_matrix[10, 13] = dt
    transition_matrix[11, 14] = dt
    transition_matrix[12, 15] = dt
    transition_matrix[13, 16] = dt
    transition_matrix[14, 17] = dt
    transition_matrix[9, 15] = 0.5 * dt * dt
    transition_matrix[10, 16] = 0.5 * dt * dt
    transition_matrix[11, 17] = 0.5 * dt * dt
    self.km.transitionMatrix = transition_matrix
    measurement_matrix = np.zeros((6, 18), dtype=np.float32)
    measurement_matrix[0, 0] = 1
    measurement_matrix[1, 1] = 1
    measurement_matrix[2, 2] = 1
    measurement_matrix[3, 9] = 1
    measurement_matrix[4, 10] = 1
    measurement_matrix[5, 11] = 1
    self.km.measurementMatrix = measurement_matrix

  def predict_and_correct(self, pos, rot) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    self.km.predict()
    est = self.km.correct(np.array([*pos, *rot], dtype=np.float32).reshape(6, 1)).flatten().tolist()
    return (est[0], est[1], est[2]), (est[9], est[10], est[11])

def run():
  pub = messaging.Pub(["plate"])
  sub = messaging.Sub(["autoaim"])

  plate_width, plate_height = 0.14, 0.125
  square_points = np.array([
    [-plate_width/2, plate_height/2, 0], # bottom left
    [plate_width/2, plate_height/2, 0], # bottom right
    [plate_width/2, -plate_height/2, 0], # top right
    [-plate_width/2, -plate_height/2, 0], # top left
  ])

  f = 16
  sx, sy = 4.96, 3.72
  width, height = 512, 256
  camera_matrix = np.array([
    [width*f/sx, 0, width/2],
    [0, height*f/sy, height/2],
    [0, 0, 1],
  ], dtype=np.float32)
  dist_coeffs = np.zeros((4, 1))

  kf = PlateKF()

  while True:
    sub.update(10)

    autoaim = sub["autoaim"]
    if sub.updated["autoaim"] and autoaim is not None:
      if autoaim["colorm"] != "none" and autoaim["colorp"] > 0.6:
        xbl, ybl = autoaim["xbl"], autoaim["ybl"]
        xbr, ybr = autoaim["xbr"], autoaim["ybr"]
        xtr, ytr = autoaim["xtr"], autoaim["ytr"]
        xtl, ytl = autoaim["xtl"], autoaim["ytl"]

        image_points = np.array([
          [xbl, ybl],
          [xbr, ybr],
          [xtr, ytr],
          [xtl, ytl],
        ], dtype=np.float32).reshape(-1, 1, 2)
        _, rvec, tvec = cv2.solvePnP(square_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        rot = R.from_rotvec(rvec.flatten()).as_euler("xyz")
        pos = tvec.flatten()

        pos, rot = kf.predict_and_correct(pos, rot)

        pub.send("plate", {
          "rot": rot,
          "pos": pos,
          "rvec": rvec.flatten().tolist(),
          "tvec": tvec.flatten().tolist(),
        })
