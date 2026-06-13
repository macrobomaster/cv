import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from ..core import messaging
from ..core.helpers import Debounce

CAMERA_MATRIX = np.array([[831.90808403,   0.        , 208.91197384],
                          [  0.        , 828.7959212 ,  45.63069367],
                          [  0.        ,   0.        ,   1.        ]], dtype=np.float32)
DIST_COEFFS = np.array([[-0.1703448 ,  0.59690183,  0.00502021, -0.00725316, -1.45582172]], dtype=np.float32)
PLATE_WIDTH, PLATE_HEIGHT = 0.095, 0.104
# Corner order matches autoaim's syndata keypoint order: TL, TR, BL, BR (image-space convention,
# y-down; 3D y-axis is flipped vs image so TL has -y).
PLATE_POINTS = np.array([
  [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # TL
  [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # TR
  [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], # BL
  [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], # BR
], dtype=np.float32)

IMG_W, IMG_H = 512, 256

class PlateKF:
  def __init__(self, dt:float=1/100):
    self.dt = dt
    self.reset()

  def predict_and_correct(self, pos, rot) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    self.km.predict()
    est = self.km.correct(np.array([*pos, *rot], dtype=np.float32).reshape(6, 1)).flatten().tolist()
    return (est[0], est[1], est[2]), (est[9], est[10], est[11])

  def reset(self):
    self.km = cv2.KalmanFilter(18, 6, 0)
    self.km.processNoiseCov = np.eye(18, dtype=np.float32) * 1e-5
    self.km.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
    self.km.errorCovPost = np.eye(18, dtype=np.float32)
    transition_matrix = np.eye(18, dtype=np.float32)
    transition_matrix[0, 3] = self.dt
    transition_matrix[1, 4] = self.dt
    transition_matrix[2, 5] = self.dt
    transition_matrix[3, 6] = self.dt
    transition_matrix[4, 7] = self.dt
    transition_matrix[5, 8] = self.dt
    transition_matrix[0, 6] = 0.5 * self.dt * self.dt
    transition_matrix[1, 7] = 0.5 * self.dt * self.dt
    transition_matrix[2, 8] = 0.5 * self.dt * self.dt
    transition_matrix[9, 12] = self.dt
    transition_matrix[10, 13] = self.dt
    transition_matrix[11, 14] = self.dt
    transition_matrix[12, 15] = self.dt
    transition_matrix[13, 16] = self.dt
    transition_matrix[14, 17] = self.dt
    transition_matrix[9, 15] = 0.5 * self.dt * self.dt
    transition_matrix[10, 16] = 0.5 * self.dt * self.dt
    transition_matrix[11, 17] = 0.5 * self.dt * self.dt
    self.km.transitionMatrix = transition_matrix
    measurement_matrix = np.zeros((6, 18), dtype=np.float32)
    measurement_matrix[0, 0] = 1
    measurement_matrix[1, 1] = 1
    measurement_matrix[2, 2] = 1
    measurement_matrix[3, 9] = 1
    measurement_matrix[4, 10] = 1
    measurement_matrix[5, 11] = 1
    self.km.measurementMatrix = measurement_matrix

def run():
  pub = messaging.Pub(["plate"])
  sub = messaging.Sub(["autoaim"])

  autoaim_valid_debounce = Debounce(1)
  kf = PlateKF()

  while True:
    sub.update()

    autoaim = sub["autoaim"]
    if autoaim is None: continue

    if sub.updated["autoaim"]:
      if autoaim["valid"]:
        # corners come from autoaim normalized to [0,1] of the (IMG_W, IMG_H) input frame,
        # ordered TL, TR, BL, BR (8 floats)
        corners_norm = autoaim["corners"]
        corners_2d = np.array([
          [corners_norm[2*i] * IMG_W, corners_norm[2*i + 1] * IMG_H]
          for i in range(4)
        ], dtype=np.float32)

        # IPPE is the planar-PnP solver; the plate is coplanar so it's the right choice.
        ok, rvec, tvec = cv2.solvePnP(PLATE_POINTS, corners_2d, CAMERA_MATRIX, DIST_COEFFS,
                                      flags=cv2.SOLVEPNP_IPPE)
        if not ok:
          if autoaim_valid_debounce.debounce(True): kf.reset()
          continue

        pos = tvec.flatten()
        rot = R.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler("xyz")

        pos, rot = kf.predict_and_correct(pos, rot)
        dist = float(np.linalg.norm(pos))
        rvec_filtered = R.from_euler("xyz", rot).as_rotvec()

        pub.send("plate", {
          "rot": rot,
          "pos": pos,
          "dist": dist,
          "rvec": rvec_filtered.tolist(),
          "tvec": list(pos),
        })

      if autoaim_valid_debounce.debounce(not autoaim["valid"]):
        kf.reset()
