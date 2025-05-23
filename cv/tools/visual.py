import time, math, sys
from collections import deque
from pathlib import Path

import cv2
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..system.core import messaging
from ..system.core.helpers import FrequencyKeeper
from ..system.plated.plated import PLATE_WIDTH, PLATE_HEIGHT, CAMERA_MATRIX, DIST_COEFFS

class HistoryTracker:
  def __init__(self, points_to_track:int=200):
    self.points_to_track = points_to_track
    self.histories = {}

  def __getitem__(self, key:str):
    if key not in self.histories:
      self.histories[key] = deque(maxlen=self.points_to_track)
    return self.histories[key]

  def __setitem__(self, key:str, value):
    if key not in self.histories:
      self.histories[key] = deque(maxlen=self.points_to_track)
    self.histories[key].append(value)

rr.init("cv")
rr.connect_tcp()

fk = FrequencyKeeper(20)

rr.log("world", rr.ViewCoordinates.RDF, static=True)

rr.log("pworld", rr.ViewCoordinates.RDF, static=True)
rr.log("pworld/camera", rr.Pinhole(resolution=(512, 256), image_from_camera=CAMERA_MATRIX, camera_xyz=rr.ViewCoordinates.RDF), static=True)
rr.log("pworld/plate", rr.Asset3D(path=Path(__file__).parent.parent.parent / "weights/armor_plate.gltf", albedo_factor=[0.1, 0.1, 0.1, 1]), static=True)

addr = sys.argv[1]
sub = messaging.Sub(["aim_error", "aim_angle", "shoot", "chassis_velocity", "camera_feed", "autoaim", "plate"], addr=addr)

for service in sub.services:
  rr.log(f"alive/{service}", rr.SeriesLine(width=10), static=True)

ht = HistoryTracker(10)
lt = time.monotonic()
while True:
  sub.update()

  rr.set_time_seconds("time", time.monotonic())

  camera_feed = sub["camera_feed"]
  if time.monotonic() - lt >= 0.1:
    if camera_feed is not None:
      frame = np.frombuffer(camera_feed["frame"], dtype=np.uint8).reshape(256, 512, 3)
      rr.log("raw_camera/feed", rr.Image(frame).compress(70))
    lt = time.monotonic()

  aim_angle = sub["aim_angle"]
  if sub.updated["aim_angle"] and aim_angle is not None:
    rr.log("aim_angle/x", rr.Scalar(aim_angle["x"]))
    rr.log("aim_angle/y", rr.Scalar(aim_angle["y"]))

  autoaim = sub["autoaim"]
  if sub.updated["autoaim"] and autoaim is not None:
    if autoaim["valid"]:
      plate_mu = autoaim["plate_mu"]
      plate_var = autoaim["plate_var"]
      x, y = plate_mu[0], plate_mu[1]
      var = max(plate_var[0], plate_var[1])
      ht["autoaim_c"] = (x, y)
      rr.log("raw_camera/autoaim_c", rr.LineStrips2D(ht["autoaim_c"]))
      rr.log("raw_camera/autoaim_c_cursor", rr.Points2D([(x, y), (x, y)], radii=[var * 10, 2]))
      x, y = plate_mu[2], plate_mu[3]
      var = max(plate_var[2], plate_var[3])
      ht["autoaim_tl"] = (x, y)
      rr.log("raw_camera/autoaim_tl", rr.LineStrips2D(ht["autoaim_tl"]))
      rr.log("raw_camera/autoaim_tl_cursor", rr.Points2D([(x, y), (x, y)], radii=[var * 10, 2]))
      x, y = plate_mu[4], plate_mu[5]
      var = max(plate_var[4], plate_var[5])
      ht["autoaim_tr"] = (x, y)
      rr.log("raw_camera/autoaim_tr", rr.LineStrips2D(ht["autoaim_tr"]))
      rr.log("raw_camera/autoaim_tr_cursor", rr.Points2D([(x, y), (x, y)], radii=[var * 10, 2]))
      x, y = plate_mu[6], plate_mu[7]
      var = max(plate_var[6], plate_var[7])
      ht["autoaim_bl"] = (x, y)
      rr.log("raw_camera/autoaim_bl", rr.LineStrips2D(ht["autoaim_bl"]))
      rr.log("raw_camera/autoaim_bl_cursor", rr.Points2D([(x, y), (x, y)], radii=[var * 10, 2]))
      x, y = plate_mu[8], plate_mu[9]
      var = max(plate_var[8], plate_var[9])
      ht["autoaim_br"] = (x, y)
      rr.log("raw_camera/autoaim_br", rr.LineStrips2D(ht["autoaim_br"]))
      rr.log("raw_camera/autoaim_br_cursor", rr.Points2D([(x, y), (x, y)], radii=[var * 10, 2]))

    rr.log("plate_var_avg", rr.Scalar(autoaim["plate_var_avg"]))

    plate = sub["plate"]
    if sub.updated["plate"] and plate is not None:
      pos = np.array(plate["pos"])
      rot = np.array(plate["rot"]) + np.array([0, math.pi, math.pi])
      quaternion = R.from_euler("xyz", rot.flatten()).as_quat()
      rr.log("pworld/plate", rr.Transform3D(translation=pos, quaternion=quaternion))

      # get distance
      dist = plate["dist"]
      rr.log("pworld/distance", rr.Arrows3D(vectors=[pos], labels=[f"{dist:.3f}m"]))

      rvec = np.array(plate["rvec"])
      tvec = np.array(plate["tvec"])
      square_points = np.array([
        [-PLATE_WIDTH/2, PLATE_HEIGHT/2, 0], # bottom left
        [PLATE_WIDTH/2, PLATE_HEIGHT/2, 0], # bottom right
        [PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # top right
        [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # top left
      ])

      imgpts, _ = cv2.projectPoints(square_points, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
      imgpts = imgpts.astype(int)[:, 0]
      # add the first point to close the loop
      imgpts = np.vstack([imgpts, imgpts[0]])
      rr.log("raw_camera/plate", rr.LineStrips2D(imgpts))

    aim_error = sub["aim_error"]
    if sub.updated["aim_error"] and aim_error is not None:
      rr.set_time_seconds("time", time.monotonic())
      x = aim_error["x"] * 256 + 256
      y = aim_error["y"] * 128 + 128
      ht["aim_error"] = (x, y)

      rr.log("raw_camera/aim_error", rr.LineStrips2D(ht["aim_error"]))
      rr.log("raw_camera/cursor", rr.Points2D([(x, y)], radii=[5], labels=[f"{autoaim['colorm']} {autoaim['numberm']}"]))

  shoot = sub["shoot"]
  if sub.updated["shoot"] and shoot is not None:
    rr.log("shoot", rr.Scalar(int(shoot)))

  chassis_velocity = sub["chassis_velocity"]
  if sub.updated["chassis_velocity"] and chassis_velocity is not None:
    # integrate velocity to get position
    velx = chassis_velocity["x"]
    velz = chassis_velocity["z"]
    x, y, z = ht["chassis_pos"][-1] if ht["chassis_pos"] else (0, 0, 0)
    x += velx
    z += velz
    ht["chassis_pos"] = (x, y, z)
    rr.log("world/pos", rr.LineStrips3D(ht["chassis_pos"]))

  # log alive status
  for k, v in sub.alive.items():
    rr.log(f"alive/{k}", rr.Scalar(int(v)))

  fk.step()
