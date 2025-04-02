import time, math
from collections import deque

import cv2
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..system.core import messaging

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

plate_width, plate_height = 0.14, 0.125
f = math.pi * 4
sx, sy = 4.96, 3.72
width, height = 512, 256
camera_matrix = np.array([
  [width*f/sx, 0, width/2],
  [0, height*f/sy, height/2],
  [0, 0, 1],
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

rr.log("world", rr.ViewCoordinates.RDF, static=True)

rr.log("pworld", rr.ViewCoordinates.RDF, static=True)
rr.log("pworld/camera", rr.Pinhole(resolution=(512, 256), image_from_camera=camera_matrix, camera_xyz=rr.ViewCoordinates.RDF), static=True)
rr.log("pworld/plate", rr.Asset3D(path="/tmp/armor_plate.gltf", albedo_factor=[0.1, 0.1, 0.1, 1]), static=True)

sub = messaging.Sub(["aim_error", "chassis_velocity", "camera_feed", "autoaim", "plate"])

ht = HistoryTracker(10)
lt = time.monotonic()
while True:
  sub.update()

  rr.set_time_seconds("time", time.monotonic())

  camera_feed = sub["camera_feed"]
  if time.monotonic() - lt >= 0.1:
    if camera_feed is not None:
      frame = np.frombuffer(camera_feed, dtype=np.uint8).reshape(256, 512, 3)
      rr.log("raw_camera/feed", rr.Image(frame).compress(70))
    lt = time.monotonic()

  aim_error = sub["aim_error"]
  if sub.updated["aim_error"] and aim_error is not None:
    rr.set_time_seconds("time", time.monotonic())
    x = aim_error["x"] * 256 + 256
    y = aim_error["y"] * 128 + 128
    ht["aim_error"] = (x, y)

    rr.log("raw_camera/aim_error", rr.LineStrips2D(ht["aim_error"]))
    rr.log("raw_camera/cursor", rr.Points2D([(x, y)], radii=[5]))

  autoaim = sub["autoaim"]
  if sub.updated["autoaim"] and autoaim is not None:
    if autoaim["colorm"] != "none" and autoaim["colorp"] > 0.6:
      x = autoaim["xc"]
      y = autoaim["yc"]
      ht["autoaim_c"] = (x, y)
      rr.log("raw_camera/autoaim_c", rr.LineStrips2D(ht["autoaim_c"]))
      rr.log("raw_camera/autoaim_c_cursor", rr.Points2D([(x, y)], radii=[2], labels=[f"{autoaim['colorm']}"]))
      x = autoaim["xtl"]
      y = autoaim["ytl"]
      ht["autoaim_tl"] = (x, y)
      rr.log("raw_camera/autoaim_tl", rr.LineStrips2D(ht["autoaim_tl"]))
      rr.log("raw_camera/autoaim_tl_cursor", rr.Points2D([(x, y)], radii=[2]))
      x = autoaim["xtr"]
      y = autoaim["ytr"]
      ht["autoaim_tr"] = (x, y)
      rr.log("raw_camera/autoaim_tr", rr.LineStrips2D(ht["autoaim_tr"]))
      rr.log("raw_camera/autoaim_tr_cursor", rr.Points2D([(x, y)], radii=[2]))
      x = autoaim["xbl"]
      y = autoaim["ybl"]
      ht["autoaim_bl"] = (x, y)
      rr.log("raw_camera/autoaim_bl", rr.LineStrips2D(ht["autoaim_bl"]))
      rr.log("raw_camera/autoaim_bl_cursor", rr.Points2D([(x, y)], radii=[2]))
      x = autoaim["xbr"]
      y = autoaim["ybr"]
      ht["autoaim_br"] = (x, y)
      rr.log("raw_camera/autoaim_br", rr.LineStrips2D(ht["autoaim_br"]))
      rr.log("raw_camera/autoaim_br_cursor", rr.Points2D([(x, y)], radii=[2]))

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
        [-plate_width/2, plate_height/2, 0], # bottom left
        [plate_width/2, plate_height/2, 0], # bottom right
        [plate_width/2, -plate_height/2, 0], # top right
        [-plate_width/2, -plate_height/2, 0], # top left
      ])

      imgpts, _ = cv2.projectPoints(square_points, rvec, tvec, camera_matrix, dist_coeffs)
      imgpts = imgpts.astype(int)[:, 0]
      # add the first point to close the loop
      imgpts = np.vstack([imgpts, imgpts[0]])
      rr.log("raw_camera/plate", rr.LineStrips2D(imgpts))

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
