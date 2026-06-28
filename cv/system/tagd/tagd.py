"""AprilTag detection daemon.

Isolates the slow, scene-variable `cv2.aruco.detectMarkers` from slamd's fusion
loop. Running it inline dragged slamd down to detection rate (~7 Hz) and stuttered
the pose; in its own process it uses its own cores and never blocks the estimate.
It only DETECTS — slamd does the PnP + field-map fix (so all the geometry/calib
stays in one place). Tags carry the frame capture time `ct`; slamd interpolates
the gimbal pose to `ct`, so detection latency is handled correctly.

Subs:  camera_feed: {ct, st, frame}     (RGB, canonical pinhole)
Pubs:  apriltags:   {ct, detections:[{id, corners:[[u,v]*4]}]}
"""
import gc, time

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.keyvalue import kv_put
from ...slam import common

# Dedicated process → let ArUco use several cores (camerad is pinned/RT, so this
# won't starve it). detectMarkers is the whole job here.
OPENCV_THREADS = getenv("OPENCV_THREADS", 4)
# Tags are intermittent and only feed occasional absolute fixes; cap the rate.
TAG_DETECT_HZ = 10.0

def run():
  gc.disable()
  cv2.setNumThreads(OPENCV_THREADS)
  pub = messaging.Pub(["apriltags"])
  sub = messaging.Sub(["camera_feed"], poll="camera_feed")
  d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, common.APRILTAG_DICT))
  detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
  last_tag_t = last_wd = 0.0

  kv_put("watchdog", "tagd", time.monotonic())

  while True:
    sub.update(timeout=200)
    now = time.monotonic()
    if now - last_wd > 1.0:
      kv_put("watchdog", "tagd", now); last_wd = now

    cam = sub["camera_feed"]
    if cam is None or not sub.updated["camera_feed"]: continue
    if now - last_tag_t < 1.0 / TAG_DETECT_HZ: continue
    last_tag_t = now

    gray = cv2.cvtColor(np.frombuffer(cam["frame"], dtype=np.uint8).reshape(
      common.IMG_H, common.IMG_W, 3), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    dets = []
    if ids is not None:
      for tag_id, c in zip(ids.flatten(), corners):
        dets.append({"id": int(tag_id), "corners": c.reshape(4, 2).astype(float).tolist()})
    pub.send("apriltags", {"ct": float(cam["ct"]), "detections": dets})
