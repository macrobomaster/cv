"""AprilTag detection daemon.

Isolates the slow, scene-variable detector from slamd's fusion loop. Running it
inline dragged slamd to detection rate (~7 Hz) and stuttered the pose; in its own
process it never blocks the estimate. It only DETECTS — slamd does the PnP +
field-map fix on the published corners. Tags carry the frame capture time `ct`;
slamd interpolates the gimbal pose to `ct`, so detection latency is handled.

Detector: AprilTag 3 (the AprilRobotics C lib via `apriltag3.py` ctypes wrapper)
when available — better quads, subpixel corners, multithreaded, `quad_decimate`
speed knob. Falls back to cv2.aruco (subpixel corners) if libapriltag isn't on
the system. Both return cv2.aruco-order corners, so slamd is detector-agnostic.

Subs:  camera_feed: {ct, st, frame}     (RGB, canonical pinhole)
Pubs:  apriltags:   {ct, detections:[{id, corners:[[u,v]*4]}]}
"""
import gc, time

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ...slam import common

# Dedicated process → let the detector use several cores (camerad is pinned/RT,
# so this won't starve it). Detection is the whole job here.
OPENCV_THREADS = getenv("OPENCV_THREADS", 4)
# Rate cap; real rate is min(this, camera rate, detector speed).
TAG_DETECT_HZ = getenv("TAG_DETECT_HZ", 30)
# AprilTag 3 quad-detection decimation (>1 = faster, slight corner-accuracy cost;
# 1 = full res, lowest noise).
AT3_QUAD_DECIMATE = getenv("AT3_QUAD_DECIMATE", 1.0)
# Gaussian blur (px stddev) before quad detection. 0 = off; ~0.8 denoises the
# corners on sharp/noisy frames → steadier pose. Raise if the pose is jittery.
AT3_QUAD_SIGMA = getenv("AT3_QUAD_SIGMA", 0.8)

def _make_detect():
  """detect(gray) -> [(id, corners (4,2) cv2-order)]. AprilTag 3 if libapriltag is
  available, else cv2.aruco with subpixel corner refinement."""
  try:
    from .apriltag3 import Detector
    at = Detector(nthreads=OPENCV_THREADS, quad_decimate=float(AT3_QUAD_DECIMATE),
                  quad_sigma=float(AT3_QUAD_SIGMA))
    logger.info("tagd: using AprilTag 3 (libapriltag)")
    return at.detect
  except Exception as e:                       # lib missing / load error → cv2.aruco
    logger.warning(f"tagd: AprilTag 3 unavailable ({e}); falling back to cv2.aruco")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, common.APRILTAG_DICT))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco = cv2.aruco.ArucoDetector(d, params)
    def detect(gray):
      corners, ids, _ = aruco.detectMarkers(gray)
      if ids is None: return []
      return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)]
    return detect

def run():
  gc.disable()
  cv2.setNumThreads(OPENCV_THREADS)
  pub = messaging.Pub(["apriltags"])
  sub = messaging.Sub(["camera_feed"], poll="camera_feed")
  detect = _make_detect()
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
    dets = [{"id": tid, "corners": c.astype(float).tolist()} for tid, c in detect(gray)]
    pub.send("apriltags", {"ct": float(cam["ct"]), "detections": dets})
