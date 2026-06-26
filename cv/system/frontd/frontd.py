"""Feature front-end daemon.

Owns the KLT optical-flow tracker and AprilTag detection. Both are cv2-bound
CPU work that doesn't share state with the filter, so isolating them gives
the MSCKF loop a clean per-frame budget and runs perception in parallel with
estimation.

Subs:
  camera_feed: {ct, st, frame}  — 512x256 RGB bytes (canonical pinhole).

Pubs:
  feature_tracks:
    {
      ct, frame_id,
      terminated: [{id, frame_ids:[int], uvs:[[u,v]]}, ...],  # finalized tracks
      live_uvs, live_ids, live_ages,                          # current tracks
    }
  apriltags:
    {
      ct, frame_id,
      detections: [{id:int, corners:[[u,v]*4]}, ...]          # tag corners (px)
    }
"""
import gc, time

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.keyvalue import kv_put
from ...slam import calib
from ...slam.frontend import FeatureFrontend
from ...slam.types import Frame

# Cap OpenCV's thread pool: each cv2 call (KLT, goodFeatures, findEssentialMat,
# ArUco) otherwise spawns an all-core pool, and with camerad + slamd running too
# the cores oversubscribe and thrash — camerad then can't capture and the whole
# camera_feed stalls. One thread per worker process keeps them cooperative.
OPENCV_THREADS = getenv("OPENCV_THREADS", 1)
# AprilTag detection is the heaviest per-frame op and spikes on cluttered
# scenes; tags are intermittent and the filter only needs occasional absolute
# fixes, so run it at a throttled rate instead of every frame.
TAG_DETECT_HZ = 10.0

def _make_tag_detector():
  d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, calib.APRILTAG_DICT))
  params = cv2.aruco.DetectorParameters()
  return cv2.aruco.ArucoDetector(d, params)

def run():
  gc.disable()
  cv2.setNumThreads(OPENCV_THREADS)

  pub = messaging.Pub(["feature_tracks", "apriltags"])
  sub = messaging.Sub(["camera_feed"], poll="camera_feed")
  fe  = FeatureFrontend()
  tag_detector = _make_tag_detector()
  last_tag_t = 0.0
  last_wd = 0.0

  kv_put("watchdog", "frontd", time.monotonic())

  while True:
    # Block on the camera feed (don't busy-spin) — a 0-timeout poll spins the
    # loop at ~10k Hz, flooding the sqlite KV with watchdog writes and starving
    # every other daemon. Wake on a frame, or every 200 ms for the watchdog.
    sub.update(timeout=200)
    now = time.monotonic()
    if now - last_wd > 1.0:
      kv_put("watchdog", "frontd", now)
      last_wd = now

    cam = sub["camera_feed"]
    if cam is None or not sub.updated["camera_feed"]: continue

    img = np.frombuffer(cam["frame"], dtype=np.uint8).reshape(calib.IMG_H, calib.IMG_W, 3)
    ct  = float(cam["ct"])
    frame = Frame(t=ct, img=img)
    terminated, frame_id = fe.process(frame)

    terminated_data = [
      {"id": tr.id,
       "frame_ids": list(tr.frame_ids),
       "uvs": [[float(uv[0]), float(uv[1])] for uv in tr.uv]}
      for tr in terminated
    ]
    live_uvs, live_ids, live_ages = [], [], []
    for tid, tr in fe.live.items():
      u, v = tr.uv[-1]
      live_uvs.append([float(u), float(v)])
      live_ids.append(int(tid))
      live_ages.append(int(len(tr)))

    pub.send("feature_tracks", {
      "ct": ct,
      "frame_id": int(frame_id),
      "terminated": terminated_data,
      "live_uvs": live_uvs,
      "live_ids": live_ids,
      "live_ages": live_ages,
    })

    # AprilTag detection (throttled). frontd only detects; slamd does PnP +
    # field-map lookup so all geometry stays in one place.
    now = time.monotonic()
    if now - last_tag_t >= 1.0 / TAG_DETECT_HZ:
      last_tag_t = now
      corners, ids, _ = tag_detector.detectMarkers(frame.gray)
      detections = []
      if ids is not None:
        for tag_id, c in zip(ids.flatten(), corners):
          detections.append({"id": int(tag_id),
                             "corners": c.reshape(4, 2).astype(float).tolist()})
      pub.send("apriltags", {"ct": ct, "frame_id": int(frame_id), "detections": detections})
