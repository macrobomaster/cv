import time, subprocess

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...common.image import resize_crop
from ...common.camera import setup_aravis, get_aravis_frame
from ..plated.plated import CAMERA_MATRIX as REAL_CAMERA_MATRIX, DIST_COEFFS as REAL_DIST_COEFFS
from ...autoaim.syndata import CANONICAL_FX_FY, CANONICAL_CX, CANONICAL_CY

def _build_undistort_maps():
  """Compute remap tables: real camera → canonical pinhole (256x512)."""
  canonical = np.array([[CANONICAL_FX_FY, 0, CANONICAL_CX],
                        [0, CANONICAL_FX_FY, CANONICAL_CY],
                        [0, 0, 1]], dtype=np.float32)
  map1, map2 = cv2.initUndistortRectifyMap(
    REAL_CAMERA_MATRIX, REAL_DIST_COEFFS, None, canonical, (512, 256), cv2.CV_32FC1)
  return map1, map2

def run():
  kv_put("watchdog", "camerad", time.monotonic())

  pub = messaging.Pub(["camera_feed"])

  # Precompute undistortion remap tables (real camera → canonical pinhole)
  undist_map1, undist_map2 = _build_undistort_maps()

  wc = getenv("WEBCAM", -1)
  if wc != -1:
    cap = cv2.VideoCapture(wc)
  else:
    cam, strm = setup_aravis()

  while True:
    kv_put("watchdog", "camerad", time.monotonic())

    ct = time.monotonic()
    if wc != -1:
      ret, frame = cap.read()
      if not ret:
        logger.error("failed to read frame")
        exit(1)
      frame = resize_crop(frame, 512, 256)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
      try:
        frame = get_aravis_frame(cam, strm)
        frame = cv2.resize(frame, (512, 256))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.remap(frame, undist_map1, undist_map2, cv2.INTER_LINEAR)
      except Exception as e:
        logger.error("failed to get frame, restarting camera")
        subprocess.run(["usbreset", "MV-CS016-10UC"])
        raise e

    st = time.monotonic()
    pub.send("camera_feed", {"ct": ct, "st": st, "frame": frame.tobytes()})
