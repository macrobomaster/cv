import gc, time, subprocess

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...common.image import resize_crop
from ...common.camera import setup_aravis, get_aravis_frame, latch_timestamp_offset
from ..plated.plated import REAL_CAMERA_MATRIX, REAL_DIST_COEFFS
from ...autoaim.common import CANONICAL_CAMERA_MATRIX

CAPTURE_MID_EXPOSURE = 3000 / 2000000

def _build_undistort_maps(src_w):
  K = REAL_CAMERA_MATRIX.copy()
  K[:2] *= src_w / 512
  map1, map2 = cv2.initUndistortRectifyMap(K, REAL_DIST_COEFFS, None, CANONICAL_CAMERA_MATRIX, (512, 256), cv2.CV_32FC1)
  return map1, map2

def run():
  gc.disable()
  kv_put("watchdog", "camerad", time.monotonic())

  pub = messaging.Pub(["camera_feed"])

  CALIB = getenv("CALIB", 0)
  raw_pub = messaging.Pub(["camera_feed_raw"]) if CALIB else None
  PROF = getenv("PROF", 0)

  wc = getenv("WEBCAM", -1)
  if wc != -1:
    cap = cv2.VideoCapture(wc)
  else:
    try:
      cam, strm, dev, ts_hz, offset = setup_aravis()
    except Exception as e:
      logger.error("failed to setup, restarting camera")
      subprocess.run(["usbreset", "MV-CS016-10UC"])
      raise e
    undist_map1, undist_map2 = _build_undistort_maps(cam.get_region()[2])   # native frame width

  last_wd = time.monotonic()
  fid = 0
  acc_wait = acc_proc = acc_pub = 0.0
  nprof = 0
  last_prof = time.monotonic()
  while True:
    if time.monotonic() - last_wd > 1:
      kv_put("watchdog", "camerad", time.monotonic())
      if wc == -1: offset = latch_timestamp_offset(dev, ts_hz)
      last_wd = time.monotonic()

    t0 = time.monotonic()
    ct = t0
    if wc != -1:
      ret, frame = cap.read()
      if not ret:
        logger.error("failed to read frame")
        exit(1)
      frame = resize_crop(frame, 512, 256)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      if raw_pub is not None:
        raw_pub.send("camera_feed_raw", {"ct": ct, "frame": frame.tobytes()})
      t_get = t_done = time.monotonic()
    else:
      try:
        frame, t_cap = get_aravis_frame(cam, strm, ts_hz, offset)
        t_get = time.monotonic()
        if frame is None: continue
        ct = t_cap + CAPTURE_MID_EXPOSURE
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        if raw_pub is not None:
          raw_pub.send("camera_feed_raw", {"ct": ct, "frame": cv2.resize(frame, (512, 256)).tobytes()})
        frame = cv2.remap(frame, undist_map1, undist_map2, cv2.INTER_LINEAR)
        t_done = time.monotonic()
      except Exception as e:
        logger.error("failed to get frame, restarting camera")
        subprocess.run(["usbreset", "MV-CS016-10UC"])
        raise e

    st = time.monotonic()
    fid += 1
    pub.send("camera_feed", {"ct": ct, "st": st, "fid": fid, "frame": frame.tobytes()})

    if PROF:
      acc_wait += t_get - t0
      acc_proc += t_done - t_get
      acc_pub += time.monotonic() - st
      nprof += 1
      if st - last_prof > 1.0:
        print(f"camerad  wait={1e3*acc_wait/nprof:.2f}ms  proc={1e3*acc_proc/nprof:.2f}ms  pub={1e3*acc_pub/nprof:.2f}ms  ({nprof} fps)")
        acc_wait = acc_proc = acc_pub = 0.0
        nprof = 0
        last_prof = st
