import time, subprocess

import cv2
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...common.image import resize_crop
from ...common.camera import setup_aravis, get_aravis_frame

def run():
  kv_put("watchdog", "camerad", time.monotonic())

  pub = messaging.Pub(["camera_feed"])

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
      except Exception as e:
        logger.error("failed to get frame, restarting camera")
        subprocess.run(["usbreset", "MV-CS016-10UC"])
        raise e

    st = time.monotonic()
    pub.send("camera_feed", {"ct": ct, "st": st, "frame": frame.tobytes()})
