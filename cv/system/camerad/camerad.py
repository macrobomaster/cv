import time

import cv2
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...common.image import resize_crop

def run():
  pub = messaging.Pub(["camera_feed"])

  cap = cv2.VideoCapture(1)

  while True:
    ret, frame = cap.read()
    if not ret:
      logger.error("failed to read frame")
      time.sleep(1)
      continue

    frame = resize_crop(frame, 512, 256)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pub.send("camera_feed", frame.tobytes())
