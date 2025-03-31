import cv2
import numpy as np

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put

def run():
  cv2.setNumThreads(0)
  cv2.ocl.setUseOpenCL(False)

  sub = messaging.Sub(["camera_feed", "autoaim"], poll="autoaim")

  while True:
    sub.update(10)

    autoaim = sub["autoaim"]
    if autoaim is None:
      continue

    frame = sub["camera_feed"]
    if frame is None:
      continue

    if sub.updated["autoaim"]:
      frame = np.frombuffer(frame, dtype=np.uint8).reshape(256, 512, 3).copy()

      cl = autoaim["cl"]
      clp = autoaim["clp"]
      x = autoaim["x"]
      y = autoaim["y"]
      colorm = autoaim["colorm"]
      colorp = autoaim["colorp"]
      numberm = autoaim["numberm"]
      numberp = autoaim["numberp"]

      # draw the annotation
      cv2.putText(frame, f"{cl}: {clp:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      if cl == 1 and clp > 0.6:
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{colorm}: {colorp:.3f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{numberm}: {numberp:.3f}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      cv2.imshow("camera_feed", frame)

      if cv2.waitKey(1) == ord("q"):
        kv_put("global_rt", "do_shutdown", True)
