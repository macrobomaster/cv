import time
from collections import deque

import rerun as rr

from ..system.core import messaging

rr.init("cv")
rr.connect_tcp()

sub = messaging.Sub(["aim_error"])

points = deque(maxlen=200)
while True:
  sub.update(100)

  aim_error = sub["aim_error"]
  if sub.updated["aim_error"] and aim_error is not None:
    rr.set_time_seconds("time", time.monotonic())
    x = aim_error["x"]
    y = aim_error["y"]
    points.append((x, y))

    rr.log("aim_error", rr.LineStrips2D(points))
