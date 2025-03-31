import time
from collections import deque

import rerun as rr
import numpy as np

from ..system.core import messaging

rr.init("cv")
rr.connect_tcp()

sub = messaging.Sub(["aim_error", "chassis_velocity", "camera_feed"])

aim_errors = deque(maxlen=200)
chassis_pos = deque(maxlen=200)
while True:
  sub.update(100)

  aim_error = sub["aim_error"]
  if sub.updated["aim_error"] and aim_error is not None:
    rr.set_time_seconds("time", time.monotonic())
    x = aim_error["x"] * 256 + 256
    y = aim_error["y"] * 128 + 128
    aim_errors.append((x, y))

    rr.log("aim_error", rr.LineStrips2D(aim_errors))

  chassis_velocity = sub["chassis_velocity"]
  if sub.updated["chassis_velocity"] and chassis_velocity is not None:
    rr.set_time_seconds("time", time.monotonic())
    # integrate velocity to get position
    velx = chassis_velocity["x"]
    vely = chassis_velocity["y"]
    x, y = chassis_pos[-1] if chassis_pos else (0, 0)
    x += velx
    y += vely
    chassis_pos.append((x, y))
    rr.log("chassis_position", rr.LineStrips2D(chassis_pos))

  camera_feed = sub["camera_feed"]
  if sub.updated["camera_feed"] and camera_feed is not None:
    rr.set_time_seconds("time", time.monotonic())

    frame = np.frombuffer(camera_feed, dtype=np.uint8).reshape(256, 512, 3)
    rr.log("camera_feed", rr.Image(frame))
