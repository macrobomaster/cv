"""Frame bridge: re-publishes the shm camera ring as full-res frames for REMOTE tools.

camerad writes each frame into a /dev/shm ring and publishes only the slot index on
`camera_feed` — zero-copy for the LOCAL hot path (autoaimd, tagd read the slot directly).
Remote viz/calib tools (run over --addr from a laptop) can't mmap the robot's shm, so
this daemon — local to the robot — reads each slot and re-publishes the raw bytes on
`camera_feed_full` for them. It pays the tobytes/cbor/TCP cost the hot path avoids, but
in its own process, OFF the hot path, rate-limited, and only when wanted: the supervisor
starts it at DEBUG>=1 (else don't run it and the hot path stays clean).

Subs:  camera_feed:      {ct, st, fid, slot}
Pubs:  camera_feed_full: {ct, st, fid, frame}   (raw RGB bytes, full res)
"""
import gc, time

from tinygrad.helpers import getenv

from ..core import messaging
from ..common.frame_ring import FrameRing, frame_view, CAMERA_RING
from ...autoaim.common import IMG_H, IMG_W

# Viz/calib don't need full camera rate; cap the republish to keep the bridge cheap.
FRAME_PUB_HZ = getenv("FRAME_PUB_HZ", 30)

def run():
  gc.disable()
  pub = messaging.Pub(["camera_feed_full"])
  sub = messaging.Sub(["camera_feed"], poll="camera_feed")
  ring = FrameRing(CAMERA_RING, (IMG_H, IMG_W, 3))

  last_pub = 0.0
  while True:
    sub.update(timeout=200)
    now = time.monotonic()
    cam = sub["camera_feed"]
    if cam is None or not sub.updated["camera_feed"]: continue
    if now - last_pub < 1.0 / FRAME_PUB_HZ: continue
    view = frame_view(ring, cam)
    if view is None: continue
    last_pub = now
    pub.send("camera_feed_full", {"ct": cam["ct"], "st": cam["st"], "fid": cam["fid"], "frame": view.tobytes()})
