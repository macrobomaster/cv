"""QR play-style selector.

Shows a QR code to the camera with one of these payloads:
  cv:play_style:balanced
  cv:play_style:center
  PLAY_STYLE=balanced
  PLAY_STYLE=center

When decoded, publishes `play_style`: {style, raw, t, fid}. stated consumes this and rebuilds its
team-specific state machine with the selected play style.
"""
import time

import cv2
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..common.frame_ring import FrameRing, frame_view, CAMERA_RING
from ...autoaim.common import IMG_H, IMG_W
from ..stated.states import PLAY_STYLES

QR_SCAN_DT = getenv("QR_SCAN_DT", 0.2)
OPENCV_THREADS = getenv("OPENCV_THREADS", 1)

def parse_play_style(payload:str) -> str|None:
  s = payload.strip()
  low = s.lower()
  if low.startswith("cv:play_style:"):
    style = low.rsplit(":", 1)[-1]
  elif low.startswith("play_style="):
    style = low.split("=", 1)[1]
  else:
    return None
  return style if style in PLAY_STYLES else None

def run():
  cv2.setNumThreads(OPENCV_THREADS)
  pub = messaging.Pub(["play_style"])
  sub = messaging.Sub(["camera_feed"], poll="camera_feed")
  ring = FrameRing(CAMERA_RING, (IMG_H, IMG_W, 3))
  detector = cv2.QRCodeDetector()
  last_scan = 0.0
  last_style = None

  while True:
    sub.update(timeout=100)
    if not sub.updated["camera_feed"] or sub.now - last_scan < QR_SCAN_DT: continue
    last_scan = sub.now
    msg = sub["camera_feed"]
    frame = frame_view(ring, msg)
    if frame is None: continue

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    payload, _, _ = detector.detectAndDecode(gray)
    if not payload: continue
    style = parse_play_style(payload)
    if style is None: continue

    if style != last_style:
      logger.info(f"qrd: play_style={style} raw={payload!r}")
      last_style = style
    pub.send("play_style", {"style": style, "raw": payload, "t": time.monotonic(), "fid": msg.get("fid")})
