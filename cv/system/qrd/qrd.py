"""QR play-style selector.

Shows a QR code to the camera with one of these payloads:
  cv:play_style:passive
  cv:play_style:aggressive
  PLAY_STYLE=passive
  PLAY_STYLE=aggressive

When decoded, publishes `play_style`: {style, raw, t, fid}. stated consumes this and rebuilds its
team-specific state machine with the selected play style. Also emits `qr_ack` once per newly acquired
QR so gimbald can provide physical feedback.
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
QR_ACK_REARM_DT = getenv("QR_ACK_REARM_DT", 2.0)
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
  pub = messaging.Pub(["play_style", "qr_ack"])
  sub = messaging.Sub(["camera_feed"], poll="camera_feed")
  ring = FrameRing(CAMERA_RING, (IMG_H, IMG_W, 3))
  detector = cv2.QRCodeDetector()
  last_scan = 0.0
  last_cv_error_log = 0.0
  last_style = None
  last_ack_style = None
  last_valid_t = -1e9

  while True:
    sub.update(timeout=100)
    if not sub.updated["camera_feed"] or sub.now - last_scan < QR_SCAN_DT: continue
    last_scan = sub.now
    msg = sub["camera_feed"]
    frame = frame_view(ring, msg)
    if frame is None: continue

    try:
      gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      payload, _, _ = detector.detectAndDecode(gray)
    except cv2.error as e:
      if sub.now - last_cv_error_log > 5.0:
        err = " ".join(line.strip() for line in str(e).splitlines() if line.strip())
        logger.warning(f"qrd: QR decode cv2 error; continuing: {err}")
        last_cv_error_log = sub.now
      continue
    if not payload: continue
    style = parse_play_style(payload)
    if style is None: continue

    rearmed = sub.now - last_valid_t > QR_ACK_REARM_DT
    last_valid_t = sub.now
    t_now = time.monotonic()
    scan = {"style": style, "raw": payload, "t": t_now, "fid": msg.get("fid")}
    if style != last_style:
      logger.info(f"qrd: play_style={style} raw={payload!r}")
      last_style = style
    if style != last_ack_style or rearmed:
      pub.send("qr_ack", scan)
      last_ack_style = style
    pub.send("play_style", scan)
