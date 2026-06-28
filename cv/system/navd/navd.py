"""Navigation daemon — hold a world pose a fixed standoff from a known tag.

Goal: park the robot STANDOFF metres in front of tag #6 and hold there. The tag
is a surveyed field landmark (`slam.common.TAG_FIELD_MAP`), so the standoff point
is a fixed world coordinate — 2 m out along the tag's face normal. navd reads the
robot's localized pose from slamd (`slam_pose`) and drives the chassis to close
the world-frame position error. No camera/PnP here: detection feeds slamd's
localization, and navd just consumes the resulting pose.

The chassis is holonomic; `chassis_velocity` is {x: forward, y: left} m/s
referenced to the GIMBAL heading (the board rotates chassis→gimbal). So we rotate
the world position error into the gimbal-heading frame (from slam_pose's q_wb) and
command it directly. Heading/orientation is not controlled — this only regulates
position.

Subs:  slam_pose:        {t, p_w, v_w, q_wb, cov_pos, n_tags}   (from slamd)
Pubs:  chassis_velocity: {x, y}   (x = forward, y = left, m/s; commsd → MOVE_ROBOT)
"""
import gc, math, time

import numpy as np

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ...slam import common

NAV_TAG_ID = 6
STANDOFF = 2.0               # m, hold this far out in front of the tag's face
POS_DEADBAND = 0.10          # m, no command once inside this radius of the goal
KP = 1.0                     # (m/s) per m of position error
MAX_SPEED = 1.0              # m/s, per-axis chassis speed cap
STALE_TIMEOUT = 0.5          # s without a fresh slam_pose before stopping

# Goal: STANDOFF metres out along tag #6's outward face normal, in the field
# (world) frame. TAG_FIELD_MAP gives (R_world_tag, t_world_tag); the tag's +z
# column is the direction its face points into the play area.
_R_wt, _t_wt = common.TAG_FIELD_MAP[NAV_TAG_ID]
_normal = _R_wt[:2, 2]
GOAL_XY = (_t_wt[:2] + STANDOFF * _normal / (np.linalg.norm(_normal) + 1e-9)).astype(np.float64)

def gimbal_heading(q_wb):
  """Horizontal unit forward vector of the gimbal in world, from q_wb=[w,x,y,z]
  (world<-body). forward = R_wb @ camera_z; returns None if it points ~straight
  up/down (heading undefined)."""
  w, x, y, z = q_wb
  fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
  n = math.hypot(fx, fy)
  return (fx / n, fy / n) if n > 1e-6 else None

def run():
  gc.disable()
  pub = messaging.Pub(["chassis_velocity"])
  sub = messaging.Sub(["slam_pose"], poll="slam_pose")
  last_wd = last_diag = last_pose_t = 0.0

  kv_put("watchdog", "navd", time.monotonic())
  logger.info(f"navd: holding {STANDOFF:.1f}m standoff at world {GOAL_XY.round(2).tolist()} (tag {NAV_TAG_ID})")

  while True:
    sub.update(timeout=100)
    now = time.monotonic()
    if now - last_wd > 1.0:
      kv_put("watchdog", "navd", now); last_wd = now

    pose = sub["slam_pose"]
    if sub.updated["slam_pose"]: last_pose_t = now
    # Don't drive on a stale pose, or before SLAM has ever anchored to a tag —
    # the world origin is meaningless until the first absolute fix (n_tags counts
    # fixes seen so far; a tighter cov_pos gate could replace this later).
    if pose is None or now - last_pose_t > STALE_TIMEOUT or pose["n_tags"] == 0:
      pub.send("chassis_velocity", {"x": 0.0, "y": 0.0})     # no usable localization ⇒ hold still
      if now - last_diag > 1.0:
        logger.info("navd: no usable slam_pose (stale or unanchored), stopped"); last_diag = now
      continue

    fwd = gimbal_heading(pose["q_wb"])
    if fwd is None: continue                                  # gimbal looking ~straight up/down
    left = (-fwd[1], fwd[0])

    e = GOAL_XY - np.asarray(pose["p_w"], np.float64)[:2]     # world position error to the goal
    dist = float(math.hypot(e[0], e[1]))
    if dist < POS_DEADBAND:
      x = y = 0.0
    else:
      x = max(-MAX_SPEED, min(MAX_SPEED, KP * (e[0] * fwd[0] + e[1] * fwd[1])))   # forward error
      y = max(-MAX_SPEED, min(MAX_SPEED, KP * (e[0] * left[0] + e[1] * left[1]))) # leftward error
    pub.send("chassis_velocity", {"x": x, "y": y})

    if now - last_diag > 1.0:
      logger.info(f"navd: err={dist:.2f}m p_w={np.asarray(pose['p_w']).round(2).tolist()} "
                  f"→ x={x:+.2f} y={y:+.2f}")
      last_diag = now
