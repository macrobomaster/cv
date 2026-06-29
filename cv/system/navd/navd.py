"""Navigation daemon — play a sequence of standoff waypoints around known tags.

Each waypoint is a fixed world point a given distance out in front of a surveyed
tag's face (`slam.common.TAG_FIELD_MAP`). A WaypointPlayer walks the mission in
order, advancing to the next once the robot has settled at the current one. navd
reads the robot's localized pose from slamd (`slam_pose`), drives the chassis to
close the world-frame position error to the active waypoint, and stops once the
mission is done. No camera/PnP here — detection feeds slamd's localization; navd
just consumes the resulting pose.

Current mission (MISSION): hold 2.5 m in front of tag #6, then 1.5 m in front.

The chassis is holonomic; `chassis_velocity` is {x: forward, y: left} m/s
referenced to the GIMBAL heading (the board rotates chassis→gimbal). So we rotate
the world position error into the gimbal-heading frame (from slam_pose's q_wb) and
command it directly. Heading/orientation is not controlled — position only.

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
# Mission: (tag_id, standoff_m) waypoints, held in order.
MISSION = [(NAV_TAG_ID, 2.5), (NAV_TAG_ID, 2)]
LOOP = False                 # True ⇒ restart the mission after the last waypoint (never finishes)

ARRIVE_RADIUS = 0.15         # m, within this of a waypoint counts as "at" it (> POS_DEADBAND)
ARRIVE_DWELL = 0.3           # s, must stay inside ARRIVE_RADIUS this long before advancing
POS_DEADBAND = 0.10          # m, profile brakes to a stop here; no command inside it
V_MAX = 1.0                  # m/s, trapezoid cruise speed (plateau / hard cap)
ACCEL = 0.6                  # m/s², trapezoid accel = decel ramp rate
MAX_DT = 0.2                 # s, clamp on the loop dt used for the accel ramp (guards stalls)
STALE_TIMEOUT = 0.5          # s without a fresh slam_pose before stopping

def tag_standoff(tag_id:int, dist:float) -> np.ndarray:
  """World-XY point `dist` metres out along tag `tag_id`'s outward face normal.
  TAG_FIELD_MAP gives (R_world_tag, t_world_tag); the tag's +z column is the
  direction its face points into the play area."""
  R_wt, t_wt = common.TAG_FIELD_MAP[tag_id]
  n = R_wt[:2, 2]
  return (t_wt[:2] + dist * n / (np.linalg.norm(n) + 1e-9)).astype(np.float64)

class WaypointPlayer:
  """Plays a fixed list of world-XY goals in order. update(p_xy, now) returns the
  active goal (None once the mission is done), advancing to the next goal after the
  robot has stayed within ARRIVE_RADIUS of the current one for ARRIVE_DWELL. With
  loop=True it wraps back to the first waypoint instead of finishing."""
  def __init__(self, waypoints, arrive_radius=ARRIVE_RADIUS, dwell=ARRIVE_DWELL, loop=False):
    self.waypoints = list(waypoints)
    self.arrive_radius = arrive_radius
    self.dwell = dwell
    self.loop = loop
    self.idx = 0
    self.since = None          # time we entered the arrive radius (None if outside)

  def goal(self):
    return None if self.idx >= len(self.waypoints) else self.waypoints[self.idx]

  def update(self, p_xy:np.ndarray, now:float):
    g = self.goal()
    if g is None: return None
    if float(math.hypot(*(g - p_xy))) <= self.arrive_radius:
      if self.since is None: self.since = now
      if now - self.since >= self.dwell:
        self.idx += 1; self.since = None
        if self.loop and self.idx >= len(self.waypoints): self.idx = 0
    else:
      self.since = None
    return self.goal()

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
  player = WaypointPlayer([tag_standoff(t, d) for t, d in MISSION], loop=LOOP)
  last_wd = last_diag = last_pose_t = 0.0
  last_idx = -1
  v_prev = 0.0                                                # last commanded speed (for the accel ramp)
  last_t = time.monotonic()

  kv_put("watchdog", "navd", time.monotonic())
  logger.info("navd: mission " + " → ".join(
    f"{d:.1f}m@tag{t} {w.round(2).tolist()}" for (t, d), w in zip(MISSION, player.waypoints)))

  while True:
    sub.update(timeout=100)
    now = time.monotonic()
    dt = min(now - last_t, MAX_DT); last_t = now
    if now - last_wd > 1.0:
      kv_put("watchdog", "navd", now); last_wd = now

    pose = sub["slam_pose"]
    if sub.updated["slam_pose"]: last_pose_t = now
    # Don't drive on a stale pose, or before SLAM has ever anchored to a tag —
    # the world origin is meaningless until the first absolute fix (n_tags counts
    # fixes seen so far; a tighter cov_pos gate could replace this later).
    if pose is None or now - last_pose_t > STALE_TIMEOUT or pose["n_tags"] == 0:
      pub.send("chassis_velocity", {"x": 0.0, "y": 0.0}); v_prev = 0.0   # no localization ⇒ hold still
      if now - last_diag > 1.0:
        logger.info("navd: no usable slam_pose (stale or unanchored), stopped"); last_diag = now
      continue

    p_xy = np.asarray(pose["p_w"], np.float64)[:2]
    goal = player.update(p_xy, now)
    if player.idx != last_idx:
      logger.info(f"navd: waypoint {player.idx}/{len(MISSION)} " +
                  ("complete, holding" if goal is None else f"→ {goal.round(2).tolist()}"))
      last_idx = player.idx

    fwd = gimbal_heading(pose["q_wb"]) if goal is not None else None
    if goal is None or fwd is None:                          # mission done, or gimbal ~vertical
      pub.send("chassis_velocity", {"x": 0.0, "y": 0.0}); v_prev = 0.0
      continue
    left = (-fwd[1], fwd[0])

    # Trapezoidal speed profile along the straight line to the waypoint: cap at
    # V_MAX (cruise), brake on √(2·a·d) so we can always stop at the deadband, and
    # limit the rise to ACCEL (accel ramp). Recomputed from the live distance each
    # tick → self-correcting, and triangular automatically when the move is short.
    e = goal - p_xy                                          # world position error to the waypoint
    d = float(math.hypot(e[0], e[1]))
    v_target = min(V_MAX, math.sqrt(2.0 * ACCEL * max(0.0, d - POS_DEADBAND)))
    v_cmd = max(0.0, min(v_target, v_prev + ACCEL * dt))
    v_prev = v_cmd
    if v_cmd <= 0.0 or d < 1e-6:
      x = y = 0.0
    else:
      vx, vy = v_cmd * e[0] / d, v_cmd * e[1] / d            # world-frame velocity toward the goal
      x = vx * fwd[0] + vy * fwd[1]                          # → gimbal forward
      y = vx * left[0] + vy * left[1]                        # → gimbal left
    pub.send("chassis_velocity", {"x": x, "y": y})

    if now - last_diag > 1.0:
      logger.info(f"navd: wp{player.idx} d={d:.2f}m v={v_cmd:.2f}m/s → x={x:+.2f} y={y:+.2f}")
      last_diag = now
