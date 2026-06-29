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
command it directly. Chassis orientation is not controlled — position only.

While navigating we also point the gimbal at the best nearby tag (closest +
most face-on) so the SLAM camera keeps an absolute anchor in view. navd doesn't
drive the gimbal directly — it publishes a `nav_setpoint` and gimbald arbitrates
(aim_setpoint from decisiond outranks it) and runs the PID. The look-at runs even
while holding position, so localization stays anchored at a waypoint too.

Subs:  slam_pose:        {t, p_w, v_w, q_wb, cov_pos, n_tags}   (from slamd)
       gimbal_state:     {yaw_gi, ...}                          (latest; for the absolute yaw setpoint)
Pubs:  chassis_velocity: {x, y}   (x = forward, y = left, m/s; commsd → MOVE_ROBOT)
       nav_setpoint:     {yaw, pitch, yaw_ff, pitch_ff}         (gimbal target; gimbald → aim_error)
"""
import gc, os, json, math, time

import numpy as np

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ..common.geometry import wrap_pi
from ..common.gimbal import GimbalBuffer
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
LOOKAHEAD = 0.4              # m, pure-pursuit lookahead (larger = smoother, cuts corners more)
PROJECT_WINDOW = 1.5         # m, forward arc-length window when re-projecting progress onto the path

LOOK_AT_TAG = True           # point the gimbal at the best face-on tag to keep SLAM anchored
VIEW_MIN_ALIGN = 0.2         # min cos(tag-normal vs line-of-sight): skip tags seen too obliquely (poor PnP)
VIEW_MAX_RANGE = common.TAG_MAX_RANGE   # m, ignore tags beyond useful PnP range
LOOK_PITCH = 0.0             # rad, gravity-relative pitch setpoint (level; the SLAM cam is yaw-only anyway)

def load_nav_path():
  """A drawn path from the path_editor tool, via the NAV_PATH env var (JSON with a
  world-XY "path"). Returns (waypoints, is_curve) or None to fall back to MISSION."""
  p = os.environ.get("NAV_PATH")
  if not p: return None
  try:
    with open(p) as f: data = json.load(f)
  except (OSError, json.JSONDecodeError) as e:
    logger.warning(f"navd: NAV_PATH {p} unreadable ({e}); using built-in MISSION"); return None
  pts = [np.asarray(xy, np.float64) for xy in data.get("path", [])]
  return (pts, bool(data.get("curve", False))) if pts else None

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

class PurePursuit:
  """Follows a polyline path by chasing a lookahead point that slides along it.
  Holonomic → we just drive straight at the lookahead and cross-track error self-
  corrects. Progress `s` (arc length) only advances (monotonic), so it never snaps
  back to an earlier crossing. update() returns (lookahead_xy, brake_dist), where
  brake_dist = arc length left to the path end (∞ when looping → no end-braking, so
  it cruises). loop=True restarts at the end — use a CLOSED path so the seam sits at
  the start. done() = within POS_DEADBAND of the final point."""
  def __init__(self, path, lookahead=LOOKAHEAD, loop=False):
    self.P = np.asarray(path, np.float64)
    seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
    self.cum = np.concatenate([[0.0], np.cumsum(seg)])     # arc length at each vertex
    self.total = float(self.cum[-1])
    self.lookahead, self.loop = lookahead, loop
    self.s = 0.0

  def _point_at(self, s):
    s = s % self.total if self.loop else min(max(s, 0.0), self.total)
    j = max(0, min(int(np.searchsorted(self.cum, s)) - 1, len(self.P) - 2))
    seglen = self.cum[j + 1] - self.cum[j]
    t = 0.0 if seglen < 1e-9 else (s - self.cum[j]) / seglen
    return self.P[j] + t * (self.P[j + 1] - self.P[j])

  def update(self, p_xy):
    # Advance progress to the nearest path point ahead of `s`, within a forward
    # window (monotonic — ignore closer points behind us or far ahead).
    end = self.s + max(2.0 * self.lookahead, PROJECT_WINDOW)
    best_s, best_d2 = self.s, float("inf")
    for j in range(max(0, int(np.searchsorted(self.cum, self.s)) - 1), len(self.P) - 1):
      if self.cum[j] > end: break
      a, ab = self.P[j], self.P[j + 1] - self.P[j]
      L2 = float(ab @ ab)
      t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, float((p_xy - a) @ ab) / L2))
      s_proj = self.cum[j] + t * math.sqrt(L2)
      if s_proj < self.s: continue
      d2 = float(np.sum((p_xy - (a + t * ab)) ** 2))
      if d2 < best_d2: best_d2, best_s = d2, s_proj
    self.s = best_s
    if self.loop and self.s >= self.total - 1e-3: self.s = 0.0           # restart the lap
    brake = math.inf if self.loop else (self.total - self.s)
    return self._point_at(self.s + self.lookahead), brake

  def done(self, p_xy):
    return (not self.loop) and float(np.hypot(*(self.P[-1] - p_xy))) < POS_DEADBAND

def gimbal_heading(q_wb):
  """Horizontal unit forward vector of the gimbal in world, from q_wb=[w,x,y,z]
  (world<-body). forward = R_wb @ camera_z; returns None if it points ~straight
  up/down (heading undefined)."""
  w, x, y, z = q_wb
  fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
  n = math.hypot(fx, fy)
  return (fx / n, fy / n) if n > 1e-6 else None

def best_tag_to_view(p_xy:np.ndarray):
  """Pick the tag best to point the gimbal at for localization: face-on (its
  normal points toward us) AND close. Score = alignment / range² (≈ apparent
  projected tag area — the driver of PnP quality). Tags seen from the back
  (alignment ≤ 0) or too obliquely are skipped. Returns (tag_id, tag_xy, range)
  or None if nothing is worth looking at."""
  best, best_score = None, 0.0
  for tid, (R_wt, t_wt) in common.TAG_FIELD_MAP.items():
    t_xy = t_wt[:2].astype(np.float64)
    los = p_xy - t_xy                                # tag → robot
    rng = float(np.linalg.norm(los))
    if rng < 1e-3 or rng > VIEW_MAX_RANGE: continue
    n = R_wt[:2, 2].astype(np.float64); nn = float(np.linalg.norm(n))
    if nn < 1e-6: continue
    align = float(los @ n) / (rng * nn)              # 1 ⇒ robot on the tag's normal (dead face-on)
    if align < VIEW_MIN_ALIGN: continue
    score = align / (rng * rng)
    if score > best_score: best, best_score = (tid, t_xy, rng), score
  return best

def look_at_setpoint(p_xy:np.ndarray, v_xy:np.ndarray, fwd, yaw_gi_at_pose:float):
  """Gimbal setpoint (gimbal-inertial yaw) to point at the best tag, or None.
  Computes the bearing error in WORLD (frame-offset-free) and combines it with the
  gimbal yaw, so it's robust to the slam yaw-drift bias. yaw_gi_at_pose MUST be the gimbal
  yaw at the pose's capture time (same instant as `fwd`/`psi`) — `(yaw_gi − psi)` is the
  slowly-drifting world↔gimbal offset, and sampling the two at different times corrupts
  it by the gimbal's slew during the slam lag → the setpoint overshoots the tag."""
  bt = best_tag_to_view(p_xy)
  if bt is None: return None
  tid, t_xy, rng = bt
  los = t_xy - p_xy                                  # robot → tag (world)
  bearing = math.atan2(los[1], los[0])               # desired gimbal world heading
  psi = math.atan2(fwd[1], fwd[0])                   # gimbal world heading at the pose's capture time
  # PLUS: ψ tracks +yaw_gi (slam rotz, faithful) so the world bearing error adds straight
  # to the gimbal yaw. (This was −sign while the gimbal yaw DRIVE polarity was inverted in
  # commsd — the − accidentally stabilized gimbald's then-positive-feedback loop. With the
  # commsd aim_error.x flip fixing the drive at the source, it's the plain +.)
  yaw_sp = yaw_gi_at_pose + wrap_pi(bearing - psi)   # absolute gimbal-inertial yaw target
  # Feedforward: driving past a static tag drifts its bearing at
  # β̇ = (los_y·v_x − los_x·v_y)/range² — feed it so the gimbal doesn't lag.
  yaw_ff = float((los[1] * v_xy[0] - los[0] * v_xy[1]) / (rng * rng))
  return {"yaw": float(yaw_sp), "pitch": LOOK_PITCH, "yaw_ff": yaw_ff, "pitch_ff": 0.0}, tid

def run():
  gc.disable()
  pub = messaging.Pub(["chassis_velocity", "nav_setpoint"])
  sub = messaging.Sub(["slam_pose"], poll="slam_pose")
  # gimbal_state non-conflated + buffered so we can sample yaw_gi at the slam pose's
  # capture time — the SAME instant as q_wb/psi. (Latest yaw_gi vs a lagged q_wb-heading
  # corrupts the world↔gimbal offset during a slew → the look-at setpoint overshoots.)
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()
  # A drawn path (NAV_PATH, ≥2 pts) is followed smoothly with PurePursuit. The
  # tag-standoff MISSION uses WaypointPlayer instead — it WANTS to stop and hold at
  # each standoff, whereas pure pursuit flows straight through to the path end.
  np_path = load_nav_path()
  player = pursuit = None
  if np_path is not None and len(np_path[0]) >= 2:
    waypoints, is_curve = np_path
    pursuit = PurePursuit(waypoints, loop=LOOP)
    logger.info(f"navd: pure-pursuit NAV_PATH — {len(waypoints)} pts, {pursuit.total:.1f} m "
                f"({'curve' if is_curve else 'straight'}{', loop' if LOOP else ''})")
  else:
    player = WaypointPlayer([tag_standoff(t, d) for t, d in MISSION], loop=LOOP)
    logger.info("navd: mission " + " → ".join(
      f"{d:.1f}m@tag{t} {w.round(2).tolist()}" for (t, d), w in zip(MISSION, player.waypoints)))
  last_wd = last_diag = last_pose_t = 0.0
  last_idx = -1
  v_prev = 0.0                                                # last commanded speed (for the accel ramp)
  last_t = time.monotonic()

  kv_put("watchdog", "navd", time.monotonic())

  while True:
    sub.update(timeout=100)
    now = time.monotonic()
    dt = min(now - last_t, MAX_DT); last_t = now
    for m in gimbal_sub.drain("gimbal_state"): gimbal_buf.push(m)
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
    fwd = gimbal_heading(pose["q_wb"])

    # --- Gimbal look-at: point at the best face-on tag to keep SLAM anchored.
    # Runs whether or not the chassis is moving; stays silent (→ gimbald holds /
    # yields) when there's no good tag or no gimbal feedback yet.
    gp = gimbal_buf.interpolate(pose["t"])     # yaw_gi at the pose's capture time (consistent with fwd)
    look_tag = None
    if LOOK_AT_TAG and fwd is not None and gp is not None:
      res = look_at_setpoint(p_xy, np.asarray(pose["v_w"], np.float64)[:2], fwd, gp[0])
      if res is not None:
        sp, look_tag = res
        pub.send("nav_setpoint", sp)

    # --- Chassis nav: pick the target point to drive at + the distance to brake
    # against. PurePursuit → a lookahead point sliding along the path, braking on the
    # remaining arc length (stops only at the path end). WaypointPlayer → the active
    # waypoint, braking on straight-line distance (stops and holds at each).
    if pursuit is not None:
      target, brake = (None, 0.0) if pursuit.done(p_xy) else pursuit.update(p_xy)
      prog = f"path {100.0 * pursuit.s / pursuit.total:.0f}%" if pursuit.total > 0 else "path"
    else:
      goal = player.update(p_xy, now)
      if player.idx != last_idx:
        logger.info(f"navd: waypoint {player.idx}/{len(MISSION)} " +
                    ("complete, holding" if goal is None else f"→ {goal.round(2).tolist()}"))
        last_idx = player.idx
      target = goal
      brake = float(math.hypot(*(goal - p_xy))) if goal is not None else 0.0
      prog = f"wp{player.idx}"

    if target is None or fwd is None:                        # arrived/finished, or gimbal ~vertical
      pub.send("chassis_velocity", {"x": 0.0, "y": 0.0}); v_prev = 0.0
      continue
    left = (-fwd[1], fwd[0])

    # Trapezoidal speed profile on the brake distance: cap at V_MAX (cruise), brake on
    # √(2·a·d) so we can always stop at the deadband, and limit the rise to ACCEL.
    # Recomputed each tick → self-correcting (brake=∞ on a loop ⇒ steady cruise).
    v_target = min(V_MAX, math.sqrt(2.0 * ACCEL * max(0.0, brake - POS_DEADBAND)))
    v_cmd = max(0.0, min(v_target, v_prev + ACCEL * dt))
    v_prev = v_cmd
    to = target - p_xy                                       # world vector toward the target point
    dist = float(math.hypot(to[0], to[1]))
    if v_cmd <= 0.0 or dist < 1e-6:
      x = y = 0.0
    else:
      vx, vy = v_cmd * to[0] / dist, v_cmd * to[1] / dist    # world-frame velocity toward the target
      x = vx * fwd[0] + vy * fwd[1]                          # → gimbal forward
      y = vx * left[0] + vy * left[1]                        # → gimbal left
    pub.send("chassis_velocity", {"x": x, "y": y})

    if now - last_diag > 1.0:
      logger.info(f"navd: {prog} brake={brake:.2f}m v={v_cmd:.2f}m/s → x={x:+.2f} y={y:+.2f}"
                  + (f"  look@tag{look_tag}" if look_tag is not None else "  look:none"))
      last_diag = now
