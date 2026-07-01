"""Navigation daemon — drive the chassis to an injected goal via a planned, obstacle-aware path.

The goal is a world point set by fresh `nav_goal` messages (the state machine deciding "go home",
"center", a tag standoff, ...). navd plans an
any-angle path to it around static walls — and dynamic obstacles like other robots, TODO — on an
occupancy grid (`cv/nav`), and follows it with pure pursuit, REPLANNING each cycle (receding
horizon) so it reacts to a new goal or a changed map. With no map configured it drives straight
to the goal. No fresh goal => idle.

navd reads the robot's localized pose from slamd (`slam_pose`); no camera/PnP here —
detection feeds slamd's localization, navd just consumes the pose.

The chassis is holonomic; `chassis_velocity` is {x: forward, y: left} m/s referenced to the
GIMBAL heading (the board rotates chassis→gimbal). We rotate the world-frame velocity toward
the path's lookahead point into that gimbal frame and command it. Chassis orientation is not
controlled — position only.

navd does not command the gimbal. State-machine states own scan/acquire behavior while navigating, so
the chassis can move while decisiond keeps aiming/shooting.

Subs:  slam_pose:        {t, p_w, v_w, q_wb, cov_pos, n_tags}   (from slamd)
       nav_goal:         {x, y, label?}                         (fresh-only injected goal)
       plate:            {class, pos_gi, spin, ...}             (enemy robot → dynamic obstacle)
       gimbal_state:     {yaw_gi, ...}                          (latest; for the absolute yaw setpoint)
Pubs:  chassis_velocity: {x, y}   (x = forward, y = left, m/s; commsd → MOVE_ROBOT)
       nav_debug:        {goal, path, obstacles}                (for visual_slam)
Map:   NAV_MAP env → JSON {bounds:[x0,y0,x1,y1], res, robot_radius?, walls:[{rect|poly}]}  (optional)
"""
import gc, os, json, math, time

import numpy as np

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ..common.gimbal import GimbalBuffer
from ...slam import common
from ...nav.occupancy import OccupancyGrid, ROBOT_RADIUS
from ...nav import planner
from ...nav.obstacles import RobotObstacles

# No default goal -> navd HOLDS POSITION (idle, zero chassis velocity) until the state machine
# injects a nav_goal. Set to a (tag_id, standoff_m) tuple, e.g. (6, 2.5), only to drive somewhere
# standalone for testing.
DEFAULT_GOAL = None

POS_DEADBAND = 0.10          # m, pure-pursuit brakes to a stop here; "arrived" inside it
V_MAX = 3.0                  # m/s, trapezoid cruise speed (plateau / hard cap)
ACCEL = 1.8                  # m/s², trapezoid accel = decel ramp rate
MAX_DT = 0.2                 # s, clamp on the loop dt used for the accel ramp (guards stalls)
STALE_TIMEOUT = 0.5          # s without a fresh slam_pose before stopping
NAV_GOAL_TIMEOUT = 0.25      # s without fresh nav_goal before clearing the current destination
LOOKAHEAD = 0.4              # m, pure-pursuit lookahead (larger = smoother, cuts corners more)
PROJECT_WINDOW = 1.5         # m, forward arc-length window when re-projecting progress onto the path
PLAN_DT = 0.2                # s, replan period (5 Hz receding horizon → reacts to goal/map changes)
ENEMY_RADIUS = 0.30          # m, painted radius of a detected enemy robot obstacle (its half-footprint)
ENEMY_AGE_GROWTH = 0.4       # m per s of staleness — a stale detection inflates (the robot may have moved)
CAM_OCC_TIMEOUT = 0.5        # s without a fresh cam_occupancy before dropping occupancyd's obstacle layer

def load_map():
  """Static obstacle map for planning. NAV_MAP env → JSON (bounds, res, walls); with none,
  an empty grid over the 12×8 field (FIELD_BOUNDS) so dynamic obstacles still work. Returns
  (OccupancyGrid, robot_radius)."""
  p = os.environ.get("NAV_MAP")
  if not p:
    return OccupancyGrid(*common.FIELD_BOUNDS, 0.10), ROBOT_RADIUS
  try:
    with open(p) as f: data = json.load(f)
  except (OSError, json.JSONDecodeError) as e:
    logger.warning(f"navd: NAV_MAP {p} unreadable ({e}); empty field grid")
    return OccupancyGrid(*common.FIELD_BOUNDS, 0.10), ROBOT_RADIUS
  x0, y0, x1, y1 = data["bounds"]
  g = OccupancyGrid(x0, y0, x1, y1, data.get("res", 0.10))
  for w in data.get("walls", []):
    if "rect" in w: g.add_rect(*w["rect"])
    elif "poly" in w: g.add_poly(w["poly"])
  logger.info(f"navd: map {g.nx}x{g.ny} @ {g.res}m, {len(data.get('walls', []))} walls")
  return g, float(data.get("robot_radius", ROBOT_RADIUS))

def plan_path(static_grid, robot_radius, p_xy, goal_xy, robots=(), cam_occ=None):
  """Plan an obstacle-aware path from p_xy to goal_xy (world XY), or None if blocked. `robots`
  = (x, y, radius) circles for detected enemies, painted onto a copy before inflating. `cam_occ`
  = optional (ny, nx) bool of camera-detected occupancy (occupancyd), ORed in before inflating."""
  g = static_grid.copy()
  if cam_occ is not None: g.occ |= cam_occ
  for rx, ry, rad in robots:
    g.add_circle(rx, ry, rad)
  return planner.plan(g.inflated(robot_radius), p_xy, goal_xy)

def enemy_center_gi(plate):
  """Enemy robot position in the gimbal-inertial frame from a plate msg (x, y), or None for a
  lost/empty track. Uses the tracked plate position (≤ r off the body centre — fine for an obstacle)."""
  cls = plate.get("class")
  if cls in (None, "LOST", "UNKNOWN"): return None
  c = plate.get("pos_gi")
  return (float(c[0]), float(c[1])) if c is not None else None

def tag_standoff(tag_id:int, dist:float) -> np.ndarray:
  """World-XY point `dist` metres out along tag `tag_id`'s outward face normal.
  TAG_FIELD_MAP gives (R_world_tag, t_world_tag); the tag's +z column is the
  direction its face points into the play area."""
  R_wt, t_wt = common.TAG_FIELD_MAP[tag_id]
  n = R_wt[:2, 2]
  return (t_wt[:2] + dist * n / (np.linalg.norm(n) + 1e-9)).astype(np.float64)

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

def run():
  gc.disable()
  pub = messaging.Pub(["chassis_velocity", "nav_debug"])
  # nav_goal (injected destination), plate (enemy robot detections), and cam_occupancy (occupancyd's
  # camera obstacle grid) are non-polled latest-wins.
  sub = messaging.Sub(["slam_pose", "nav_goal", "plate", "cam_occupancy"], poll="slam_pose")
  # gimbal_state non-conflated + buffered so we can sample yaw_gi at the slam pose's
  # capture time — the SAME instant as q_wb/psi. (Latest yaw_gi vs a lagged q_wb-heading
  # corrupts the world↔gimbal offset during a slew → the look-at setpoint overshoots.)
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()
  obstacles = RobotObstacles()

  static_grid, robot_radius = load_map()
  injected = tag_standoff(*DEFAULT_GOAL) if DEFAULT_GOAL else None   # startup seed; nav_goal overrides
  inj_label = "default"
  logger.info(f"navd: goal mode {'[' + str(int(static_grid.occ.sum())) + ' wall cells]' if static_grid.occ.any() else '[open field]'}"
              + (f", seed {injected.round(2).tolist()}" if injected is not None else ", idle until nav_goal"))

  pursuit = last_goal = None
  cam_occ = None                                              # latest camera occupancy grid (ny, nx) bool
  last_cam_occ_t = -math.inf
  last_wd = last_diag = last_plan = last_pose_t = last_cam_warn = 0.0
  last_goal_t = time.monotonic() if injected is not None else -math.inf
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
    gp = gimbal_buf.interpolate(pose["t"])     # yaw_gi at the pose's capture time (consistent with fwd)

    # --- Dynamic obstacles: persist detected enemy robots in the world frame. plated gives
    # the enemy CENTER in gimbal-inertial; rotate into world by ψ0 = ψ_world − yaw_gi (the
    # gimbal-inertial→world azimuth, which is slew-INVARIANT so detection latency is fine).
    # TODO HW-verify: place a robot at a known spot, confirm the obstacle lands there.
    plate = sub["plate"]
    if plate is not None and sub.updated["plate"] and fwd is not None and gp is not None:
      c_gi = enemy_center_gi(plate)
      if c_gi is not None:
        psi0 = math.atan2(fwd[1], fwd[0]) - gp[0]
        cs, sn = math.cos(psi0), math.sin(psi0)
        obstacles.update(p_xy[0] + cs * c_gi[0] - sn * c_gi[1],
                         p_xy[1] + sn * c_gi[0] + cs * c_gi[1], now)

    # --- Camera occupancy: occupancyd publishes a world-frame obstacle grid (floor-vs-obstacle
    # IPM) over the same FIELD_BOUNDS/res as the planning grid. OR it in before inflation so
    # detection-independent obstacles (allies, extra enemies, debris) block the path too. Drop it
    # if the geometry doesn't match the planning grid, or once it goes stale (occupancyd died).
    co = sub["cam_occupancy"]
    if co is not None and sub.updated["cam_occupancy"]:
      if co["nx"] == static_grid.nx and co["ny"] == static_grid.ny and abs(co["res"] - static_grid.res) < 1e-9:
        cam_occ = np.frombuffer(co["occ"], bool).reshape(co["ny"], co["nx"])
        last_cam_occ_t = now
      elif now - last_cam_warn > 5.0:
        logger.warning("navd: cam_occupancy grid geometry mismatch with plan grid; ignoring"); last_cam_warn = now
    cam_occ_fresh = cam_occ if now - last_cam_occ_t < CAM_OCC_TIMEOUT else None

    # --- Chassis nav: plan an obstacle-aware path to the current goal (injected nav_goal)
    # and follow it with pure pursuit, replanning on goal/map change or the PLAN_DT timer
    # (receding horizon — each plan starts at the live pose, so pure pursuit follows between).
    ng = sub["nav_goal"]
    if ng is not None and sub.updated["nav_goal"]:
      injected = np.array([float(ng["x"]), float(ng["y"])]); inj_label = ng.get("label", "goal")
      last_goal_t = now
    if now - last_goal_t > NAV_GOAL_TIMEOUT:
      injected = None; pursuit = last_goal = None
    goal_xy = injected
    robots = [(x, y, ENEMY_RADIUS + ENEMY_AGE_GROWTH * age) for x, y, age in obstacles.active(now)]
    if goal_xy is not None and (last_goal is None or pursuit is None
                                or not np.allclose(goal_xy, last_goal) or now - last_plan > PLAN_DT):
      path = plan_path(static_grid, robot_radius, p_xy, goal_xy, robots, cam_occ_fresh)
      pursuit = PurePursuit(path) if path is not None and len(path) >= 2 else None
      if pursuit is None and now - last_diag > 1.0:
        logger.info(f"navd: no path to {np.round(goal_xy, 2).tolist()} ({inj_label}) — blocked"); last_diag = now
      last_goal = np.asarray(goal_xy, np.float64).copy(); last_plan = now

    pub.send("nav_debug", {"t": now,
      "goal": goal_xy.tolist() if goal_xy is not None else None,
      "path": pursuit.P.tolist() if pursuit is not None else [],
      "obstacles": [[x, y, r] for x, y, r in robots]})

    navigating = goal_xy is not None and pursuit is not None and fwd is not None and not pursuit.done(p_xy)

    if not navigating:                                        # idle/blocked/arrived ⇒ hold, gimbal free
      pub.send("chassis_velocity", {"x": 0.0, "y": 0.0}); v_prev = 0.0
      continue
    target, brake = pursuit.update(p_xy)
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
      logger.info(f"navd: goal:{inj_label} brake={brake:.2f}m v={v_cmd:.2f}m/s → x={x:+.2f} y={y:+.2f}")
      last_diag = now
