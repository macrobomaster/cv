"""Reusable stated state building blocks.

Concrete match behavior belongs in states.py. This file holds state classes that encode common mechanics:
resetting, navigation to a goal, periodic navigation, timed holds, acquisition hysteresis, and scanning.
"""
import os, json, math, time

from ..core.logging import logger
from ..common.geometry import wrap_pi
from ...nav.occupancy import OccupancyGrid
from ...nav.planner import line_of_sight
from ...slam import common
from .state_machine import StateBase

_TAG_LOOK_GRID = None
def _field_grid():
  g = OccupancyGrid(*common.FIELD_BOUNDS, 0.10)
  for r in common.FIELD_WALLS: g.add_rect(*r)
  return g

def tag_look_grid():
  global _TAG_LOOK_GRID
  if _TAG_LOOK_GRID is not None: return _TAG_LOOK_GRID
  p = os.environ.get("NAV_MAP")
  if not p:
    _TAG_LOOK_GRID = _field_grid()
    return _TAG_LOOK_GRID
  try:
    with open(p) as f: data = json.load(f)
    x0, y0, x1, y1 = data["bounds"]
    g = OccupancyGrid(x0, y0, x1, y1, data.get("res", 0.10))
    for w in data.get("walls", []):
      if "rect" in w: g.add_rect(*w["rect"])
      elif "poly" in w: g.add_poly(w["poly"])
    _TAG_LOOK_GRID = g
  except (OSError, json.JSONDecodeError, KeyError) as e:
    logger.warning(f"stated: NAV_MAP {p!r} unreadable for tag look-at ({e}); using FIELD_WALLS grid")
    _TAG_LOOK_GRID = _field_grid()
  return _TAG_LOOK_GRID

class NavToGoalState(StateBase):
  name = "nav_to_goal"
  goal_label = "goal"
  goal_xy: tuple[float, float]|None = None
  arrive_radius = 0.25

  def __init__(self, goal_xy:tuple[float, float]|None=None):
    self.goal_xy = goal_xy
    self.done = False

  def reset(self, ctx=None):
    self.done = False

  def observe(self, ctx):
    if not bool(ctx["game_running"]): self.reset(ctx)

  def arrived(self, ctx) -> bool:
    if self.goal_xy is None: return True
    pose = ctx["slam_pose"]
    if pose is None: return False
    px, py = pose["p_w"][:2]
    return math.hypot(float(px) - self.goal_xy[0], float(py) - self.goal_xy[1]) < self.arrive_radius

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.goal_xy is not None and not self.done

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    if self.arrived(ctx):
      self.done = True
      return True
    return False

  def run(self, ctx, pub):
    if ctx.entered: logger.info(f"stated: navigating to {self.goal_label}")
    pub.send("nav_goal", {"x": self.goal_xy[0], "y": self.goal_xy[1], "label": self.goal_label})

class PeriodicNavToGoalState(NavToGoalState):
  period = math.inf
  first_delay = 0.0

  def __init__(self, goal_xy:tuple[float, float]|None=None):
    super().__init__(goal_xy)
    self.next_due = time.monotonic() + self.first_delay

  def reset(self, ctx=None):
    super().reset(ctx)
    self.next_due = (ctx.now if ctx is not None else time.monotonic()) + self.first_delay

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.goal_xy is not None and ctx.now >= self.next_due

  def mark_done(self, ctx):
    self.next_due = ctx.now + self.period

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    if self.arrived(ctx):
      self.mark_done(ctx)
      return True
    return False

class TimedNavToGoalState(NavToGoalState):
  trigger_at = math.inf
  preempt = True

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.goal_xy is not None and not self.done and ctx.match_elapsed >= self.trigger_at

class ScanAcquireMixin:
  acquire_hold_dt = 0.8
  sweep_amplitude = math.radians(45.0)
  sweep_dt = 1.0
  turn_dt = 0.5
  pitch = 0.0

  def _init_scan_acquire(self):
    self._scan_acquire_last_valid_t = -math.inf
    self._scan_acquire_center = 0.0
    self._scan_acquire_start_t = time.monotonic()
    self._scan_acquire_have_center = False

  def _reset_scan_acquire(self, ctx=None):
    self._scan_acquire_last_valid_t = -math.inf
    self._scan_acquire_start_t = ctx.now if ctx is not None else time.monotonic()
    self._scan_acquire_have_center = False

  def _observe_scan_acquire(self, ctx):
    if not bool(ctx["game_running"]):
      self._reset_scan_acquire(ctx)
      return
    autoaim = ctx["autoaim"]
    if ctx.updated["autoaim"] and autoaim is not None and autoaim.get("valid", False): self._scan_acquire_last_valid_t = ctx.now

  def _recently_saw_plate_scan_acquire(self, now:float) -> bool:
    return now - self._scan_acquire_last_valid_t < self.acquire_hold_dt

  def _scan_acquire_setpoint(self, t:float) -> dict:
    cycle_dt = self.sweep_dt + self.turn_dt
    t_scan = t - self._scan_acquire_start_t
    cycle = int(t_scan // cycle_dt)
    t_cycle = t_scan - cycle * cycle_dt
    base = self._scan_acquire_center + (cycle % 2) * math.pi
    if t_cycle < self.sweep_dt:
      w = 2 * math.pi / self.sweep_dt
      yaw = wrap_pi(base + self.sweep_amplitude * math.sin(w * t_cycle))
      yaw_ff = self.sweep_amplitude * w * math.cos(w * t_cycle)
    else:
      yaw_ff = math.pi / self.turn_dt
      yaw = wrap_pi(base + yaw_ff * (t_cycle - self.sweep_dt))
    return {"yaw": yaw, "pitch": self.pitch, "yaw_ff": yaw_ff, "pitch_ff": 0.0}

  def _run_scan_acquire(self, ctx, pub):
    if self._recently_saw_plate_scan_acquire(ctx.now): return
    gs = ctx["gimbal_state"]
    if gs is not None and (ctx.entered or not self._scan_acquire_have_center):
      self._scan_acquire_center = gs["yaw_gi"]
      self._scan_acquire_have_center = True
    if ctx.entered: self._scan_acquire_start_t = ctx.now
    pub.send("state_setpoint", self._scan_acquire_setpoint(ctx.now))

def _ctx_get(ctx, service:str):
  try: return ctx[service]
  except (KeyError, TypeError): return None

def _segment_intersects_rect(a, b, rect):
  x0, y0, x1, y1 = rect
  xmin, xmax = min(x0, x1), max(x0, x1)
  ymin, ymax = min(y0, y1), max(y0, y1)
  ax, ay = float(a[0]), float(a[1])
  bx, by = float(b[0]), float(b[1])
  dx, dy = bx - ax, by - ay
  t0, t1 = 0.0, 1.0
  for p, q in ((-dx, ax - xmin), (dx, xmax - ax), (-dy, ay - ymin), (dy, ymax - ay)):
    if abs(p) < 1e-12:
      if q < 0.0: return False
      continue
    r = q / p
    if p < 0.0:
      if r > t1: return False
      t0 = max(t0, r)
    else:
      if r < t0: return False
      t1 = min(t1, r)
  return t1 >= t0 and t1 > 1e-6 and t0 < 1.0 - 1e-6

def _field_wall_los(a, b):
  return not any(_segment_intersects_rect(a, b, rect) for rect in common.FIELD_WALLS)

class TagScanMixin:
  tag_scan_pitch = 0.0
  tag_scan_slow_rate = math.radians(45.0)
  tag_scan_fast_rate = math.radians(360.0)
  tag_scan_fast_arc = max(math.radians(20.0), min(math.radians(60.0),
                                                0.85 * 2.0 * math.atan(common.IMG_W / (2.0 * common.FX))))
  tag_track_lost_dt = 1.0

  def _init_tag_scan(self):
    self._tag_scan_start_yaw = 0.0
    self._tag_scan_start_t = time.monotonic()
    self._tag_scan_have_start = False
    self._tag_scan_last_seen_t = -math.inf
    self._tag_scan_track_id = None
    self._tag_scan_track_yaw = None
    self._tag_scan_track_pitch = self.tag_scan_pitch
    self._tag_scan_was_tracking = False

  def _reset_tag_scan(self, ctx=None):
    self._tag_scan_start_t = ctx.now if ctx is not None else time.monotonic()
    self._tag_scan_have_start = False
    self._tag_scan_last_seen_t = -math.inf
    self._tag_scan_track_id = None
    self._tag_scan_track_yaw = None
    self._tag_scan_track_pitch = self.tag_scan_pitch
    self._tag_scan_was_tracking = False

  def _tag_detection_center_area(self, det):
    corners = det.get("corners") if isinstance(det, dict) else None
    if corners is None or len(corners) < 4: return None
    pts = [(float(p[0]), float(p[1])) for p in corners[:4]]
    u = sum(p[0] for p in pts) / 4.0
    v = sum(p[1] for p in pts) / 4.0
    area = 0.0
    for i, (x0, y0) in enumerate(pts):
      x1, y1 = pts[(i + 1) % len(pts)]
      area += x0 * y1 - x1 * y0
    return u, v, abs(area) * 0.5

  def _select_tag_detection(self, detections):
    best = tracked = None
    for det in detections:
      c = self._tag_detection_center_area(det)
      if c is None: continue
      try: tag_id = int(det.get("id"))
      except (TypeError, ValueError): tag_id = None
      cand = (c[2], tag_id, c)
      if tag_id is not None and tag_id == self._tag_scan_track_id: tracked = cand
      if best is None or cand[0] > best[0]: best = cand
    return tracked or best

  def _tag_visual_track_setpoint(self, gs, center_area):
    u, v, _ = center_area
    yaw_sign = -1.0 if common.YAW_FLIPPED else 1.0
    yaw_err = -yaw_sign * math.atan2(u - common.CX, common.FX)
    pitch_err = -math.atan2(v - common.CY, common.FY)
    pitch_now = float(gs.get("pitch_gi", self.tag_scan_pitch))
    return wrap_pi(float(gs["yaw_gi"]) + yaw_err), pitch_now + pitch_err

  def _heading_world_tag_scan(self, q_wb):
    w, x, y, z = q_wb
    fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
    n = math.hypot(fx, fy)
    return None if n < 1e-6 else math.atan2(fy, fx)

  def _tag_world_track_setpoint(self, ctx, gs):
    if self._tag_scan_track_id is None: return None
    entry = common.TAG_FIELD_MAP.get(self._tag_scan_track_id)
    pose = _ctx_get(ctx, "slam_pose")
    if entry is None or pose is None: return None
    if pose.get("n_tags", 0) == 0: return None
    heading_w = self._heading_world_tag_scan(pose["q_wb"])
    if heading_w is None: return None
    _, t_wt = entry
    px, py = pose["p_w"][:2]
    dx, dy = float(t_wt[0]) - float(px), float(t_wt[1]) - float(py)
    d2 = dx * dx + dy * dy
    if d2 < 1e-6: return None
    yaw_sign = -1.0 if common.YAW_FLIPPED else 1.0
    bearing_w = math.atan2(dy, dx)
    yaw_rel = yaw_sign * wrap_pi(bearing_w - heading_w)
    vx, vy = pose.get("v_w", [0.0, 0.0])[:2]
    yaw_ff = yaw_sign * (dy * float(vx) - dx * float(vy)) / d2
    return {"yaw": wrap_pi(float(gs["yaw_gi"]) + yaw_rel), "pitch": self.tag_scan_pitch,
            "yaw_ff": yaw_ff, "pitch_ff": 0.0}

  def _tag_scan_track_occluded(self, ctx):
    if self._tag_scan_track_id is None: return False
    entry = common.TAG_FIELD_MAP.get(self._tag_scan_track_id)
    pose = _ctx_get(ctx, "slam_pose")
    if entry is None or pose is None: return False
    _, t_wt = entry
    px, py = pose["p_w"][:2]
    return not _field_wall_los((px, py), (float(t_wt[0]), float(t_wt[1])))

  def _observe_tag_scan(self, ctx):
    if not bool(ctx["game_running"]):
      self._reset_tag_scan(ctx)
      return
    tags = _ctx_get(ctx, "apriltags")
    gs = _ctx_get(ctx, "gimbal_state")
    if (getattr(ctx, "updated", {}).get("apriltags") and tags is not None
        and tags.get("detections") and gs is not None):
      selected = self._select_tag_detection(tags["detections"])
      if selected is None: return
      _, tag_id, center_area = selected
      self._tag_scan_track_id = tag_id
      self._tag_scan_track_yaw, self._tag_scan_track_pitch = self._tag_visual_track_setpoint(gs, center_area)
      self._tag_scan_last_seen_t = ctx.now

  def _tag_scan_setpoint(self, t:float) -> dict:
    fast_arc = min(max(self.tag_scan_fast_arc, math.radians(5.0)), math.pi - math.radians(5.0))
    slow_arc = math.pi - fast_arc
    slow_rate = max(self.tag_scan_slow_rate, 1e-6)
    fast_rate = max(self.tag_scan_fast_rate, slow_rate)
    slow_dt, fast_dt = slow_arc / slow_rate, fast_arc / fast_rate
    cycle_dt = slow_dt + fast_dt
    phase = max(0.0, t - self._tag_scan_start_t)
    cycle = int(phase // cycle_dt)
    t_cycle = phase - cycle * cycle_dt
    base = self._tag_scan_start_yaw + cycle * math.pi
    if t_cycle < slow_dt:
      yaw = base + slow_rate * t_cycle
      yaw_ff = slow_rate
    else:
      yaw = base + slow_arc + fast_rate * (t_cycle - slow_dt)
      yaw_ff = fast_rate
    return {"yaw": wrap_pi(yaw), "pitch": self.tag_scan_pitch, "yaw_ff": yaw_ff, "pitch_ff": 0.0}

  def _run_tag_scan(self, ctx, pub):
    gs = _ctx_get(ctx, "gimbal_state")
    if gs is None: return
    if ctx.entered or not self._tag_scan_have_start:
      self._tag_scan_start_yaw = float(gs["yaw_gi"])
      self._tag_scan_start_t = ctx.now
      self._tag_scan_have_start = True
      self._tag_scan_was_tracking = False
    if self._tag_scan_track_yaw is not None and ctx.now - self._tag_scan_last_seen_t < self.tag_track_lost_dt:
      self._tag_scan_was_tracking = True
      # If the mapped tag is now behind a wall, drop the track immediately instead of
      # continuing to aim at its stale visual center through geometry.
      if self._tag_scan_track_occluded(ctx):
        self._tag_scan_start_yaw = float(gs["yaw_gi"])
        self._tag_scan_start_t = ctx.now
        self._tag_scan_track_id = None
        self._tag_scan_track_yaw = None
        self._tag_scan_track_pitch = self.tag_scan_pitch
        self._tag_scan_was_tracking = False
      else:
        sp = self._tag_world_track_setpoint(ctx, gs)
        if sp is None:
          sp = {"yaw": self._tag_scan_track_yaw, "pitch": self._tag_scan_track_pitch, "yaw_ff": 0.0, "pitch_ff": 0.0}
        pub.send("state_setpoint", sp)
        return
    if self._tag_scan_was_tracking:
      self._tag_scan_start_yaw = float(gs["yaw_gi"])
      self._tag_scan_start_t = ctx.now
      self._tag_scan_track_id = None
      self._tag_scan_track_yaw = None
      self._tag_scan_track_pitch = self.tag_scan_pitch
      self._tag_scan_was_tracking = False
    pub.send("state_setpoint", self._tag_scan_setpoint(ctx.now))

class LookAtMappedTagMixin:
  tag_pitch = 0.0
  tag_los_offset = 0.05
  tag_max_range = common.TAG_MAX_RANGE
  tag_max_view_angle = math.radians(75.0)
  tag_yaw_change_weight = 0.5
  tag_view_angle_weight = 0.5
  tag_distance_weight = 1.0
  tag_occluded_penalty = 0.75
  tag_switch_margin = 0.30
  tag_max_yaw_rel = math.radians(120.0)
  tag_max_switch_yaw = math.radians(120.0)
  tag_lost_hold_dt = 0.5
  tag_motion_seed_min_speed = 0.15

  def _init_look_at_tag(self):
    self._look_at_tag_id = None
    self._look_at_tag_pos = None
    self._look_at_heading_w = None
    self._look_at_tag_yaw_rel = 0.0
    self._look_at_tag_t = -math.inf

  def _reset_look_at_tag(self, ctx=None):
    self._look_at_tag_id = None
    self._look_at_tag_pos = None
    self._look_at_heading_w = None
    self._look_at_tag_yaw_rel = 0.0
    self._look_at_tag_t = -math.inf

  def _observe_look_at_tag(self, ctx):
    if not bool(ctx["game_running"]): self._reset_look_at_tag(ctx)

  def _heading_world(self, q_wb):
    w, x, y, z = q_wb
    fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
    n = math.hypot(fx, fy)
    return None if n < 1e-6 else math.atan2(fy, fx)

  def _motion_heading_world(self, pose):
    vx, vy = pose.get("v_w", [0.0, 0.0])[:2]
    speed = math.hypot(float(vx), float(vy))
    return None if speed < self.tag_motion_seed_min_speed else math.atan2(float(vy), float(vx))

  def _tag_los(self, grid, px, py, tx, ty, nx, ny):
    a = grid.world_to_cell(px, py)
    b = grid.world_to_cell(tx + nx * self.tag_los_offset, ty + ny * self.tag_los_offset)
    return line_of_sight(grid, a, b)

  def _select_tag(self, pose, heading_w:float, score_heading_w:float):
    p = pose["p_w"]
    px, py = float(p[0]), float(p[1])
    grid = tag_look_grid()
    best = current = None
    min_view_cos = math.cos(self.tag_max_view_angle)
    for tag_id, (R_wt, t_wt) in common.TAG_FIELD_MAP.items():
      tx, ty, tz = float(t_wt[0]), float(t_wt[1]), float(t_wt[2])
      nx, ny = float(R_wt[0, 2]), float(R_wt[1, 2])
      dx, dy = tx - px, ty - py
      d2 = dx * dx + dy * dy
      if d2 < 1e-6 or d2 > self.tag_max_range * self.tag_max_range: continue
      dist = math.sqrt(d2)
      view_cos = ((px - tx) * nx + (py - ty) * ny) / dist
      if view_cos < min_view_cos: continue                  # too oblique / behind the tag face
      occluded = not self._tag_los(grid, px, py, tx, ty, nx, ny)
      bearing_w = math.atan2(dy, dx)
      yaw_sign = -1.0 if common.YAW_FLIPPED else 1.0
      yaw_rel = yaw_sign * wrap_pi(bearing_w - heading_w)
      if abs(yaw_rel) > self.tag_max_yaw_rel: continue
      yaw_score = wrap_pi(bearing_w - score_heading_w)
      score = self.tag_distance_weight * dist
      score += self.tag_yaw_change_weight * abs(yaw_score) + self.tag_view_angle_weight * (1.0 - view_cos)
      if occluded: score += self.tag_occluded_penalty
      cand = (score, int(tag_id), (tx, ty, tz), bearing_w, yaw_rel)
      if tag_id == self._look_at_tag_id: current = cand
      if best is None or score < best[0]: best = cand
    if current is not None and best is not None and current[0] <= best[0] + self.tag_switch_margin:
      return current
    if current is not None and best is not None and self._look_at_heading_w is not None:
      if abs(wrap_pi(best[3] - self._look_at_heading_w)) > self.tag_max_switch_yaw:
        return current
    return best

  def _run_look_at_tag(self, ctx, pub):
    pose, gs = ctx["slam_pose"], ctx["gimbal_state"]
    if pose is None or gs is None: return
    heading_w = self._heading_world(pose["q_wb"])
    if heading_w is None: return
    if ctx.entered: self._reset_look_at_tag(ctx)
    if self._look_at_heading_w is None:
      motion_heading_w = self._motion_heading_world(pose)
      self._look_at_heading_w = heading_w if motion_heading_w is None else motion_heading_w
    best = self._select_tag(pose, heading_w, self._look_at_heading_w)
    if best is None:
      if self._look_at_tag_pos is None or ctx.now - self._look_at_tag_t > self.tag_lost_hold_dt:
        self._reset_look_at_tag(ctx)
        return
    else:
      _, tag_id, self._look_at_tag_pos, self._look_at_heading_w, yaw_rel = best
      self._look_at_tag_id = tag_id
      self._look_at_tag_yaw_rel = yaw_rel
      self._look_at_tag_t = ctx.now
    px, py = pose["p_w"][:2]
    tx, ty, _ = self._look_at_tag_pos
    dx, dy = tx - float(px), ty - float(py)
    if dx * dx + dy * dy < 1e-6: return
    yaw_sign = -1.0 if common.YAW_FLIPPED else 1.0
    bearing_w = math.atan2(dy, dx)
    yaw_rel = yaw_sign * wrap_pi(bearing_w - heading_w)
    self._look_at_heading_w = bearing_w
    self._look_at_tag_yaw_rel = yaw_rel
    yaw = wrap_pi(float(gs["yaw_gi"]) + yaw_rel)
    vx, vy = pose.get("v_w", [0.0, 0.0])[:2]
    yaw_ff = yaw_sign * (dy * float(vx) - dx * float(vy)) / max(dx * dx + dy * dy, 1e-6)
    pub.send("state_setpoint", {"yaw": yaw, "pitch": self.tag_pitch, "yaw_ff": yaw_ff, "pitch_ff": 0.0})

class SpinChassisMixin:
  def _run_chassis_spin(self, ctx, pub):
    if bool(ctx["game_running"]): pub.send("spinning", True)

class SequenceState(StateBase):
  name = "sequence"

  def __init__(self, states:list[StateBase]):
    self.states = states
    self.idx = 0
    self.child_started = False

  def reset(self, ctx=None):
    self.idx = 0
    self.child_started = False
    for state in self.states:
      state.reset_state(ctx)

  def observe(self, ctx):
    if not bool(ctx["game_running"]):
      self.reset(ctx)
      return
    if self.idx < len(self.states): self.states[self.idx].observe_state(ctx)

  def should_transition(self, current:StateBase, ctx) -> bool:
    return False

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    if self.idx >= len(self.states): return True
    if self.child_started and self.states[self.idx].can_transition(ctx):
      self.idx += 1
      self.child_started = False
      if self.idx >= len(self.states):
        self.on_complete(ctx)
        return True
    return False

  def on_complete(self, ctx):
    pass

  def run(self, ctx, pub):
    if ctx.entered: self.reset(ctx)
    if self.idx >= len(self.states): return
    child = self.states[self.idx]
    child_entered = not self.child_started
    self.child_started = True
    if child_entered: child.observe_state(ctx)
    machine_entered = ctx.entered
    ctx.entered = child_entered
    child.run_state(ctx, pub)
    ctx.entered = machine_entered

class PeriodicSequenceState(SequenceState):
  period = math.inf
  first_delay = 0.0

  def __init__(self, states:list[StateBase]):
    super().__init__(states)
    self.next_due = time.monotonic() + self.first_delay

  def reset(self, ctx=None):
    super().reset(ctx)
    self.next_due = (ctx.now if ctx is not None else time.monotonic()) + self.first_delay

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and ctx.now >= self.next_due

  def on_complete(self, ctx):
    self.next_due = ctx.now + self.period

class TimedHoldGoalState(StateBase):
  name = "timed_hold_goal"
  goal_label = "hold"
  goal_xy: tuple[float, float]|None = None
  hold_dt = 1.0

  def __init__(self, goal_xy:tuple[float, float]|None=None):
    self.goal_xy = goal_xy
    self.until = -math.inf

  def reset(self, ctx=None):
    self.until = -math.inf

  def should_transition(self, current:StateBase, ctx) -> bool:
    return False

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    return ctx.now >= self.until

  def run(self, ctx, pub):
    if ctx.entered:
      self.until = ctx.now + self.hold_dt
      logger.info(f"stated: holding {self.goal_label}")
    pub.send("nav_goal", {"x": self.goal_xy[0], "y": self.goal_xy[1], "label": self.goal_label})

class AcquireHoldState(StateBase):
  name = "acquire"
  hold_dt = 0.8

  def __init__(self):
    self.last_valid_t = -math.inf

  def reset(self, ctx=None):
    self.last_valid_t = -math.inf

  def observe(self, ctx):
    if not bool(ctx["game_running"]):
      self.reset(ctx)
      return
    autoaim = ctx["autoaim"]
    if ctx.updated["autoaim"] and autoaim is not None and autoaim.get("valid", False): self.last_valid_t = ctx.now

  def recently_saw_plate(self, now:float) -> bool:
    return now - self.last_valid_t < self.hold_dt

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.recently_saw_plate(ctx.now)

  def can_transition(self, ctx) -> bool:
    return (not bool(ctx["game_running"])) or not self.recently_saw_plate(ctx.now)

  def run(self, ctx, pub):
    pass
