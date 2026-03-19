import time, math
from dataclasses import dataclass

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ..core.helpers import Debounce, FrequencyKeeper

# -- path segment types --

@dataclass
class LineSegment:
  direction: tuple[float, float]
  speed: float
  length: float

  @property
  def duration(self) -> float:
    return self.length / self.speed

  def velocity(self, t: float) -> tuple[float, float]:
    return self.direction[0] * self.speed, self.direction[1] * self.speed

@dataclass
class ArcSegment:
  center: tuple[float, float]
  radius: float
  start_angle: float
  sweep: float  # signed: positive = CCW, negative = CW
  speed: float

  @property
  def length(self) -> float:
    return abs(self.radius * self.sweep)

  @property
  def duration(self) -> float:
    return self.length / self.speed

  def velocity(self, t: float) -> tuple[float, float]:
    frac = t / self.duration if self.duration > 0 else 0
    angle = self.start_angle + frac * self.sweep
    sign = 1 if self.sweep > 0 else -1
    vx = sign * -math.sin(angle) * self.speed
    vz = sign *  math.cos(angle) * self.speed
    return vx, vz

# -- waypoint follower with arc blending --

class WaypointFollower:
  def __init__(self, waypoints: list[tuple[float, float]], speed: float, blend_radius: float, loop: bool = False):
    self.segments = _build_path(waypoints, speed, blend_radius)
    self.total_duration = sum(s.duration for s in self.segments)
    self.elapsed = 0.0
    self.loop = loop

  def step(self, dt: float) -> tuple[float, float]:
    self.elapsed += dt
    if self.elapsed >= self.total_duration:
      if self.loop:
        self.elapsed %= self.total_duration
      else:
        return 0.0, 0.0
    t = self.elapsed
    for seg in self.segments:
      if t <= seg.duration:
        return seg.velocity(t)
      t -= seg.duration
    return 0.0, 0.0

def _build_path(waypoints: list[tuple[float, float]], speed: float, blend_radius: float):
  n = len(waypoints)
  assert n >= 2, "need at least 2 waypoints"

  # segment directions and lengths
  dirs: list[tuple[float, float]] = []
  lens: list[float] = []
  for i in range(n - 1):
    dx = waypoints[i + 1][0] - waypoints[i][0]
    dz = waypoints[i + 1][1] - waypoints[i][1]
    l = math.hypot(dx, dz)
    assert l > 1e-9, f"duplicate waypoints at index {i} and {i + 1}"
    dirs.append((dx / l, dz / l))
    lens.append(l)

  # blend info at each internal corner (indices 1..n-2)
  # tangent_dists[i] = tangent distance consumed at waypoint i (0 for endpoints)
  tangent_dists = [0.0] * n
  corner_info: list[dict | None] = [None] * n

  for i in range(1, n - 1):
    d1 = dirs[i - 1]
    d2 = dirs[i]

    cos_theta = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
    theta = math.acos(cos_theta)

    if theta < 1e-6 or theta > math.pi - 1e-6:
      continue  # nearly straight or U-turn, skip blend

    phi = math.pi - theta
    td = blend_radius / math.tan(phi / 2)

    # clamp so blends don't overlap on short segments
    max_td = min(lens[i - 1] / 2, lens[i] / 2)
    td = min(td, max_td)
    actual_r = td * math.tan(phi / 2)

    cross = d1[0] * d2[1] - d1[1] * d2[0]

    tangent_dists[i] = td
    corner_info[i] = {
      "td": td,
      "radius": actual_r,
      "theta": theta,
      "ccw": cross > 0,
      "d1": d1,
      "d2": d2,
    }

  # assemble segment list
  segments: list[LineSegment | ArcSegment] = []

  for i in range(n - 1):
    d = dirs[i]
    line_len = lens[i] - tangent_dists[i] - tangent_dists[i + 1]

    if line_len > 1e-6:
      segments.append(LineSegment(direction=d, speed=speed, length=line_len))

    # arc at the end-corner of this segment (waypoint i+1)
    ci = corner_info[i + 1]
    if ci is not None:
      p = waypoints[i + 1]
      d1 = ci["d1"]
      td = ci["td"]
      r = ci["radius"]
      ccw = ci["ccw"]

      # tangent point on incoming segment
      t1x = p[0] - d1[0] * td
      t1z = p[1] - d1[1] * td

      # normal toward arc center
      if ccw:
        nx, nz = -d1[1], d1[0]
      else:
        nx, nz = d1[1], -d1[0]

      cx = t1x + nx * r
      cz = t1z + nz * r

      start_angle = math.atan2(t1z - cz, t1x - cx)
      sweep = ci["theta"] if ccw else -ci["theta"]

      segments.append(ArcSegment(
        center=(cx, cz), radius=r,
        start_angle=start_angle, sweep=sweep, speed=speed,
      ))

  return segments

# -- path config --

PATH_WAYPOINTS = [(0, 0), (5.5, 0), (5.5, 1), (0, 1), (0, 2), (5.5, 2), (5.5, 3), (0, 3), (0, 4), (5, 4)]
PATH_SPEED = 1
PATH_BLEND_RADIUS = 0.5

# -- main loop --

def run():
  pub = messaging.Pub(["aim_error", "aim_angle", "chassis_velocity", "shoot"])
  sub = messaging.Sub(["autoaim", "plate", "game_running", "team_color"], poll="autoaim")

  follower = WaypointFollower(
    waypoints=PATH_WAYPOINTS,
    speed=PATH_SPEED,
    blend_radius=PATH_BLEND_RADIUS,
    loop=True,
  )

  fk = FrequencyKeeper(100)

  st = time.monotonic()
  last_step_t = None
  while True:
    sub.update(timeout=0)

    now = time.monotonic()
    wall_dt = now - st
    if wall_dt > 9:
      logger.warning("STARTING")
    if wall_dt > 10 and wall_dt <= 120:
      if last_step_t is None:
        last_step_t = now
      dt = now - last_step_t
      last_step_t = now
      vx, vz = follower.step(dt)
      pub.send("chassis_velocity", {"x": vx, "z": vz})
    else:
      pub.send("chassis_velocity", {"x": 0.0, "z": 0.0})

    fk.step()
