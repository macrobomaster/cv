import time, math
from dataclasses import dataclass
from collections import deque

import numpy as np
import cv2

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

PATH_WAYPOINTS = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
PATH_SPEED = 1
PATH_BLEND_RADIUS = 0.1

# -- aiming constants --

MAINTAIN_DIST = 2         # m, target follow distance
CHASE_SPEED = 2           # m/s max chassis speed
PROJECTILE_SPEED = 25     # m/s
GRAVITY = 9.81            # m/s^2
IMG_W, IMG_H = 512, 256
FOCAL_Y = 829.0           # camera focal length in pixels

# -- aiming components --

class AimErrorKF:
  """Kalman filter on aim error: state = [x, y, vx, vy, ax, ay].
  Provides smoothed position and velocity estimates for lead prediction."""
  def __init__(self, dt=1/100):
    self.dt = dt
    self.reset()

  def predict_and_correct(self, x, y):
    self.km.predict()
    est = self.km.correct(np.array([[x], [y]], dtype=np.float32)).flatten()
    return float(est[0]), float(est[1]), float(est[2]), float(est[3])

  def reset(self):
    self.km = cv2.KalmanFilter(6, 2, 0)
    self.km.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
    self.km.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
    self.km.errorCovPost = np.eye(6, dtype=np.float32)
    dt = self.dt
    F = np.eye(6, dtype=np.float32)
    F[0, 2] = dt; F[1, 3] = dt; F[2, 4] = dt; F[3, 5] = dt
    F[0, 4] = 0.5 * dt * dt; F[1, 5] = 0.5 * dt * dt
    self.km.transitionMatrix = F
    H = np.zeros((2, 6), dtype=np.float32)
    H[0, 0] = 1; H[1, 1] = 1
    self.km.measurementMatrix = H

class ShootDecision:
  """Burst fire when aim error stays small for a sustained window."""
  def __init__(self, threshold=0.25, burst_dur=0.5, cooldown=0.5):
    self.threshold = threshold
    self.burst_dur = burst_dur
    self.cooldown = cooldown
    self.window = deque(maxlen=10)
    self.burst_start = 0
    self.last_burst = 0

  def step(self, x, y):
    self.window.append(math.hypot(x, y))
    now = time.monotonic()
    if self.burst_start > 0:
      if now - self.burst_start > self.burst_dur:
        self.last_burst = now
        self.burst_start = 0
        return False
      return True
    if now - self.last_burst > self.cooldown and len(self.window) == self.window.maxlen:
      if sum(self.window) / len(self.window) < self.threshold:
        self.burst_start = now
    return False

def gravity_drop_offset(dist):
  """Normalized-coord y offset for bullet drop at given distance (meters).
  Uses projectile physics: drop = 0.5*g*t^2, projected back to image space."""
  if dist < 0.5:
    return 0.0
  tof = dist / PROJECTILE_SPEED
  drop = 0.5 * GRAVITY * tof * tof
  return (FOCAL_Y * drop / dist) / (IMG_H / 2)

# -- main loop --

def run():
  pub = messaging.Pub(["aim_error", "aim_angle", "chassis_velocity", "shoot", "spinning"])
  sub = messaging.Sub(["autoaim", "plate", "game_running", "team_color"], poll="autoaim")

  autoaim_valid_debounce = Debounce(1)
  aim_kf = AimErrorKF()
  shoot_decision = ShootDecision(threshold=3.0)  # degrees

  follower = WaypointFollower(
    waypoints=PATH_WAYPOINTS,
    speed=PATH_SPEED,
    blend_radius=PATH_BLEND_RADIUS,
    loop=False,
  )

  fk = FrequencyKeeper(100)
  cv_smooth = [0.0, 0.0]  # EMA-smoothed chassis velocity [x, z]

  st = time.monotonic()
  last_step_t = None
  while True:
    sub.update(timeout=0)

    autoaim = sub["autoaim"]
    plate = sub["plate"]
    has_target = False

    if autoaim is not None and sub.updated["autoaim"]:
      if autoaim["valid"] and plate is not None:
        has_target = True

        pos = plate["pos"]
        dist = plate["dist"]

        # aim using 3D angle to target (distance-independent)
        aim_x = math.degrees(math.atan2(pos[0], pos[2]))
        aim_y = math.degrees(math.atan2(pos[1], pos[2]))

        # lead prediction using 3D velocity from plate KF
        if dist > 0.5 and "vel" in plate:
          vel = plate["vel"]
          tof = dist / PROJECTILE_SPEED
          pred = (pos[0] + vel[0] * tof, pos[1] + vel[1] * tof, pos[2] + vel[2] * tof)
          aim_x = math.degrees(math.atan2(pred[0], pred[2]))
          aim_y = math.degrees(math.atan2(pred[1], pred[2]))

        # gravity drop compensation (aim higher)
        if dist > 0.5:
          tof = dist / PROJECTILE_SPEED
          drop = 0.5 * GRAVITY * tof * tof
          aim_y -= math.degrees(math.atan2(drop, dist))

        pub.send("aim_angle", {"x": aim_x, "y": aim_y})

        shoot = shoot_decision.step(aim_x, aim_y)

        # send angle-based aim error, scaled by distance
        x_err = (aim_x / max(1, dist)) * 0.5
        y_err = (aim_y / max(1, dist)) * 0.5
        pub.send("aim_error", {"x": x_err, "y": y_err})
        pub.send("shoot", shoot)

        # chassis: maintain distance to target
        cv_target = 0.0
        if dist > MAINTAIN_DIST + 0.1:
          cv_target = min(CHASE_SPEED, max(0, dist - MAINTAIN_DIST))
        elif dist < MAINTAIN_DIST - 0.1:
          cv_target = -min(CHASE_SPEED, MAINTAIN_DIST - min(MAINTAIN_DIST, dist))

        # smooth chassis velocity to prevent oscillation
        alpha = 0.1
        cv_smooth[0] += alpha * (cv_target - cv_smooth[0])
        pub.send("chassis_velocity", {"x": cv_smooth[0], "z": 0.0})
      else:
        pub.send("aim_error", {"x": 0.0, "y": 0.0})
        pub.send("spinning", True)  # keep alive = don't spin when no target

      if autoaim_valid_debounce.debounce(not autoaim["valid"]):
        aim_kf.reset()

    # # fall back to waypoint path when no target
    # if not has_target:
    #   now = time.monotonic()
    #   wall_dt = now - st
    #   if wall_dt > 9:
    #     logger.warning("STARTING")
    #   if wall_dt > 10 and wall_dt <= 120:
    #     if last_step_t is None:
    #       last_step_t = now
    #     dt = now - last_step_t
    #     last_step_t = now
    #     vx, vz = follower.step(dt)
    #     pub.send("chassis_velocity", {"x": vx, "z": vz})
    #   else:
    #     pub.send("chassis_velocity", {"x": 0.0, "z": 0.0})
    #
    # fk.step()
