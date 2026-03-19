import time, math
from dataclasses import dataclass
from collections import deque

from ..core import messaging
from ..core.logging import logger
from ..core.helpers import Debounce, FrequencyKeeper
from ..plated.plated import CAMERA_MATRIX

MAINTAIN_DIST = 2
CHASE_SPEED = 2
PROJECTILE_SPEED = 25  # m/s
MAX_LEAD_PX = 80       # max lead offset in pixels, prevents runaway predictions

@dataclass(frozen=True)
class Waypoint:
  x: float
  z: float
  dt: float

class WaypointFollower:
  def __init__(self, waypoints:list[Waypoint]):
    self.waypoints = waypoints
    self.last_waypoint = Waypoint(0, 0, 0)
    self.cur_waypoint = self.waypoints.pop(0)
    self.dt_elapsed = self.cur_waypoint.dt
    self.elapsed = 0

  def step(self, dt:float) -> tuple[float, float]:
    self.elapsed += dt
    if not self.waypoints and self.elapsed > self.dt_elapsed:
      return 0, 0

    if self.elapsed > self.dt_elapsed:
      self.last_waypoint = self.cur_waypoint
      self.cur_waypoint = self.waypoints.pop(0)
      self.dt_elapsed += self.cur_waypoint.dt

    # compute velocity required to reach the waypoint in the dt
    dx = self.cur_waypoint.x - self.last_waypoint.x
    dz = self.cur_waypoint.z - self.last_waypoint.z
    vx = dx / self.cur_waypoint.dt
    vz = dz / self.cur_waypoint.dt
    return vx, vz

def compute_lead_offset(pos:tuple, vel:tuple, dist:float) -> tuple[float, float]:
  """Compute pixel-space lead offset to aim ahead of a moving target.

  Uses the 3D velocity from the plate Kalman filter and projectile time-of-flight
  to predict where the target will be when the projectile arrives. Returns the
  pixel offset between the current and predicted positions.
  """
  fx, fy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1]
  cx, cy = CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]

  px, py, pz = pos
  vx, vy, vz = vel

  tof = dist / PROJECTILE_SPEED

  # predicted position when projectile arrives
  pred_x = px + vx * tof
  pred_y = py + vy * tof
  pred_z = pz + vz * tof

  # guard against degenerate depth
  if pz < 0.1 or pred_z < 0.1:
    return 0.0, 0.0

  # project current and predicted positions to pixel space (pinhole model)
  cur_px = fx * px / pz + cx
  cur_py = fy * py / pz + cy
  pred_px = fx * pred_x / pred_z + cx
  pred_py = fy * pred_y / pred_z + cy

  delta_xc = pred_px - cur_px
  delta_yc = pred_py - cur_py

  # clamp to prevent runaway predictions from noisy velocity estimates
  lead_mag = math.sqrt(delta_xc * delta_xc + delta_yc * delta_yc)
  if lead_mag > MAX_LEAD_PX:
    scale = MAX_LEAD_PX / lead_mag
    delta_xc *= scale
    delta_yc *= scale

  return delta_xc, delta_yc

class ShootDecision:
  def __init__(self):
    self.window = deque(maxlen=10)

    # only shoot a 3 round burst
    self.burst_start = 0
    self.last_burst = 0

  def step(self, x:float, y:float) -> bool:
    # add distance to the window
    dist = math.sqrt(x*x + y*y)
    self.window.append(dist)

    now = time.monotonic()
    if self.burst_start > 0:
      if now - self.burst_start > 0.5:
        self.last_burst = now
        self.burst_start = 0
        return False
      else:
        # shoot
        return True
    else:
      if now - self.last_burst > 0.5:
        if len(self.window) == self.window.maxlen:
          avg = sum(self.window) / len(self.window)
          if avg < 0.25:
            self.burst_start = now
    return False

def run():
  pub = messaging.Pub(["aim_error", "aim_angle", "chassis_velocity", "shoot"])
  sub = messaging.Sub(["autoaim", "plate"], poll="autoaim")

  autoaim_valid_debounce = Debounce(1)
  shoot_decision = ShootDecision()

  follower = WaypointFollower([
    Waypoint(6.2, 0, 6),
    Waypoint(6.2, 6.2, 6),
    Waypoint(0, 6.2, 6),
    Waypoint(0, 0, 6),
  ])

  fk = FrequencyKeeper(200)

  ste = time.monotonic()
  st = time.monotonic()
  while True:
    sub.update()

    autoaim = sub["autoaim"]
    if autoaim is None: continue
    plate = sub["plate"]
    if plate is None: continue

    if sub.updated["autoaim"]:
      if autoaim["valid"]:
        plate_mu = autoaim["plate_mu"]
        xc, yc = plate_mu[0], plate_mu[1]

        # aim ahead of moving targets based on projectile time-of-flight
        if "vel" in plate:
          lead_x, lead_y = compute_lead_offset(plate["pos"], plate["vel"], plate["dist"])
          xc += lead_x
          yc += lead_y

        x = (xc - 256) / 256
        y = (yc - 128) / 128

        # offset y by some amount relative to the distance to the plate
        y -= 0.1 * plate["dist"]
        y += 0.7

        shoot = shoot_decision.step(x, y)

        # scale error based on distance
        x = x / max(1, plate["dist"])
        y = y / max(1, plate["dist"])

        pub.send("aim_error", {"x": x * 0.5, "y": y * 0.5})
        pub.send("shoot", shoot)

        chassis_velocity = {"x": 0.0, "z": 0.0}
        if plate["dist"] > MAINTAIN_DIST + 0.1:
          chassis_velocity["x"] = min(CHASE_SPEED, max(0, plate["dist"] - MAINTAIN_DIST))
        elif plate["dist"] < MAINTAIN_DIST - 0.1:
          chassis_velocity["x"] = -min(CHASE_SPEED, MAINTAIN_DIST - min(MAINTAIN_DIST, plate["dist"]))

        pos = plate["pos"]

        # compute angle on xz plane
        angle_x = math.degrees(math.atan2(pos[2], pos[0])) - 87
        # compute angle on yz plane
        angle_y = math.degrees(math.atan2(pos[1], pos[2]))
        pub.send("aim_angle", {"x": angle_x, "y": angle_y})

        if angle_x > 0.5:
          chassis_velocity["z"] = min(CHASE_SPEED, abs(angle_x) / 5)
        elif angle_x < -0.5:
          chassis_velocity["z"] = -min(CHASE_SPEED, abs(angle_x) / 5)

        pub.send("chassis_velocity", chassis_velocity)
      else:
        pub.send("shoot", False)

      if autoaim_valid_debounce.debounce(not autoaim["valid"]):
        pass
