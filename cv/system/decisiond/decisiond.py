import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from ..core import messaging
from ..core.logging import logger
from ..core.helpers import FrequencyKeeper
from ...autoaim.common import (
  MUZZLE_VELOCITY, GRAVITY,
  DELTA_INPUT, DELTA_TRIGGER, GIMBAL_TAU, GIMBAL_OMEGA_MAX,
)

# Chassis
MAINTAIN_DIST = 2.0
CHASE_SPEED = 2.0

# Aim / trigger tunings
TOL_STATIC = math.radians(0.7)        # |aim_error| below this to commit a STATIC/LINEAR shot
TOL_SPIN = math.radians(1.0)          # tighter angular hold for SPIN
N_TICKS_FIRE = 3                       # consecutive in-tolerance ticks before STATIC/LINEAR fire
COOLDOWN_STATIC = 0.20                # s between non-spin shots
E_WINDOW = 0.040                       # m, hit window for spin trigger
E_WINDOW_OMEGA_K = 2.0                 # shrink window by k·r·σ_ω·t_f (σ_ω placeholder until estimated)
SIGMA_OMEGA = 0.5                      # rad/s, placeholder spin-rate uncertainty
BALLISTIC_MAX_ITER = 5
BALLISTIC_TOL = 5e-4                   # s, fixed-point convergence

MUZZLE_OFFSET = np.zeros(3)            # muzzle position in gimbal-inertial. TODO: measure barrel offset.

# --- Math helpers ---

def wrap_pi(x:float) -> float:
  return (x + math.pi) % (2 * math.pi) - math.pi

def delta_settle(d_theta:float) -> float:
  return GIMBAL_TAU + abs(d_theta) / GIMBAL_OMEGA_MAX

# --- Ballistic solver (drag-free, low-arc) ---

def _ballistic_pitch(target:np.ndarray, muzzle:np.ndarray=MUZZLE_OFFSET) -> Optional[tuple[float, float]]:
  rel = target - muzzle
  r = math.hypot(rel[0], rel[2])
  h = rel[1]
  if r < 1e-6: return None
  v0 = MUZZLE_VELOCITY
  g = GRAVITY
  disc = v0**4 - g * (g * r**2 + 2 * h * v0**2)
  if disc < 0: return None
  tan_theta = (v0**2 - math.sqrt(disc)) / (g * r)
  theta = math.atan(tan_theta)
  t_f = r / (v0 * math.cos(theta))
  return theta, t_f

def solve_with_lead(predict_fn:Callable[[float], np.ndarray], t_now:float, t_state:float,
                    d_theta_prev:float=0.0):
  # Returns (yaw_gi_cmd, pitch_cmd, t_arrival, target_at_arrival) or None.
  target = predict_fn(t_now)
  r0 = math.hypot(target[0] - MUZZLE_OFFSET[0], target[2] - MUZZLE_OFFSET[2])
  t_f = r0 / MUZZLE_VELOCITY
  dp_pipeline = (t_now - t_state) + DELTA_INPUT + DELTA_TRIGGER + delta_settle(d_theta_prev)

  for _ in range(BALLISTIC_MAX_ITER):
    t_arrival = t_now + dp_pipeline + t_f
    target = predict_fn(t_arrival)
    sol = _ballistic_pitch(target)
    if sol is None: return None
    theta, t_f_new = sol
    if abs(t_f_new - t_f) < BALLISTIC_TOL:
      t_f = t_f_new
      break
    t_f = t_f_new

  t_arrival = t_now + dp_pipeline + t_f
  rel = target - MUZZLE_OFFSET
  yaw_cmd = math.atan2(-rel[2], rel[0])
  return yaw_cmd, theta, t_arrival, target

# --- Target trajectory prediction ---

def make_predict_fn(plate:dict) -> Optional[Callable[[float], np.ndarray]]:
  cls = plate["class"]
  pos = np.array(plate["pos_gi"])
  vel = np.array(plate["vel_gi"])
  t_state = plate["t_state"]

  if cls == "STATIC":
    return lambda t, p=pos: p.copy()
  if cls == "LINEAR":
    return lambda t, p=pos, v=vel, ts=t_state: p + v * (t - ts)
  if cls == "SPIN":
    spin = plate["spin"]
    if spin is None: return None
    c_0 = np.array(spin["c_0"])
    v_c = np.array(spin["v_c"])
    t_ref = spin["t_ref"]
    def predict(t):
      dt = t - t_ref
      return np.array([c_0[0] + v_c[0]*dt, c_0[1], c_0[2] + v_c[1]*dt])
    return predict
  return None

# --- Spin trigger: best visible plate & perpendicular offset ---

@dataclass
class SpinHit:
  k: int
  e_perp: float
  plate_pos: np.ndarray

def _spin_plate_pos(spin:dict, k:int, t:float) -> Optional[np.ndarray]:
  meta = spin["plates"][k]
  if meta["known"]:
    r, h = meta["r"], meta["h"]
  else:
    known = [p for p in spin["plates"] if p["known"]]
    if not known: return None
    r = float(np.mean([p["r"] for p in known]))
    h = float(np.mean([p["h"] for p in known]))
  c_0 = np.array(spin["c_0"])
  v_c = np.array(spin["v_c"])
  dt = t - spin["t_ref"]
  c = np.array([c_0[0] + v_c[0]*dt, c_0[1], c_0[2] + v_c[1]*dt])
  theta_k = spin["omega"] * dt + spin["theta_body_0"] + k * (math.pi / 2)
  return np.array([c[0] + r*math.cos(theta_k), h, c[2] + r*math.sin(theta_k)])

def _spin_center_at(spin:dict, t:float) -> np.ndarray:
  c_0 = np.array(spin["c_0"])
  v_c = np.array(spin["v_c"])
  dt = t - spin["t_ref"]
  return np.array([c_0[0] + v_c[0]*dt, c_0[1], c_0[2] + v_c[1]*dt])

def best_visible_plate(spin:dict, t:float, muzzle:np.ndarray=MUZZLE_OFFSET) -> Optional[SpinHit]:
  center = _spin_center_at(spin, t)
  # theta_los: direction from center toward muzzle in xz plane. Plate's body-frame phase must
  # match within theta_facing for the plate to face us.
  theta_los = math.atan2(muzzle[2] - center[2], muzzle[0] - center[0])
  theta_facing = spin["theta_facing"]

  los_vec = center - muzzle
  los_norm = float(np.linalg.norm(los_vec))
  if los_norm < 1e-6: return None
  los_unit = los_vec / los_norm

  best: Optional[SpinHit] = None
  for k in range(4):
    plate_pos = _spin_plate_pos(spin, k, t)
    if plate_pos is None: continue
    dt = t - spin["t_ref"]
    theta_k = spin["omega"] * dt + spin["theta_body_0"] + k * (math.pi / 2)
    if abs(wrap_pi(theta_k - theta_los)) > theta_facing:
      continue
    p_rel = plate_pos - muzzle
    along = float(p_rel @ los_unit)
    perp = p_rel - along * los_unit
    e_perp = float(np.linalg.norm(perp))
    if best is None or e_perp < best.e_perp:
      best = SpinHit(k=k, e_perp=e_perp, plate_pos=plate_pos)
  return best

# --- Trigger gate ---

class TriggerGate:
  def __init__(self):
    self.consecutive_in_tol = 0
    self.last_fire_t = -math.inf
    self.last_target_id = -1
    self.last_burst_idx: Optional[int] = None
    self.last_spin_t_arrival: float = 0.0

  def _reset_for_new_target(self):
    self.consecutive_in_tol = 0
    self.last_burst_idx = None

  def evaluate(self, plate:dict, yaw_err:float, pitch_err:float, t_arrival:float,
               t_now:float) -> bool:
    target_id = plate["target_id"]
    if target_id != self.last_target_id:
      self._reset_for_new_target()
      self.last_target_id = target_id

    cls = plate["class"]
    if cls in ("LOST", "UNKNOWN"):
      self.consecutive_in_tol = 0
      return False

    aim_err = math.hypot(yaw_err, pitch_err)

    if cls in ("STATIC", "LINEAR"):
      if aim_err < TOL_STATIC:
        self.consecutive_in_tol += 1
      else:
        self.consecutive_in_tol = 0
      if self.consecutive_in_tol < N_TICKS_FIRE: return False
      if t_now - self.last_fire_t < COOLDOWN_STATIC: return False
      self.last_fire_t = t_now
      return True

    if cls == "SPIN":
      spin = plate["spin"]
      if spin is None: return False
      if aim_err > TOL_SPIN: return False
      hit = best_visible_plate(spin, t_arrival)
      if hit is None: return False
      omega = abs(spin["omega"])
      if omega < 1e-3: return False
      # Hit window shrinks under ω uncertainty over the flight time.
      t_f = t_arrival - t_now
      known = [p for p in spin["plates"] if p["known"]]
      r_mean = float(np.mean([p["r"] for p in known])) if known else 0.15
      e_window_eff = E_WINDOW - E_WINDOW_OMEGA_K * r_mean * SIGMA_OMEGA * max(0.0, t_f)
      if e_window_eff <= 0: return False
      if hit.e_perp > e_window_eff: return False
      # Cooldown: one shot per plate-window (T_period/4 for a 4-plate spinner).
      T_period = 2 * math.pi / omega
      if t_now - self.last_fire_t < T_period / 4.0: return False
      self.last_fire_t = t_now
      return True

    return False

# --- Chassis chase (preserve previous behavior, retarget to gimbal-inertial pos) ---

class ChassisController:
  def step(self, plate:dict) -> dict:
    if plate["class"] in ("LOST", "UNKNOWN"):
      return {"x": 0.0, "z": 0.0}
    pos = np.array(plate["pos_gi"])
    dist = float(math.hypot(pos[0], pos[2]))
    cv = {"x": 0.0, "z": 0.0}
    if dist > MAINTAIN_DIST + 0.1:
      cv["x"] = min(CHASE_SPEED, dist - MAINTAIN_DIST)
    elif dist < MAINTAIN_DIST - 0.1:
      cv["x"] = -min(CHASE_SPEED, MAINTAIN_DIST - dist)
    # Chassis yaw sign convention matches the old decisiond: chassis_velocity.z>0 turns toward
    # positive-z (left) targets. (Aim yaw uses the opposite sign because it follows the gimbal
    # R_yaw convention; chassis yaw is a separate firmware contract.)
    yaw_to_target = math.atan2(pos[2], pos[0])
    if abs(yaw_to_target) > math.radians(3.0):
      cv["z"] = math.copysign(min(CHASE_SPEED, abs(yaw_to_target) / math.radians(5)), yaw_to_target)
    return cv

# --- Daemon entry point ---

def _latest_gimbal(gimbal_sub:messaging.Sub) -> Optional[tuple[float, float]]:
  msgs = gimbal_sub.drain("gimbal_state")
  if not msgs: return None
  last = msgs[-1]
  return last["yaw_gi"], last["pitch_enc"]

def run():
  pub = messaging.Pub(["aim_error", "aim_angle", "chassis_velocity", "shoot"])
  sub = messaging.Sub(["plate"], poll="plate")
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)

  trigger = TriggerGate()
  chassis = ChassisController()

  yaw_gi_now, pitch_enc_now = 0.0, 0.0
  d_theta_prev = 0.0
  fk = FrequencyKeeper(200)
  warned_no_gimbal = False

  while True:
    fk.step()
    sub.update(timeout=10)

    gp = _latest_gimbal(gimbal_sub)
    if gp is None:
      if not warned_no_gimbal:
        logger.warning("decisiond: no gimbal_state samples; using zero gimbal pose")
        warned_no_gimbal = True
    else:
      yaw_gi_now, pitch_enc_now = gp

    plate = sub["plate"]
    if plate is None: continue
    if not sub.updated["plate"]: continue

    cls = plate["class"]
    if cls in ("LOST", "UNKNOWN"):
      pub.send("aim_error", {"x": 0.0, "y": 0.0})
      pub.send("shoot", False)
      pub.send("chassis_velocity", {"x": 0.0, "z": 0.0})
      continue

    predict = make_predict_fn(plate)
    if predict is None:
      pub.send("aim_error", {"x": 0.0, "y": 0.0})
      pub.send("shoot", False)
      continue

    t_now = time.monotonic()
    sol = solve_with_lead(predict, t_now, plate["t_state"], d_theta_prev)
    if sol is None:
      pub.send("aim_error", {"x": 0.0, "y": 0.0})
      pub.send("shoot", False)
      continue
    yaw_gi_cmd, pitch_cmd, t_arrival, target_at_arrival = sol

    yaw_err = wrap_pi(yaw_gi_cmd - yaw_gi_now)
    pitch_err = pitch_cmd - pitch_enc_now
    d_theta_prev = math.hypot(yaw_err, pitch_err)

    fire = trigger.evaluate(plate, yaw_err, pitch_err, t_arrival, t_now)

    pub.send("aim_error", {"x": yaw_err, "y": pitch_err})
    pub.send("aim_angle", {"x": math.degrees(yaw_gi_cmd), "y": math.degrees(pitch_cmd)})
    pub.send("shoot", fire)
    pub.send("chassis_velocity", chassis.step(plate))
