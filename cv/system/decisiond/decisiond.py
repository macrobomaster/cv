import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from ..core import messaging
from ..core.logging import logger
from ..core.helpers import FrequencyKeeper
from ..common.geometry import wrap_pi
from ..common.gimbal import GimbalBuffer
from ...autoaim.common import (
  MUZZLE_VELOCITY, GRAVITY,
  DELTA_INPUT, DELTA_TRIGGER, GIMBAL_TAU, GIMBAL_OMEGA_MAX,
  AIM_GAINS, AIM_I_CLAMP, AIM_FF_DT, AIM_D_TAU,
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

def delta_settle(yaw_err:float, pitch_err:float) -> float:
  # Gimbal isn't settled until BOTH axes are → max of the per-axis slew+settle times.
  sy = GIMBAL_TAU["yaw"] + abs(yaw_err) / GIMBAL_OMEGA_MAX["yaw"]
  sp = GIMBAL_TAU["pitch"] + abs(pitch_err) / GIMBAL_OMEGA_MAX["pitch"]
  return max(sy, sp)

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
                    d_yaw_prev:float=0.0, d_pitch_prev:float=0.0):
  # Returns (yaw_gi_cmd, pitch_cmd, t_arrival, target_at_arrival) or None.
  target = predict_fn(t_now)
  r0 = math.hypot(target[0] - MUZZLE_OFFSET[0], target[2] - MUZZLE_OFFSET[2])
  t_f = r0 / MUZZLE_VELOCITY
  dp_pipeline = (t_now - t_state) + DELTA_INPUT + DELTA_TRIGGER + delta_settle(d_yaw_prev, d_pitch_prev)

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
  yaw_cmd = math.atan2(rel[2], rel[0])  # +z = right in the corrected (proper) gimbal frame; was -rel[2] to undo R_MOUNT's mirror
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

# --- Gimbal tracking controller (rate/joystick output) ---

class AxisPID:
  """Velocity feedforward + PID on the angular position error → aim_error (a rate command, divided
  by K_JOYSTICK). With KI=KD=0 this is feedforward + P. omega_ff carries the aim-point motion so the
  loop tracks moving targets without the steady-state lag a pure position controller leaves.

  The derivative is computed as the ERROR rate from clean signals — d(θ_err)/dt = θ̇_target − θ̇_gimbal
  = omega_ff − gimbal_rate — not by differencing the noisy position error, then low-pass filtered.
  That's kick-free (a retarget step in θ_target never spikes it) and ~0 during good tracking."""
  def __init__(self, kp, ki, kd, k_joystick, i_clamp, d_tau):
    self.kp, self.ki, self.kd, self.kj, self.i_clamp, self.d_tau = kp, ki, kd, k_joystick, i_clamp, d_tau
    self.reset()

  def reset(self):
    self.integ = 0.0
    self.d_filt = 0.0

  def update(self, err:float, omega_ff:float, gimbal_rate:float, dt:float) -> float:
    if dt > 0:
      self.integ += err * dt
      if self.ki > 0:  # clamp the integral's velocity contribution (anti-windup)
        lim = self.i_clamp / self.ki
        self.integ = max(-lim, min(lim, self.integ))
    d_err = omega_ff - gimbal_rate                          # error derivative from clean signals
    if dt > 0 and self.d_tau > 0:
      self.d_filt += (dt / (self.d_tau + dt)) * (d_err - self.d_filt)
    else:
      self.d_filt = d_err
    omega_des = omega_ff + self.kp * err + self.ki * self.integ + self.kd * self.d_filt
    return omega_des / self.kj

# --- Daemon entry point ---

def run():
  pub = messaging.Pub(["aim_error", "aim_angle", "chassis_velocity", "shoot"])
  sub = messaging.Sub(["plate"], poll="plate")
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()

  trigger = TriggerGate()
  # chassis = ChassisController()
  yaw_pid = AxisPID(**AIM_GAINS["yaw"], i_clamp=AIM_I_CLAMP, d_tau=AIM_D_TAU)
  pitch_pid = AxisPID(**AIM_GAINS["pitch"], i_clamp=AIM_I_CLAMP, d_tau=AIM_D_TAU)

  yaw_gi_now, pitch_gi_now = 0.0, 0.0
  yaw_rate_now, pitch_rate_now = 0.0, 0.0
  d_yaw_prev, d_pitch_prev = 0.0, 0.0
  last_t = None
  last_target_id = -1
  fk = FrequencyKeeper(200)
  warned_no_gimbal = False

  while True:
    fk.step()
    sub.update(timeout=10)

    for m in gimbal_sub.drain("gimbal_state"): gimbal_buf.push(m)
    gp = gimbal_buf.latest()
    if gp is None:
      if not warned_no_gimbal:
        logger.warning("decisiond: no gimbal_state samples; using zero gimbal pose")
        warned_no_gimbal = True
    else:
      yaw_gi_now, pitch_gi_now, yaw_rate_now, pitch_rate_now = gp

    plate = sub["plate"]
    if plate is None: continue
    if not sub.updated["plate"]: continue

    # Reset the controllers on retarget so integral/derivative state doesn't carry across targets.
    if plate["target_id"] != last_target_id:
      yaw_pid.reset(); pitch_pid.reset()
      last_t = None
      last_target_id = plate["target_id"]

    cls = plate["class"]
    if cls in ("LOST", "UNKNOWN"):
      yaw_pid.reset(); pitch_pid.reset(); last_t = None
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
    sol = solve_with_lead(predict, t_now, plate["t_state"], d_yaw_prev, d_pitch_prev)
    if sol is None:
      pub.send("aim_error", {"x": 0.0, "y": 0.0})
      pub.send("shoot", False)
      continue
    yaw_gi_cmd, pitch_cmd, t_arrival, target_at_arrival = sol

    # Aim-point angular velocity (feedforward): how fast the lead-compensated command moves in real
    # time → tracks moving/spinning targets. Advance BOTH t_now and t_state by FF_DT: in real time a
    # fresh plate arrives with t_state advanced too, so the (t_now - t_state) lead term stays constant
    # (advancing only t_now would double-count it and over-feedforward by ~2×).
    sol_ff = solve_with_lead(predict, t_now + AIM_FF_DT, plate["t_state"] + AIM_FF_DT, d_yaw_prev, d_pitch_prev)
    if sol_ff is not None:
      yaw_ff = wrap_pi(sol_ff[0] - yaw_gi_cmd) / AIM_FF_DT
      pitch_ff = (sol_ff[1] - pitch_cmd) / AIM_FF_DT
    else:
      yaw_ff = pitch_ff = 0.0

    yaw_err = wrap_pi(yaw_gi_cmd - yaw_gi_now)
    pitch_err = pitch_cmd - pitch_gi_now
    d_yaw_prev, d_pitch_prev = abs(yaw_err), abs(pitch_err)

    dt = 0.0 if last_t is None else (t_now - last_t)
    last_t = t_now
    aim_x = yaw_pid.update(yaw_err, yaw_ff, yaw_rate_now, dt)
    aim_y = pitch_pid.update(pitch_err, pitch_ff, pitch_rate_now, dt)

    fire = trigger.evaluate(plate, yaw_err, pitch_err, t_arrival, t_now)

    pub.send("aim_error", {"x": aim_x, "y": aim_y})
    pub.send("aim_angle", {"x": math.degrees(yaw_gi_cmd), "y": math.degrees(pitch_cmd)})
    pub.send("shoot", fire)
    # pub.send("chassis_velocity", chassis.step(plate))
