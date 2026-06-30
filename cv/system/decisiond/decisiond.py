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
  AIM_FF_DT,
)

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
MAX_CENTER_SPEED = 3.0                 # m/s — clamp the target velocity used for lead (LINEAR vel_gi AND
                                       # SPIN center v_c). These are weakly observable / spike on plate
                                       # handoffs, and get extrapolated over the full ~0.6 s lead, so a
                                       # spurious spike flings the aim. Bound to the chassis's physical top speed.

# Muzzle position relative to the yaw axis (gimbal-inertial z-up). The VERTICAL term sets pitch:
# h = pos_gi[2] - MUZZLE_OFFSET[2], and with the yaw-only camera's T_CAM_z=0, pos_gi[2] is height-above-
# CAMERA, so MUZZLE_OFFSET[2] = -(camera optical centre height above the muzzle). MEASURE it (CAD/tape) —
# the yaw-only camera calibration can't observe a vertical lever arm. fwd/lat are parallax (2nd-order at a few m).
CAM_ABOVE_MUZZLE = 0.3                 # m, camera optical centre height ABOVE the muzzle. TODO: measure.
MUZZLE_OFFSET = np.array([0.0, 0.0, -CAM_ABOVE_MUZZLE])

# --- Math helpers ---

def delta_settle(yaw_err:float, pitch_err:float) -> float:
  # Gimbal isn't settled until BOTH axes are → max of the per-axis slew+settle times.
  sy = GIMBAL_TAU["yaw"] + abs(yaw_err) / GIMBAL_OMEGA_MAX["yaw"]
  sp = GIMBAL_TAU["pitch"] + abs(pitch_err) / GIMBAL_OMEGA_MAX["pitch"]
  return max(sy, sp)

# --- Ballistic solver (drag-free, low-arc) ---

def _ballistic_pitch(target:np.ndarray, muzzle:np.ndarray=MUZZLE_OFFSET) -> Optional[tuple[float, float]]:
  rel = target - muzzle
  r = math.hypot(rel[0], rel[1])   # z-up: horizontal range in the x-y plane
  h = rel[2]                        # z-up: height is the world-z component
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
  r0 = math.hypot(target[0] - MUZZLE_OFFSET[0], target[1] - MUZZLE_OFFSET[1])
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
  # z-up RH gimbal-inertial (shared with slam/plated): yaw = rotz(yaw_gi) about +z, so the target
  # azimuth is the plain RH atan2(y, x) — same handedness as gimbal_state.yaw_gi, no sign hack. This
  # retires the y-up roty(yaw) ±rel[2] hunt (that ambiguity WAS the left-handed frame). If the gimbal
  # IMU yaw still reads inverted vs rotz, fix it ONCE at the source (a YAW_FLIPPED-style flag), not here.
  yaw_cmd = math.atan2(rel[1], rel[0])
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
    spd = float(math.hypot(vel[0], vel[1]))          # bound the target velocity (z-up horizontal) so a
    if spd > MAX_CENTER_SPEED:                         # handoff/noise spike can't fling the aim over the lead
      vel = vel * (MAX_CENTER_SPEED / spd)
    return lambda t, p=pos, v=vel, ts=t_state: p + v * (t - ts)
  if cls == "SPIN":
    spin = plate["spin"]
    if spin is None: return None
    c_0 = np.array(spin["c_0"])
    v_c = np.array(spin["v_c"])
    spd = float(math.hypot(v_c[0], v_c[1]))           # bound the weakly-observable center velocity so a
    if spd > MAX_CENTER_SPEED:                         # spurious spike can't fling the aim over the lead
      v_c = v_c * (MAX_CENTER_SPEED / spd)
    t_ref = spin["t_ref"]
    def predict(t):
      dt = t - t_ref
      return np.array([c_0[0] + v_c[0]*dt, c_0[1] + v_c[1]*dt, c_0[2]])   # z-up: center moves in x-y, height=z
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
  c = np.array([c_0[0] + v_c[0]*dt, c_0[1] + v_c[1]*dt, c_0[2]])
  theta_k = spin["omega"] * dt + spin["theta_body_0"] + k * (math.pi / 2)
  return np.array([c[0] + r*math.cos(theta_k), c[1] + r*math.sin(theta_k), h])   # z-up: (x, y, height=z)

def _spin_center_at(spin:dict, t:float) -> np.ndarray:
  c_0 = np.array(spin["c_0"])
  v_c = np.array(spin["v_c"])
  dt = t - spin["t_ref"]
  return np.array([c_0[0] + v_c[0]*dt, c_0[1] + v_c[1]*dt, c_0[2]])   # z-up: center moves in x-y, height=z

def best_visible_plate(spin:dict, t:float, muzzle:np.ndarray=MUZZLE_OFFSET) -> Optional[SpinHit]:
  center = _spin_center_at(spin, t)
  # theta_los: direction from center toward muzzle in the x-y horizontal plane (z-up). Plate's
  # body-frame phase must match within theta_facing for the plate to face us.
  theta_los = math.atan2(muzzle[1] - center[1], muzzle[0] - center[0])
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
    self.reason = "init"                  # why the last evaluate() did/didn't fire (diagnostics)

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
      self.reason = f"class={cls}"
      return False

    aim_err = math.hypot(yaw_err, pitch_err)

    if cls in ("STATIC", "LINEAR"):
      if aim_err < TOL_STATIC:
        self.consecutive_in_tol += 1
      else:
        self.consecutive_in_tol = 0
      if self.consecutive_in_tol < N_TICKS_FIRE:
        self.reason = f"aim {math.degrees(aim_err):.2f}°≥{math.degrees(TOL_STATIC):.1f}° (n={self.consecutive_in_tol}/{N_TICKS_FIRE})"
        return False
      if t_now - self.last_fire_t < COOLDOWN_STATIC:
        self.reason = "cooldown"
        return False
      self.last_fire_t = t_now
      self.reason = "FIRE"
      return True

    if cls == "SPIN":
      spin = plate["spin"]
      if spin is None:
        self.reason = "spin=None"
        return False
      if aim_err > TOL_SPIN:
        self.reason = f"aim {math.degrees(aim_err):.2f}°>{math.degrees(TOL_SPIN):.1f}°"
        return False
      hit = best_visible_plate(spin, t_arrival)
      if hit is None:
        self.reason = "no plate facing"
        return False
      omega = abs(spin["omega"])
      if omega < 1e-3:
        self.reason = "ω≈0"
        return False
      # Hit window shrinks under ω uncertainty over the flight time.
      t_f = t_arrival - t_now
      known = [p for p in spin["plates"] if p["known"]]
      r_mean = float(np.mean([p["r"] for p in known])) if known else 0.15
      e_window_eff = E_WINDOW - E_WINDOW_OMEGA_K * r_mean * SIGMA_OMEGA * max(0.0, t_f)
      if e_window_eff <= 0:
        self.reason = "window collapsed (range/σω)"
        return False
      if hit.e_perp > e_window_eff:
        self.reason = f"e_perp {hit.e_perp*1000:.0f}>{e_window_eff*1000:.0f}mm (k={hit.k})"
        return False
      # Cooldown: one shot per plate-window (T_period/4 for a 4-plate spinner).
      T_period = 2 * math.pi / omega
      if t_now - self.last_fire_t < T_period / 4.0:
        self.reason = "spin cooldown"
        return False
      self.last_fire_t = t_now
      self.reason = "FIRE"
      return True

    self.reason = f"class={cls}?"
    return False

# --- Daemon entry point ---
# decisiond owns aiming POLICY (ballistic + lead → target angle, and the trigger) but NOT the gimbal
# control loop: it publishes a gimbal SETPOINT {yaw, pitch, yaw_ff, pitch_ff} that gimbald's PID closes,
# so navd can drive the same gimbal without routing through here. No aim_setpoint published when there's
# no targetable plate ⇒ gimbald yields the gimbal to navd.

def run():
  pub = messaging.Pub(["aim_setpoint", "aim_angle", "shoot"])
  sub = messaging.Sub(["plate"], poll="plate")
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()

  trigger = TriggerGate()

  yaw_gi_now, pitch_gi_now = 0.0, 0.0
  d_yaw_prev, d_pitch_prev = 0.0, 0.0
  fk = FrequencyKeeper(200)
  warned_no_gimbal = False
  last_diag = 0.0                         # throttle for the trigger-reason diagnostic
  last_setpoint = None                    # last aim_setpoint, re-sent to HOLD through a brief re-acquire

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
      yaw_gi_now, pitch_gi_now, _, _ = gp

    t_now = time.monotonic()
    plate = sub["plate"]
    if plate is None: continue
    if not sub.updated["plate"]: continue

    cls = plate["class"]
    if cls == "LOST":
      pub.send("shoot", False)
      last_setpoint = None                   # truly lost → release the gimbal (scan/nav can take over)
      if t_now - last_diag > 1.0:
        logger.info("no-fire: class=LOST (target lost)"); last_diag = t_now
      continue
    if cls == "UNKNOWN":
      # UNKNOWN = same target re-settling after a retarget (e.g. a spin plate-handoff), NOT "no target".
      # HOLD the last aim (zero feedforward) so the gimbal stays on the robot instead of going stale and
      # letting gimbald fall through to the search scan (→ gimbal jumps to a random direction). No fire
      # until the class settles.
      pub.send("shoot", False)
      if last_setpoint is not None:
        pub.send("aim_setpoint", {**last_setpoint, "yaw_ff": 0.0, "pitch_ff": 0.0})
      if t_now - last_diag > 1.0:
        logger.info("no-fire: class=UNKNOWN (re-acquiring, holding aim)"); last_diag = t_now
      continue

    predict = make_predict_fn(plate)
    if predict is None:
      pub.send("shoot", False)
      continue

    sol = solve_with_lead(predict, t_now, plate["t_state"], d_yaw_prev, d_pitch_prev)
    if sol is None:
      pub.send("shoot", False)
      if t_now - last_diag > 1.0:
        logger.info(f"no-fire: {cls} no ballistic solution (out of range?)"); last_diag = t_now
      continue
    yaw_gi_cmd, pitch_cmd, t_arrival, target_at_arrival = sol

    # Aim-point angular velocity (feedforward): how fast the lead-compensated command moves in real
    # time → lets gimbald's loop track moving/spinning targets. Advance BOTH t_now and t_state by FF_DT:
    # in real time a fresh plate arrives with t_state advanced too, so the (t_now - t_state) lead term
    # stays constant (advancing only t_now would double-count it and over-feedforward by ~2×).
    sol_ff = solve_with_lead(predict, t_now + AIM_FF_DT, plate["t_state"] + AIM_FF_DT, d_yaw_prev, d_pitch_prev)
    if sol_ff is not None:
      yaw_ff = wrap_pi(sol_ff[0] - yaw_gi_cmd) / AIM_FF_DT
      pitch_ff = (sol_ff[1] - pitch_cmd) / AIM_FF_DT
    else:
      yaw_ff = pitch_ff = 0.0
    # Bound the feedforward to the gimbal's slew limit: the /AIM_FF_DT finite difference amplifies any
    # blip (velocity noise, ballistic-iteration jitter) 100×, and feedforwarding faster than the gimbal
    # can physically turn only makes it overshoot. A real target's tracked rate never exceeds OMEGA_MAX.
    yaw_ff = max(-GIMBAL_OMEGA_MAX["yaw"], min(GIMBAL_OMEGA_MAX["yaw"], yaw_ff))
    pitch_ff = max(-GIMBAL_OMEGA_MAX["pitch"], min(GIMBAL_OMEGA_MAX["pitch"], pitch_ff))

    # Aim error vs the live gimbal pose (gimbald closes the loop; here it only gates the trigger).
    yaw_err = wrap_pi(yaw_gi_cmd - yaw_gi_now)
    pitch_err = pitch_cmd - pitch_gi_now
    d_yaw_prev, d_pitch_prev = abs(yaw_err), abs(pitch_err)

    fire = trigger.evaluate(plate, yaw_err, pitch_err, t_arrival, t_now)
    if fire or t_now - last_diag > 1.0:
      extra = ""
      if cls == "SPIN" and plate["spin"]:
        vc = plate["spin"]["v_c"]
        extra = f"  |v_c|={math.hypot(vc[0], vc[1]):.1f}m/s ω={math.degrees(plate['spin']['omega']):+.0f}°/s"
      logger.info(f"{'FIRE' if fire else 'no-fire'}: {cls} "
                  f"aim=({math.degrees(yaw_err):+.2f},{math.degrees(pitch_err):+.2f})° "
                  f"ff=({math.degrees(yaw_ff):+.0f},{math.degrees(pitch_ff):+.0f})°/s → {trigger.reason}{extra}")
      last_diag = t_now

    last_setpoint = {"yaw": yaw_gi_cmd, "pitch": pitch_cmd, "yaw_ff": yaw_ff, "pitch_ff": pitch_ff}
    pub.send("aim_setpoint", last_setpoint)
    pub.send("aim_angle", {"x": math.degrees(yaw_gi_cmd), "y": math.degrees(pitch_cmd)})
    pub.send("shoot", fire)
