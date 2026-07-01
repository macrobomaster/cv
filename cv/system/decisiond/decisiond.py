import time
import math
from collections import deque
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
TOL_STATIC = math.radians(1.1)        # |aim_error| below this to commit a shot. ~small-plate half-angle at
                                       # 3.2m (135×125mm); sits above the ~0.5° measured settle jitter so it
                                       # still fires. 2.0° only guaranteed a hit inside ~1.8m (fired off-plate).
N_TICKS_FIRE = 2                       # consecutive in-tolerance ticks before firing
COOLDOWN_STATIC = 0.01                # s between shots
BARREL_HEAT_PER_SHOT = 10             # RoboMaster 17 mm projectile heat cost
BALLISTIC_MAX_ITER = 5
BALLISTIC_TOL = 5e-4                   # s, fixed-point convergence
MAX_CENTER_SPEED = 3.0                 # m/s — clamp the target velocity used for lead. Weakly observable,
                                       # spikes on handoffs, extrapolated over the ~0.6 s lead → bound to the
                                       # chassis's physical top speed so a spurious spike can't fling the aim.
LEAD_MIN_SPEED = 0.20                  # m/s — below this don't bother leading (static / near-static)
LEAD_PERP_FRAC = 0.7                   # if > this fraction of the velocity is ⊥ to the line of sight, the
                                       # target is ORBITING (spin) → suppress the lead (aim at filtered pos)
SPIN_CONF_AIM = 0.60                   # min plate["spin"]["conf"] to AIM at the bearing-space spin centre
                                       # (holds the gimbal steady on a spinner). Firing stays the settle gate.

# Muzzle position relative to the yaw axis (gimbal-inertial z-up). The VERTICAL term sets pitch:
# h = pos_gi[2] - MUZZLE_OFFSET[2], and with the yaw-only camera's T_CAM_z=0, pos_gi[2] is height-above-
# CAMERA, so MUZZLE_OFFSET[2] = -(camera optical centre height above the muzzle). MEASURE it (CAD/tape) —
# the yaw-only camera calibration can't observe a vertical lever arm. fwd/lat are parallax (2nd-order at a few m).
CAM_ABOVE_MUZZLE = 0.16                 # m, camera optical centre height ABOVE the muzzle. TODO: measure.
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
  if cls in ("LOST", "UNKNOWN"): return None

  # Confident spinner → aim at the STABLE bearing-space centre (no orbit swing in the command), leading
  # the centre bearing for a translating spinner. The gimbal HOLDS on the spin axis instead of chasing the
  # ±10° plate swing; firing stays the normal settle gate. Falls through below when spin isn't confident.
  spin = plate.get("spin")
  if spin is not None and spin.get("conf", 0.0) >= SPIN_CONF_AIM:
    cb, br = spin["center_bearing"], spin["bearing_rate"]
    ce, cr, ts = spin["center_elev"], spin["center_range"], plate["t_state"]
    def predict_center(t, cb=cb, br=br, ce=ce, cr=cr, ts=ts):
      b = cb + br * (t - ts)                            # lead the centre bearing if the robot translates
      return np.array([cr * math.cos(b), cr * math.sin(b), ce])
    return predict_center

  pos = np.array(plate["pos_gi"])
  vel = np.array(plate["vel_gi"])
  t_state = plate["t_state"]

  # Near-static (velocity below LEAD_MIN_SPEED — mostly CV-filter jitter on a still plate) → hold a STEADY
  # aim (constant position, FF falls to ~0) so the aim settles in tolerance and the trigger can commit.
  # Leading the jitter would keep the aim bouncing so it never accumulates the in-tolerance ticks to fire.
  spd = float(math.hypot(vel[0], vel[1]))
  if spd < LEAD_MIN_SPEED:
    return lambda t, p=pos: p.copy()
  if spd > MAX_CENTER_SPEED:
    vel = vel * (MAX_CENTER_SPEED / spd)
    spd = MAX_CENTER_SPEED
  # A spinning plate's velocity is ~tangential (⊥ to the line of sight): leading it flings the aim off the
  # orbit. When the velocity is mostly ⊥ to the LOS (orbiting), suppress the lead — aim at the filtered
  # position, which the CV filter averages toward the spin center.
  los = pos[:2] - MUZZLE_OFFSET[:2]
  los_n = float(math.hypot(los[0], los[1]))
  if los_n > 1e-6:
    perp = abs(vel[0] * (-los[1] / los_n) + vel[1] * (los[0] / los_n))   # |velocity ⊥ to LOS|
    if perp > LEAD_PERP_FRAC * spd:                                       # mostly tangential ⇒ orbiting
      return lambda t, p=pos: p.copy()
  return lambda t, p=pos, v=vel, ts=t_state: p + v * (t - ts)

# --- Trigger gate ---

def barrel_heat_allows_fire(barrel_heat:Optional[dict], fresh:bool) -> tuple[bool, str]:
  # Heat gating only applies when the referee actually reports a POSITIVE limit. Missing / stale / limit<=0
  # means the referee isn't enforcing heat (bench test, referee off, comms hiccup) → fire freely rather than
  # fail closed and never shoot. A real match always reports limit>0, so this only loosens the no-referee case.
  if barrel_heat is None or not fresh: return True, ""
  limit, current = barrel_heat.get("limit"), barrel_heat.get("current")
  if not limit or current is None: return True, ""         # limit 0/None ⇒ no heat enforcement
  if current + BARREL_HEAT_PER_SHOT > limit:
    return False, f"heat {current}/{limit} (+{BARREL_HEAT_PER_SHOT})"
  return True, ""

class TriggerGate:
  def __init__(self):
    self.consecutive_in_tol = 0
    self.last_fire_t = -math.inf
    self.last_target_id = -1
    self.reason = "init"                  # why the last evaluate() did/didn't fire (diagnostics)

  def _reset_for_new_target(self):
    self.consecutive_in_tol = 0

  def evaluate(self, plate:dict, yaw_err:float, pitch_err:float, t_now:float, heat_ok:bool, heat_reason:str) -> bool:
    target_id = plate["target_id"]
    if target_id != self.last_target_id:
      self._reset_for_new_target()
      self.last_target_id = target_id

    if plate["class"] != "TRACKING":         # LOST / UNKNOWN
      self.consecutive_in_tol = 0
      self.reason = f"class={plate['class']}"
      return False

    aim_err = math.hypot(yaw_err, pitch_err)
    if aim_err < TOL_STATIC:
      self.consecutive_in_tol += 1
    else:
      self.consecutive_in_tol = 0
    if self.consecutive_in_tol < N_TICKS_FIRE:
      self.reason = f"aim {math.degrees(aim_err):.2f}°≥{math.degrees(TOL_STATIC):.1f}° (n={self.consecutive_in_tol}/{N_TICKS_FIRE})"
      return False
    if not heat_ok:
      self.reason = heat_reason
      return False
    if t_now - self.last_fire_t < COOLDOWN_STATIC:
      self.reason = "cooldown"
      return False
    self.last_fire_t = t_now
    self.reason = "FIRE"
    return True

# --- Daemon entry point ---
# decisiond owns aiming POLICY (ballistic + lead → target angle, and the trigger) but NOT the gimbal
# control loop: it publishes a gimbal SETPOINT {yaw, pitch, yaw_ff, pitch_ff} that gimbald's PID closes,
# so navd can drive the same gimbal without routing through here. No aim_setpoint published when there's
# no targetable plate ⇒ gimbald yields the gimbal to navd.

def run():
  pub = messaging.Pub(["aim_setpoint", "aim_angle", "shoot"])
  sub = messaging.Sub(["plate", "barrel_heat"], poll="plate")
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
      # UNKNOWN = same target re-settling after a retarget (fresh acquire / handoff), NOT "no target".
      # Aim at the tracked plate NOW (zero lead, zero FF, no fire) so the gimbal stays on the robot instead
      # of falling through to the search scan. On a fresh acquire from scan last_setpoint is None, so we
      # must COMPUTE an aim rather than re-send nothing.
      pub.send("shoot", False)
      pos = np.array(plate["pos_gi"])
      sol = solve_with_lead(lambda t, p=pos: p.copy(), t_now, plate["t_state"], d_yaw_prev, d_pitch_prev)
      if sol is not None:
        last_setpoint = {"yaw": sol[0], "pitch": sol[1], "yaw_ff": 0.0, "pitch_ff": 0.0}
        pub.send("aim_setpoint", last_setpoint)
      elif last_setpoint is not None:
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

    heat_ok, heat_reason = barrel_heat_allows_fire(sub["barrel_heat"], sub.updated["barrel_heat"] or sub.alive["barrel_heat"])
    fire = trigger.evaluate(plate, yaw_err, pitch_err, t_now, heat_ok, heat_reason)
    if fire or t_now - last_diag > 1.0:
      logger.info(f"{'FIRE' if fire else 'no-fire'}: {cls} "
                  f"aim=({math.degrees(yaw_err):+.2f},{math.degrees(pitch_err):+.2f})° "
                  f"ff=({math.degrees(yaw_ff):+.0f},{math.degrees(pitch_ff):+.0f})°/s → {trigger.reason}")
      last_diag = t_now

    last_setpoint = {"yaw": yaw_gi_cmd, "pitch": pitch_cmd, "yaw_ff": yaw_ff, "pitch_ff": pitch_ff}
    pub.send("aim_setpoint", last_setpoint)
    pub.send("aim_angle", {"x": math.degrees(yaw_gi_cmd), "y": math.degrees(pitch_cmd)})
    pub.send("shoot", fire)
