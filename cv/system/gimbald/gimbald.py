"""Gimbal control daemon — owns the single gimbal tracking loop (velocity feedforward + PID → aim_error).

Multiple controllers want to point the gimbal: decisiond (aim at a target) and navd (scan / point for
navigation). Rather than have them fight over aim_error or read each other's state, each publishes a
gimbal SETPOINT on its own topic, and gimbald arbitrates and runs the one control loop that closes on
gimbal_state and emits aim_error to commsd.

Setpoint contract (both topics): {yaw, pitch, yaw_ff, pitch_ff}
  yaw, pitch        absolute gimbal target — gimbal-inertial yaw, gravity-relative pitch (rad)
  yaw_ff, pitch_ff  feedforward angular rate of the aim point (rad/s); 0 if unknown

Arbitration: aim_setpoint wins while fresh (decisiond only publishes it when it has a target);
otherwise nav_setpoint; otherwise hold (zero rate). The PID is reset on every source switch so its
integral/derivative don't carry across a setpoint discontinuity.

Subs:  aim_setpoint, nav_setpoint, gimbal_state
Pubs:  aim_error: {x, y}   (rate/joystick command to commsd)
"""
import time
import math

from ..core import messaging
from ..core.logging import logger
from ..core.helpers import FrequencyKeeper
from ..common.geometry import wrap_pi
from ..common.gimbal import GimbalBuffer
from ...autoaim.common import AIM_GAINS, AIM_I_CLAMP, AIM_D_TAU

# Setpoint sources in PRIORITY order — first one with a fresh sample wins the gimbal.
SOURCES = ["aim_setpoint", "nav_setpoint"]
SETPOINT_TIMEOUT = 0.1   # s — a setpoint older than this is stale; its source yields to the next

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

def run():
  pub = messaging.Pub(["aim_error"])
  sub = messaging.Sub(SOURCES)                       # conflated: latest setpoint per source
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()

  yaw_pid = AxisPID(**AIM_GAINS["yaw"], i_clamp=AIM_I_CLAMP, d_tau=AIM_D_TAU)
  pitch_pid = AxisPID(**AIM_GAINS["pitch"], i_clamp=AIM_I_CLAMP, d_tau=AIM_D_TAU)

  yaw_now = pitch_now = yaw_rate = pitch_rate = 0.0
  last_fresh = {s: -math.inf for s in SOURCES}
  active = None
  last_t = None
  fk = FrequencyKeeper(200)
  warned_no_gimbal = False
  last_diag = 0.0

  while True:
    fk.step()
    sub.update(timeout=10)
    now = time.monotonic()

    for m in gimbal_sub.drain("gimbal_state"): gimbal_buf.push(m)
    gp = gimbal_buf.latest()
    if gp is None:
      if not warned_no_gimbal:
        logger.warning("gimbald: no gimbal_state samples; holding")
        warned_no_gimbal = True
    else:
      yaw_now, pitch_now, yaw_rate, pitch_rate = gp

    for s in SOURCES:
      if sub.updated[s]: last_fresh[s] = now

    # First fresh source in priority order wins the gimbal.
    src = next((s for s in SOURCES if now - last_fresh[s] < SETPOINT_TIMEOUT), None)
    sp = sub[src] if src is not None else None
    if src != active:
      yaw_pid.reset(); pitch_pid.reset(); last_t = None; active = src
      if now - last_diag > 1.0:
        logger.info(f"gimbald: source → {src or 'hold'}"); last_diag = now

    if sp is None or gp is None:
      pub.send("aim_error", {"x": 0.0, "y": 0.0})  # no setpoint / no feedback ⇒ hold (zero rate)
      continue

    dt = 0.0 if last_t is None else (now - last_t)
    last_t = now
    yaw_err = wrap_pi(sp["yaw"] - yaw_now)
    pitch_err = sp["pitch"] - pitch_now
    aim_x = yaw_pid.update(yaw_err, sp["yaw_ff"], yaw_rate, dt)
    aim_y = pitch_pid.update(pitch_err, sp["pitch_ff"], pitch_rate, dt)
    pub.send("aim_error", {"x": aim_x, "y": aim_y})
