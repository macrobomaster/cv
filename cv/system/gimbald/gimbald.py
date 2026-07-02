"""Gimbal control daemon — owns the single gimbal tracking loop (velocity feedforward + PID → aim_error).

Multiple controllers want to point the gimbal: decisiond (aim at a target), qrd (QR acknowledgement),
and stated (search scan).
Rather than have them fight over aim_error or read each other's state, each publishes a
gimbal SETPOINT on its own topic, and gimbald arbitrates and runs the one control loop that closes on
gimbal_state and emits aim_error to commsd.

Setpoint contract (`aim_setpoint`, `state_setpoint`): {yaw, pitch, yaw_ff, pitch_ff}
  yaw, pitch        absolute gimbal target — gimbal-inertial yaw, gravity-relative pitch (rad)
  yaw_ff, pitch_ff  feedforward angular rate of the aim point (rad/s); 0 if unknown
`qr_ack` is a trigger event; gimbald turns it into a short local setpoint sequence.

Arbitration (priority, first fresh wins): aim_setpoint (decisiond, only when it has a target);
otherwise a short qr_ack nod; otherwise state_setpoint (stated's search scan); otherwise hold (zero rate).
The PID is reset on every source switch so its integral/derivative don't carry across a setpoint discontinuity.

Subs:  aim_setpoint, qr_ack, state_setpoint, gimbal_state
Pubs:  aim_error: {x, y}   (rate/joystick command to commsd)
"""
import time
import math

from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.helpers import FrequencyKeeper
from ..common.geometry import wrap_pi
from ..common.gimbal import GimbalBuffer
from ...autoaim.common import AIM_GAINS, AIM_I_CLAMP, AIM_D_TAU

# External setpoint topics; qr_ack is handled as a generated local setpoint between these priorities.
SOURCES = ["aim_setpoint", "state_setpoint"]
SETPOINT_TIMEOUT = 0.1   # s — a setpoint older than this is stale; its source yields to the next
QR_ACK_DT = getenv("QR_ACK_DT", 0.55)
QR_ACK_PITCH = math.radians(getenv("QR_ACK_PITCH_DEG", 7.0))

def qr_ack_setpoint(now:float, start:float, yaw:float, pitch:float) -> dict|None:
  phase = (now - start) / QR_ACK_DT
  if phase < 0.0 or phase >= 1.0: return None
  pitch_offset = -QR_ACK_PITCH * math.sin(math.pi * phase) ** 2
  pitch_ff = -QR_ACK_PITCH * (math.pi / QR_ACK_DT) * math.sin(2.0 * math.pi * phase)
  return {"yaw": yaw, "pitch": pitch + pitch_offset, "yaw_ff": 0.0, "pitch_ff": pitch_ff}

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
  sub = messaging.Sub(SOURCES + ["qr_ack"])          # conflated: latest setpoint/event per source
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()

  yaw_pid = AxisPID(**AIM_GAINS["yaw"], i_clamp=AIM_I_CLAMP, d_tau=AIM_D_TAU)
  pitch_pid = AxisPID(**AIM_GAINS["pitch"], i_clamp=AIM_I_CLAMP, d_tau=AIM_D_TAU)

  yaw_now = pitch_now = yaw_rate = pitch_rate = 0.0
  last_fresh = {s: -math.inf for s in SOURCES}
  qr_ack_start = -math.inf
  qr_ack_yaw = qr_ack_pitch = 0.0
  active = None
  last_t = None
  fk = FrequencyKeeper(200)
  warned_no_gimbal = False
  last_diag = 0.0
  last_status = 0.0

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
    if sub.updated["qr_ack"] and gp is not None:
      qr_ack_start = now
      qr_ack_yaw, qr_ack_pitch = yaw_now, pitch_now
      msg = sub["qr_ack"]
      logger.info(f"gimbald: QR ack nod style={msg.get('style') if isinstance(msg, dict) else msg}")

    # First fresh/high-priority source wins the gimbal.
    qr_sp = qr_ack_setpoint(now, qr_ack_start, qr_ack_yaw, qr_ack_pitch)
    if now - last_fresh["aim_setpoint"] < SETPOINT_TIMEOUT:
      src, sp = "aim_setpoint", sub["aim_setpoint"]
    elif qr_sp is not None:
      src, sp = "qr_ack", qr_sp
    elif now - last_fresh["state_setpoint"] < SETPOINT_TIMEOUT:
      src, sp = "state_setpoint", sub["state_setpoint"]
    else:
      src, sp = None, None
    if src != active:
      yaw_pid.reset(); pitch_pid.reset(); last_t = None; active = src
      if now - last_diag > 1.0:
        logger.info(f"gimbald: source → {src or 'hold'}"); last_diag = now

    if sp is None or gp is None:
      pub.send("aim_error", {"x": 0.0, "y": 0.0})  # no setpoint / no feedback ⇒ hold (zero rate)
      if now - last_status > 1.0:
        ages = {s: (now - last_fresh[s] if last_fresh[s] > -math.inf else math.inf) for s in SOURCES}
        logger.info(f"gimbald: active={active or 'hold'} feedback={gp is not None} ages={ages}")
        last_status = now
      continue

    dt = 0.0 if last_t is None else (now - last_t)
    last_t = now
    yaw_err = wrap_pi(sp["yaw"] - yaw_now)
    pitch_err = sp["pitch"] - pitch_now
    aim_x = yaw_pid.update(yaw_err, sp["yaw_ff"], yaw_rate, dt)
    aim_y = pitch_pid.update(pitch_err, sp["pitch_ff"], pitch_rate, dt)
    pub.send("aim_error", {"x": aim_x, "y": aim_y})
    if now - last_status > 1.0:
      logger.info(f"gimbald: active={active} yaw_err={math.degrees(yaw_err):+.1f}deg "
                  f"yaw_ff={math.degrees(sp['yaw_ff']):+.0f}deg/s aim=({aim_x:+.3f},{aim_y:+.3f})")
      last_status = now
