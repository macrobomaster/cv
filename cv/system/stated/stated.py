"""State daemon - temporary high-level gimbal state machine.

For now this only owns armor search:
  SCAN: sweep +/-45 deg, turn 180 deg, then sweep the opposite half while autoaim sees no usable plate.
  YIELD: publish nothing for a short acquisition window after autoaim sees a plate; decisiond owns all aiming.

Publishes a `state_setpoint` consumed by gimbald. gimbald still arbitrates priority, so decisiond's
`aim_setpoint` can override this when enabled.
"""
import math, time

from ..core import messaging
from ..core.logging import logger
from ..common.geometry import wrap_pi

SCAN_SWEEP_AMPLITUDE = math.radians(45.0)
SCAN_SWEEP_DT = 1.0
SCAN_TURN_DT = 0.5
SCAN_PITCH = 0.0
ACQUIRE_HOLD_DT = 0.8

def _recently_saw_plate(last_valid_t:float, now:float) -> bool:
  return now - last_valid_t < ACQUIRE_HOLD_DT

def _scan_setpoint(t:float, center:float, start_t:float) -> dict:
  cycle_dt = SCAN_SWEEP_DT + SCAN_TURN_DT
  t_scan = t - start_t
  cycle = int(t_scan // cycle_dt)
  t_cycle = t_scan - cycle * cycle_dt
  base = center + (cycle % 2) * math.pi
  if t_cycle < SCAN_SWEEP_DT:
    w = 2 * math.pi / SCAN_SWEEP_DT
    yaw = wrap_pi(base + SCAN_SWEEP_AMPLITUDE * math.sin(w * t_cycle))
    yaw_ff = SCAN_SWEEP_AMPLITUDE * w * math.cos(w * t_cycle)
  else:
    yaw_ff = math.pi / SCAN_TURN_DT
    yaw = wrap_pi(base + yaw_ff * (t_cycle - SCAN_SWEEP_DT))
  return {"yaw": yaw, "pitch": SCAN_PITCH, "yaw_ff": yaw_ff, "pitch_ff": 0.0}

def run():
  pub = messaging.Pub(["state_setpoint"])
  sub = messaging.Sub(["autoaim", "gimbal_state"], poll="autoaim")

  mode = None
  scan_center = 0.0
  scan_start_t = time.monotonic()
  have_scan_center = False
  last_valid_t = -math.inf
  last_diag = 0.0

  while True:
    sub.update(timeout=50)
    now = time.monotonic()

    gs = sub["gimbal_state"]
    if gs is not None and not have_scan_center:
      scan_center = gs["yaw_gi"]
      have_scan_center = True

    autoaim = sub["autoaim"]
    if sub.updated["autoaim"] and autoaim is not None and autoaim.get("valid", False): last_valid_t = now

    next_mode = "YIELD" if _recently_saw_plate(last_valid_t, now) else "SCAN"
    if next_mode != mode:
      if next_mode == "SCAN":
        if gs is not None: scan_center = gs["yaw_gi"]
        scan_start_t = now
      mode = next_mode
      logger.info(f"stated: {mode.lower()}")

    if mode == "SCAN":
      pub.send("state_setpoint", _scan_setpoint(now, scan_center, scan_start_t))

    if now - last_diag > 1.0:
      valid = autoaim.get("valid") if autoaim is not None else None
      logger.info(f"stated: mode={mode} autoaim_valid={valid} last_valid={now - last_valid_t:.2f}s")
      last_diag = now
