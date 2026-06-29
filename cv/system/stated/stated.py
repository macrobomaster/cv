"""State daemon - temporary high-level gimbal state machine.

For now this only owns armor search:
  SCAN: sweep the gimbal in yaw while autoaim does not see a usable armor plate.
  YIELD: publish nothing once autoaim sees a plate; decisiond owns all aiming.

Publishes a `state_setpoint` consumed by gimbald. gimbald still arbitrates priority, so decisiond's
`aim_setpoint` can override this when enabled.
"""
import math, time

from ..core import messaging
from ..core.logging import logger
from ..common.geometry import wrap_pi

SCAN_YAW_AMPLITUDE = math.radians(45.0)
SCAN_PERIOD = 3.0
SCAN_PITCH = 0.0
AUTOAIM_TIMEOUT = 0.25

def _sees_plate(autoaim:dict|None, last_autoaim_t:float, now:float) -> bool:
  return autoaim is not None and autoaim.get("valid", False) and now - last_autoaim_t < AUTOAIM_TIMEOUT

def _scan_setpoint(t:float, center:float, start_t:float) -> dict:
  w = 2 * math.pi / SCAN_PERIOD
  phase = w * (t - start_t)
  yaw = wrap_pi(center + SCAN_YAW_AMPLITUDE * math.sin(phase))
  yaw_ff = SCAN_YAW_AMPLITUDE * w * math.cos(phase)
  return {"yaw": yaw, "pitch": SCAN_PITCH, "yaw_ff": yaw_ff, "pitch_ff": 0.0}

def run():
  pub = messaging.Pub(["state_setpoint"])
  sub = messaging.Sub(["autoaim", "gimbal_state"], poll="autoaim")

  mode = None
  scan_center = 0.0
  scan_start_t = time.monotonic()
  have_scan_center = False
  last_autoaim_t = -math.inf
  last_diag = 0.0

  while True:
    sub.update(timeout=50)
    now = time.monotonic()

    gs = sub["gimbal_state"]
    if gs is not None and not have_scan_center:
      scan_center = gs["yaw_gi"]
      have_scan_center = True

    if sub.updated["autoaim"]: last_autoaim_t = now
    autoaim = sub["autoaim"]

    next_mode = "YIELD" if _sees_plate(autoaim, last_autoaim_t, now) else "SCAN"
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
      logger.info(f"stated: mode={mode} autoaim_valid={valid}")
      last_diag = now
