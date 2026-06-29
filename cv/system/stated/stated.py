from ..core import messaging
from ..core.logging import logger
from .states import make_state_machine

def run():
  pub = messaging.Pub(["state_setpoint", "nav_goal"])
  sub = messaging.Sub(["game_running", "autoaim", "gimbal_state", "slam_pose"], poll="game_running")
  sm = make_state_machine()
  last_diag = 0.0

  while True:
    sub.update(timeout=50)
    autoaim = sub["autoaim"]

    sm.tick(sub, pub)
    if sm.entered: logger.info(f"stated: {sm.current.name}")

    if sub.now - last_diag > 1.0:
      valid = autoaim.get("valid") if autoaim is not None else None
      logger.info(f"stated: state={sm.current.name} game={bool(sub['game_running'])} autoaim_valid={valid}")
      last_diag = sub.now
