import os

from ..core import messaging
from ..core.logging import logger
from .states import make_state_machine, PLAY_STYLES

TEAM_COLORS = {"red", "blue"}

def _style_from_msg(msg):
  style = msg.get("style") if isinstance(msg, dict) else msg
  return style if style in PLAY_STYLES else None

def run():
  pub = messaging.Pub(["state_setpoint", "nav_goal", "spinning"])
  sub = messaging.Sub(["game_running", "team_color", "play_style", "autoaim", "gimbal_state", "slam_pose", "apriltags"], poll="game_running")
  default_style = os.environ.get("PLAY_STYLE", "balanced")
  if default_style not in PLAY_STYLES:
    logger.warning(f"stated: unknown PLAY_STYLE={default_style!r}; using balanced")
    default_style = "balanced"
  sm = None
  active_team = None
  active_style = None
  last_diag = 0.0

  while True:
    sub.update(timeout=50)
    autoaim = sub["autoaim"]
    team_color = sub["team_color"]
    play_style = _style_from_msg(sub["play_style"]) or default_style

    if team_color in TEAM_COLORS and (sm is None or team_color != active_team or play_style != active_style):
      sm = make_state_machine(team_color, play_style)
      active_team, active_style = team_color, play_style
      logger.info(f"stated: configured team={active_team} style={active_style}")

    if sm is not None:
      sm.tick(sub, pub)
      if sm.entered: logger.info(f"stated: {sm.current.name}")

    if sub.now - last_diag > 1.0:
      valid = autoaim.get("valid") if autoaim is not None else None
      state = sm.current.name if sm is not None else "unconfigured"
      logger.info(f"stated: state={state} game={bool(sub['game_running'])} team={team_color} "
                  f"style={play_style} autoaim_valid={valid}")
      last_diag = sub.now
