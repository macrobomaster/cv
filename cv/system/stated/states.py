"""Concrete match state machines for stated.

Factory input:
  team_color: our alliance color ("red"/"blue")
  play_style: high-level strategy preset ("balanced"/"center")
"""
import math

from ..core.logging import logger
from .base_states import PeriodicNavToGoalState, NavToGoalState, PeriodicSequenceState, TimedHoldGoalState, AcquireHoldState, SearchScanState
from .state_machine import StateBase, StateMachine

SCAN_SWEEP_AMPLITUDE = math.radians(45.0)
SCAN_SWEEP_DT = 1.0
SCAN_TURN_DT = 0.5
SCAN_PITCH = 0.0
ACQUIRE_HOLD_DT = 0.8
NAV_ARRIVE_RADIUS = 0.25
RECENTER_PERIOD = 10.0
RETREAT_PERIOD = 60.0
RETREAT_HOLD_DT = 10.0

TEAM_GOALS = {
  "red": {
    "center": (11.02, 6.98),
    "back": (11.00, 6.00),
  },
  "blue": {
    "center": (0.98, 6.98),
    "back": (1.00, 6.00),
  },
}
PLAY_STYLES = {"balanced", "center"}

class IdleState(StateBase):
  name = "idle"

  def should_transition(self, current:StateBase, ctx) -> bool:
    return not bool(ctx["game_running"])

  def can_transition(self, ctx) -> bool:
    return True

  def run(self, ctx, pub):
    pass

class NavToCenterState(PeriodicNavToGoalState):
  name = "nav_to_center"
  goal_label = "center"
  arrive_radius = NAV_ARRIVE_RADIUS
  period = RECENTER_PERIOD
  first_delay = 0.0

  def __init__(self, goal_xy:tuple[float, float]):
    self.goal_xy = goal_xy
    super().__init__()

class NavToBackState(NavToGoalState):
  name = "nav_to_back"
  goal_label = "back"
  arrive_radius = NAV_ARRIVE_RADIUS

  def __init__(self, goal_xy:tuple[float, float]):
    self.goal_xy = goal_xy

class HoldBackState(TimedHoldGoalState):
  name = "hold_back"
  goal_label = "back_hold"
  hold_dt = RETREAT_HOLD_DT

  def __init__(self, goal_xy:tuple[float, float]):
    self.goal_xy = goal_xy
    super().__init__()

class ReturnCenterState(NavToGoalState):
  name = "return_center"
  goal_label = "center"
  arrive_radius = NAV_ARRIVE_RADIUS

  def __init__(self, goal_xy:tuple[float, float]):
    self.goal_xy = goal_xy

class RetreatSequenceState(PeriodicSequenceState):
  name = "retreat_sequence"
  period = RETREAT_PERIOD
  first_delay = RETREAT_PERIOD

  def __init__(self, center_goal:tuple[float, float], back_goal:tuple[float, float]):
    super().__init__([NavToBackState(back_goal), HoldBackState(back_goal), ReturnCenterState(center_goal)])

class AcquireState(AcquireHoldState):
  name = "acquire"
  hold_dt = ACQUIRE_HOLD_DT

class SearchState(SearchScanState):
  name = "search"
  sweep_amplitude = SCAN_SWEEP_AMPLITUDE
  sweep_dt = SCAN_SWEEP_DT
  turn_dt = SCAN_TURN_DT
  pitch = SCAN_PITCH

def _goals(team_color:str) -> dict:
  if team_color not in TEAM_GOALS:
    raise ValueError(f"unknown team_color {team_color!r}; expected one of {sorted(TEAM_GOALS)}")
  return TEAM_GOALS[team_color]

def make_state_machine(team_color:str, play_style:str="balanced") -> StateMachine:
  goals = _goals(team_color)
  play_style = play_style.lower()
  nav_to_center = NavToCenterState(goals["center"])
  acquire = AcquireState()
  search = SearchState()

  if play_style == "balanced":
    states = [IdleState(), RetreatSequenceState(goals["center"], goals["back"]), nav_to_center, acquire, search]
  elif play_style == "center":
    states = [IdleState(), nav_to_center, acquire, search]
  else:
    logger.warning(f"stated: unknown PLAY_STYLE={play_style!r}; using balanced")
    states = [IdleState(), RetreatSequenceState(goals["center"], goals["back"]), nav_to_center, acquire, search]
  return StateMachine(states)
