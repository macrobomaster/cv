"""Concrete match state machines for stated.

Factory input:
  team_color: our alliance color ("red"/"blue")
  play_style: high-level strategy preset ("balanced"/"center")
"""
import math

from .base_states import LookAtVisibleTagMixin, ScanAcquireMixin, NavToGoalState, PeriodicNavToGoalState, PeriodicSequenceState, TimedHoldGoalState, AcquireHoldState
from .state_machine import StateBase, StateMachine

SCAN_SWEEP_AMPLITUDE = math.radians(45.0)
SCAN_SWEEP_DT = 1.0
SCAN_TURN_DT = 0.5
SCAN_PITCH = 0.0
ACQUIRE_HOLD_DT = 0.8
NAV_ARRIVE_RADIUS = 0.1
RECENTER_PERIOD = 10.0
RETREAT_PERIOD = 60.0
RETREAT_HOLD_DT = 10.0

PLAY_STYLES = {"balanced", "center"}

TEAM_GOALS = {
  "red": {
    "center": (6.79, 3.24),
    "home": (11.01, 3.52),
  },
  "blue": {
    "center": (5.16, 3.21),
    "home": (0.91, 3.45),
  },
}

class IdleState(StateBase):
  name = "idle"

  def should_transition(self, current:StateBase, ctx) -> bool:
    return not bool(ctx["game_running"])

  def can_transition(self, ctx) -> bool:
    return True

  def run(self, ctx, pub):
    pass

class NavToCenterState(LookAtVisibleTagMixin, PeriodicNavToGoalState):
  name = "nav_to_center"
  goal_label = "center"
  arrive_radius = NAV_ARRIVE_RADIUS
  period = RECENTER_PERIOD
  first_delay = 0.0

class NavToHomeState(LookAtVisibleTagMixin, NavToGoalState):
  name = "nav_to_home"
  goal_label = "home"
  arrive_radius = NAV_ARRIVE_RADIUS

class HoldHomeState(ScanAcquireMixin, TimedHoldGoalState):
  name = "hold_home"
  goal_label = "home_hold"
  hold_dt = RETREAT_HOLD_DT

class ReturnCenterState(LookAtVisibleTagMixin, NavToGoalState):
  name = "return_center"
  goal_label = "center"
  arrive_radius = NAV_ARRIVE_RADIUS

class RetreatSequenceState(PeriodicSequenceState):
  name = "retreat_sequence"
  period = RETREAT_PERIOD
  first_delay = RETREAT_PERIOD

  def __init__(self, center_goal:tuple[float, float], back_goal:tuple[float, float]):
    super().__init__([NavToHomeState(back_goal), HoldHomeState(back_goal), ReturnCenterState(center_goal)])

class AcquireState(AcquireHoldState):
  name = "acquire"
  hold_dt = ACQUIRE_HOLD_DT

class SearchState(ScanAcquireMixin, StateBase):
  name = "search"
  sweep_amplitude = SCAN_SWEEP_AMPLITUDE
  sweep_dt = SCAN_SWEEP_DT
  turn_dt = SCAN_TURN_DT
  pitch = SCAN_PITCH

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"])

  def can_transition(self, ctx) -> bool:
    return True

  def run(self, ctx, pub):
    pass

def _goals(team_color:str) -> dict:
  if team_color not in TEAM_GOALS:
    raise ValueError(f"unknown team_color {team_color!r}; expected one of {sorted(TEAM_GOALS)}")
  return TEAM_GOALS[team_color]

def make_state_machine(team_color:str, play_style:str="balanced") -> StateMachine:
  play_style = play_style.lower()
  goals = _goals(team_color)

  nav_to_center = NavToCenterState(goals["center"])
  acquire = AcquireState()
  search = SearchState()
  retreat = RetreatSequenceState(goals["center"], goals["home"])

  states = [IdleState(), retreat, nav_to_center, acquire, search]
  return StateMachine(states)
