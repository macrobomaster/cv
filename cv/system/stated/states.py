"""Concrete match state machines for stated.

Factory input:
  team_color: our alliance color ("red"/"blue")
  play_style: high-level strategy preset ("balanced"/"center")
"""
import math

from ..core.logging import logger
from .base_states import ScanAcquireMixin, NavToGoalState, PeriodicNavToGoalState, PeriodicSequenceState, TimedHoldGoalState, AcquireHoldState
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

class NavToHome(NavToGoalState):
  name = "nav_to_home"
  goal_label = "home"
  arrive_radius = NAV_ARRIVE_RADIUS

def make_state_machine(team_color:str, play_style:str="balanced") -> StateMachine:
  play_style = play_style.lower()

  acquire = AcquireState()
  search = SearchState()
  nav_to_home = NavToHome((1.5, 0.15))

  states = [IdleState(), nav_to_home, acquire, search]
  return StateMachine(states)
