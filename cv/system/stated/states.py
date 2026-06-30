"""Concrete match states for stated.

State priority is the order returned by make_state_machine():
  IDLE -> RETREAT_SEQUENCE -> NAV_TO_CENTER -> ACQUIRE -> SEARCH
"""
import math

from .base_states import ResetState, PeriodicNavToGoalState, NavToGoalState, PeriodicSequenceState, TimedHoldGoalState, AcquireHoldState, SearchScanState
from .state_machine import StateMachine

SCAN_SWEEP_AMPLITUDE = math.radians(45.0)
SCAN_SWEEP_DT = 1.0
SCAN_TURN_DT = 0.5
SCAN_PITCH = 0.0
ACQUIRE_HOLD_DT = 0.8
NAV_ARRIVE_RADIUS = 0.25
RECENTER_PERIOD = 10.0
RETREAT_PERIOD = 60.0
RETREAT_HOLD_DT = 10.0

CENTER_GOAL = (11.02, 6.98)
BACK_GOAL = (11, 6)

class IdleState(ResetState):
  name = "idle"

class NavToCenterState(PeriodicNavToGoalState):
  name = "nav_to_center"
  goal_label = "center"
  goal_xy = CENTER_GOAL
  arrive_radius = NAV_ARRIVE_RADIUS
  period = RECENTER_PERIOD
  first_delay = 0.0

class NavToBackState(NavToGoalState):
  name = "nav_to_back"
  goal_label = "back"
  goal_xy = BACK_GOAL
  arrive_radius = NAV_ARRIVE_RADIUS

class HoldBackState(TimedHoldGoalState):
  name = "hold_back"
  goal_label = "back_hold"
  goal_xy = BACK_GOAL
  hold_dt = RETREAT_HOLD_DT

class ReturnCenterState(NavToGoalState):
  name = "return_center"
  goal_label = "center"
  goal_xy = CENTER_GOAL
  arrive_radius = NAV_ARRIVE_RADIUS

class RetreatSequenceState(PeriodicSequenceState):
  name = "retreat_sequence"
  period = RETREAT_PERIOD
  first_delay = RETREAT_PERIOD

  def __init__(self, recenter:NavToCenterState):
    super().__init__([NavToBackState(), HoldBackState(), ReturnCenterState()])
    self.recenter = recenter

  def on_complete(self, ctx):
    super().on_complete(ctx)
    self.recenter.mark_done(ctx)

class AcquireState(AcquireHoldState):
  name = "acquire"
  hold_dt = ACQUIRE_HOLD_DT

class SearchState(SearchScanState):
  name = "search"
  sweep_amplitude = SCAN_SWEEP_AMPLITUDE
  sweep_dt = SCAN_SWEEP_DT
  turn_dt = SCAN_TURN_DT
  pitch = SCAN_PITCH

def make_state_machine() -> StateMachine:
  nav_to_center = NavToCenterState()
  retreat_sequence = RetreatSequenceState(nav_to_center)
  acquire = AcquireState()
  reset_states = [retreat_sequence, nav_to_center, acquire]
  states = [IdleState(reset_states), retreat_sequence, nav_to_center, acquire, SearchState()]
  return StateMachine(states)
