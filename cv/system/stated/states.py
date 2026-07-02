"""Concrete match state machines for stated.

Factory input:
  team_color: our alliance color ("red"/"blue")
  play_style: high-level strategy preset ("balanced"/"center")
"""
import os, math

from .base_states import (LookAtMappedTagMixin, ScanAcquireMixin, SpinChassisMixin, NavToGoalState,
                          PeriodicNavToGoalState, TimedNavToGoalState, SequenceState, TimedHoldGoalState,
                          AcquireHoldState, PeriodicSequenceState)
from .state_machine import StateBase, StateMachine

SCAN_SWEEP_AMPLITUDE = math.radians(45.0)
SCAN_SWEEP_DT = 1.0
SCAN_TURN_DT = 0.5
SCAN_PITCH = 0.0
ACQUIRE_HOLD_DT = 0.8
NAV_ARRIVE_RADIUS = 0.1
RECENTER_PERIOD = 30.0
RETREAT_HOLD_DT = 10.0
RETREAT_HP = 150
ATTACK_DEEP_PERIOD = 10
ATTACK_DEEP_AT = 120
ATTACK_DEEP_HOLD_DT = 10.0

PLAY_STYLES = {"passive", "aggressive"}

TEAM_GOALS = {
  "blue": {
    "passive_center": (6.47, 3.52),
    "aggressive_center": (5.48, 3.52),
    "home": (11.16, 3.52),
    "attack_deep": (2.62, 1.00),
  },
  "red": {
    "passive_center": (5.48, 3.52),
    "aggressive_center": (6.47, 3.52),
    "home": (0.73, 3.52),
    "attack_deep": (9.38, 1.00),
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

class NavToCenterState(LookAtMappedTagMixin, PeriodicNavToGoalState):
  name = "nav_to_center"
  goal_label = "center"
  arrive_radius = NAV_ARRIVE_RADIUS
  period = RECENTER_PERIOD
  first_delay = 0.0

class NavToHomeState(LookAtMappedTagMixin, NavToGoalState):
  name = "nav_to_home"
  goal_label = "home"
  arrive_radius = NAV_ARRIVE_RADIUS

class HoldHomeState(SpinChassisMixin, ScanAcquireMixin, TimedHoldGoalState):
  name = "hold_home"
  goal_label = "home_hold"
  hold_dt = RETREAT_HOLD_DT

class ReturnCenterState(LookAtMappedTagMixin, NavToGoalState):
  name = "return_center"
  goal_label = "center"
  arrive_radius = NAV_ARRIVE_RADIUS

def _robot_hp(ctx) -> float|None:
  hp = ctx["robot_hp"]
  return None if hp is None else float(hp)

class RetreatSequenceState(SequenceState):
  name = "retreat_sequence"
  preempt = True
  hp_threshold = RETREAT_HP

  def __init__(self, center_goal:tuple[float, float], back_goal:tuple[float, float]):
    super().__init__([NavToHomeState(back_goal), HoldHomeState(back_goal), ReturnCenterState(center_goal)])
    self.armed = True

  def reset(self, ctx=None):
    super().reset(ctx)
    self.armed = True

  def observe(self, ctx):
    if not bool(ctx["game_running"]):
      self.reset(ctx)
      return
    super().observe(ctx)
    hp = _robot_hp(ctx)
    if hp is not None and hp > self.hp_threshold: self.armed = True

  def should_transition(self, current:StateBase, ctx) -> bool:
    hp = _robot_hp(ctx)
    return current is not self and bool(ctx["game_running"]) and self.armed and hp is not None and hp <= self.hp_threshold

  def run(self, ctx, pub):
    entered = ctx.entered
    super().run(ctx, pub)
    if entered: self.armed = False

class NavToAttackDeepState(SpinChassisMixin, ScanAcquireMixin, TimedNavToGoalState):
  name = "attack_deep"
  goal_label = "attack_deep"
  arrive_radius = NAV_ARRIVE_RADIUS
  trigger_at = ATTACK_DEEP_AT

class HoldAttackDeepState(SpinChassisMixin, ScanAcquireMixin, TimedHoldGoalState):
  name = "hold_attack_deep"
  goal_label = "attack_deep_hold"
  hold_dt = ATTACK_DEEP_HOLD_DT

class AttackDeepState(PeriodicSequenceState):
  name = "attack_deep_sequence"
  period = ATTACK_DEEP_PERIOD
  first_delay = ATTACK_DEEP_AT

  def __init__(self, attack_deep_goal:tuple[float, float]):
    super().__init__([NavToAttackDeepState(attack_deep_goal), HoldAttackDeepState(attack_deep_goal)])

class AcquireState(SpinChassisMixin, AcquireHoldState):
  name = "acquire"
  hold_dt = ACQUIRE_HOLD_DT

class SearchState(SpinChassisMixin, ScanAcquireMixin, StateBase):
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

def make_state_machine(team_color:str, play_style:str="passive") -> StateMachine:
  play_style = play_style.lower()
  goals = _goals(team_color)

  match play_style:
    case "passive":
      center_goal = goals["passive_center"]
    case "aggressive":
      center_goal = goals["aggressive_center"]
    case _:
      center_goal = goals["passive_center"]

  nav_to_center = NavToCenterState(center_goal)
  acquire = AcquireState()
  search = SearchState()
  retreat = RetreatSequenceState(center_goal, goals["home"])

  states = [IdleState(), retreat, nav_to_center, acquire, search]
  return StateMachine(states)
