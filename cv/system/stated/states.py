"""Concrete match states for stated.

State priority is the order returned by make_state_machine():
  IDLE -> NAV_TO_CENTER -> ACQUIRE -> SEARCH

Only nav states publish nav_goal. navd treats nav_goal as fresh-only, so every other state
clears navigation by not publishing it.
"""
import os, math, time

from ..core.logging import logger
from ..common.geometry import wrap_pi
from ...slam import common
from .state_machine import StateBase, StateMachine

SCAN_SWEEP_AMPLITUDE = math.radians(45.0)
SCAN_SWEEP_DT = 1.0
SCAN_TURN_DT = 0.5
SCAN_PITCH = 0.0
ACQUIRE_HOLD_DT = 0.8
NAV_ARRIVE_RADIUS = 0.25

def scan_setpoint(t:float, center:float, start_t:float) -> dict:
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

class IdleState(StateBase):
  name = "idle"

  def __init__(self, reset_states:list[StateBase]):
    self.reset_states = reset_states

  def should_transition(self, current:StateBase, ctx) -> bool:
    return not bool(ctx["game_running"])

  def can_transition(self, ctx) -> bool:
    return True

  def run(self, ctx, pub):
    for state in self.reset_states:
      reset = getattr(state, "reset", None)
      if reset is not None: reset()

class NavToGoalState(StateBase):
  name = "nav_to_goal"
  goal_label = "goal"
  goal_xy: tuple[float, float]|None = None

  def __init__(self):
    self.done = False

  def reset(self):
    self.done = False

  def arrived(self, ctx) -> bool:
    if self.goal_xy is None: return True
    pose = ctx["slam_pose"]
    if pose is None: return False
    px, py = pose["p_w"][:2]
    return math.hypot(float(px) - self.goal_xy[0], float(py) - self.goal_xy[1]) < NAV_ARRIVE_RADIUS

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and not self.done and self.goal_xy is not None

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    if self.arrived(ctx):
      self.done = True
      return True
    return False

  def run(self, ctx, pub):
    if ctx.entered: logger.info(f"stated: navigating to {self.goal_label}")
    pub.send("nav_goal", {"x": self.goal_xy[0], "y": self.goal_xy[1], "label": self.goal_label})

class NavToCenterState(NavToGoalState):
  name = "nav_to_center"
  goal_label = "center"
  goal_xy = (11.02, 6.98)

class AcquireState(StateBase):
  name = "acquire"

  def __init__(self):
    self.last_valid_t = -math.inf

  def reset(self):
    self.last_valid_t = -math.inf

  def observe(self, ctx):
    autoaim = ctx["autoaim"]
    if ctx.updated["autoaim"] and autoaim is not None and autoaim.get("valid", False): self.last_valid_t = ctx.now

  def recently_saw_plate(self, now:float) -> bool:
    return now - self.last_valid_t < ACQUIRE_HOLD_DT

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.recently_saw_plate(ctx.now)

  def can_transition(self, ctx) -> bool:
    return (not bool(ctx["game_running"])) or not self.recently_saw_plate(ctx.now)

  def run(self, ctx, pub):
    pass

class SearchState(StateBase):
  name = "search"

  def __init__(self):
    self.scan_center = 0.0
    self.scan_start_t = time.monotonic()
    self.have_scan_center = False

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"])

  def can_transition(self, ctx) -> bool:
    return True

  def run(self, ctx, pub):
    gs = ctx["gimbal_state"]
    if gs is not None and (ctx.entered or not self.have_scan_center):
      self.scan_center = gs["yaw_gi"]
      self.have_scan_center = True
    if ctx.entered: self.scan_start_t = ctx.now
    pub.send("state_setpoint", scan_setpoint(ctx.now, self.scan_center, self.scan_start_t))

def make_state_machine() -> StateMachine:
  nav_to_center = NavToCenterState()
  acquire = AcquireState()
  states = [IdleState([nav_to_center, acquire]), nav_to_center, acquire, SearchState()]
  return StateMachine(states)
