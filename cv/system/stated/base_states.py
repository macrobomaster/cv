"""Reusable stated state building blocks.

Concrete match behavior belongs in states.py. This file holds state classes that encode common mechanics:
resetting, navigation to a goal, periodic navigation, timed holds, acquisition hysteresis, and scanning.
"""
import math, time

from ..core.logging import logger
from ..common.geometry import wrap_pi
from .state_machine import StateBase

class NavToGoalState(StateBase):
  name = "nav_to_goal"
  goal_label = "goal"
  goal_xy: tuple[float, float]|None = None
  arrive_radius = 0.25

  def reset(self, ctx=None):
    pass

  def observe(self, ctx):
    if not bool(ctx["game_running"]): self.reset(ctx)

  def arrived(self, ctx) -> bool:
    if self.goal_xy is None: return True
    pose = ctx["slam_pose"]
    if pose is None: return False
    px, py = pose["p_w"][:2]
    return math.hypot(float(px) - self.goal_xy[0], float(py) - self.goal_xy[1]) < self.arrive_radius

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.goal_xy is not None

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    return self.arrived(ctx)

  def run(self, ctx, pub):
    if ctx.entered: logger.info(f"stated: navigating to {self.goal_label}")
    pub.send("nav_goal", {"x": self.goal_xy[0], "y": self.goal_xy[1], "label": self.goal_label})

class PeriodicNavToGoalState(NavToGoalState):
  period = math.inf
  first_delay = 0.0

  def __init__(self):
    self.next_due = time.monotonic() + self.first_delay

  def reset(self, ctx=None):
    self.next_due = (ctx.now if ctx is not None else time.monotonic()) + self.first_delay

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.goal_xy is not None and ctx.now >= self.next_due

  def mark_done(self, ctx):
    self.next_due = ctx.now + self.period

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    if self.arrived(ctx):
      self.mark_done(ctx)
      return True
    return False

class SequenceState(StateBase):
  name = "sequence"

  def __init__(self, states:list[StateBase]):
    self.states = states
    self.idx = 0
    self.child_started = False

  def reset(self, ctx=None):
    self.idx = 0
    self.child_started = False
    for state in self.states:
      reset = getattr(state, "reset", None)
      if reset is not None: reset(ctx)

  def observe(self, ctx):
    if not bool(ctx["game_running"]): self.reset(ctx)

  def should_transition(self, current:StateBase, ctx) -> bool:
    return False

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    if self.idx >= len(self.states): return True
    if self.child_started and self.states[self.idx].can_transition(ctx):
      self.idx += 1
      self.child_started = False
      if self.idx >= len(self.states):
        self.on_complete(ctx)
        return True
    return False

  def on_complete(self, ctx):
    pass

  def run(self, ctx, pub):
    if ctx.entered: self.reset(ctx)
    if self.idx >= len(self.states): return
    child = self.states[self.idx]
    child_entered = not self.child_started
    self.child_started = True
    machine_entered = ctx.entered
    ctx.entered = child_entered
    child.run(ctx, pub)
    ctx.entered = machine_entered

class PeriodicSequenceState(SequenceState):
  period = math.inf
  first_delay = 0.0

  def __init__(self, states:list[StateBase]):
    super().__init__(states)
    self.next_due = time.monotonic() + self.first_delay

  def reset(self, ctx=None):
    super().reset(ctx)
    self.next_due = (ctx.now if ctx is not None else time.monotonic()) + self.first_delay

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and ctx.now >= self.next_due

  def on_complete(self, ctx):
    self.next_due = ctx.now + self.period

class TimedHoldGoalState(StateBase):
  name = "timed_hold_goal"
  goal_label = "hold"
  goal_xy: tuple[float, float]|None = None
  hold_dt = 1.0

  def __init__(self):
    self.until = -math.inf

  def reset(self, ctx=None):
    self.until = -math.inf

  def should_transition(self, current:StateBase, ctx) -> bool:
    return False

  def can_transition(self, ctx) -> bool:
    if not bool(ctx["game_running"]): return True
    return ctx.now >= self.until

  def run(self, ctx, pub):
    if ctx.entered:
      self.until = ctx.now + self.hold_dt
      logger.info(f"stated: holding {self.goal_label}")
    pub.send("nav_goal", {"x": self.goal_xy[0], "y": self.goal_xy[1], "label": self.goal_label})

class AcquireHoldState(StateBase):
  name = "acquire"
  hold_dt = 0.8

  def __init__(self):
    self.last_valid_t = -math.inf

  def reset(self, ctx=None):
    self.last_valid_t = -math.inf

  def observe(self, ctx):
    if not bool(ctx["game_running"]):
      self.reset(ctx)
      return
    autoaim = ctx["autoaim"]
    if ctx.updated["autoaim"] and autoaim is not None and autoaim.get("valid", False): self.last_valid_t = ctx.now

  def recently_saw_plate(self, now:float) -> bool:
    return now - self.last_valid_t < self.hold_dt

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"]) and self.recently_saw_plate(ctx.now)

  def can_transition(self, ctx) -> bool:
    return (not bool(ctx["game_running"])) or not self.recently_saw_plate(ctx.now)

  def run(self, ctx, pub):
    pass

class SearchScanState(StateBase):
  name = "search"
  sweep_amplitude = math.radians(45.0)
  sweep_dt = 1.0
  turn_dt = 0.5
  pitch = 0.0

  def __init__(self):
    self.scan_center = 0.0
    self.scan_start_t = time.monotonic()
    self.have_scan_center = False

  def should_transition(self, current:StateBase, ctx) -> bool:
    return bool(ctx["game_running"])

  def can_transition(self, ctx) -> bool:
    return True

  def setpoint(self, t:float) -> dict:
    cycle_dt = self.sweep_dt + self.turn_dt
    t_scan = t - self.scan_start_t
    cycle = int(t_scan // cycle_dt)
    t_cycle = t_scan - cycle * cycle_dt
    base = self.scan_center + (cycle % 2) * math.pi
    if t_cycle < self.sweep_dt:
      w = 2 * math.pi / self.sweep_dt
      yaw = wrap_pi(base + self.sweep_amplitude * math.sin(w * t_cycle))
      yaw_ff = self.sweep_amplitude * w * math.cos(w * t_cycle)
    else:
      yaw_ff = math.pi / self.turn_dt
      yaw = wrap_pi(base + yaw_ff * (t_cycle - self.sweep_dt))
    return {"yaw": yaw, "pitch": self.pitch, "yaw_ff": yaw_ff, "pitch_ff": 0.0}

  def run(self, ctx, pub):
    gs = ctx["gimbal_state"]
    if gs is not None and (ctx.entered or not self.have_scan_center):
      self.scan_center = gs["yaw_gi"]
      self.have_scan_center = True
    if ctx.entered: self.scan_start_t = ctx.now
    pub.send("state_setpoint", self.setpoint(ctx.now))
