"""Generic class-based state machine for stated.

Each concrete state exposes three methods:
  should_transition(current, ctx) -> bool
  can_transition(ctx) -> bool
  run(ctx, pub) -> None
"""
from abc import ABC, abstractmethod
from typing import Any

class StateBase(ABC):
  name = "base"
  preempt = False

  def _call_hooks(self, prefix:str, *args):
    seen = set()
    for cls in type(self).__mro__:
      for name, fn in cls.__dict__.items():
        if name.startswith(prefix) and name not in seen and callable(fn):
          seen.add(name)
          fn(self, *args)

  def _ensure_hooks_initialized(self):
    if getattr(self, "_hooks_initialized", False): return
    self._call_hooks("_init_")
    self._hooks_initialized = True

  def reset(self, ctx:Any=None):
    pass

  def reset_state(self, ctx:Any=None):
    self._ensure_hooks_initialized()
    self.reset(ctx)
    self._call_hooks("_reset_", ctx)

  def observe(self, ctx:Any):
    pass

  def observe_state(self, ctx:Any):
    self._ensure_hooks_initialized()
    self.observe(ctx)
    self._call_hooks("_observe_", ctx)

  def run_state(self, ctx:Any, pub:Any):
    self._ensure_hooks_initialized()
    self.run(ctx, pub)
    self._call_hooks("_run_", ctx, pub)

  @abstractmethod
  def should_transition(self, current:"StateBase", ctx:Any) -> bool:
    pass

  @abstractmethod
  def can_transition(self, ctx:Any) -> bool:
    pass

  @abstractmethod
  def run(self, ctx:Any, pub:Any):
    pass

class StateMachine:
  def __init__(self, states:list[StateBase], initial:StateBase|None=None):
    assert states, "state machine needs at least one state"
    self.states = states
    self.current = initial or states[0]
    self.previous = self.current
    self.entered = False

  def tick(self, ctx:Any, pub:Any):
    for state in self.states:
      state.observe_state(ctx)
    next_state = self.current
    for state in self.states:
      if state.preempt and state.should_transition(self.current, ctx):
        next_state = state
        break
    if next_state is self.current and self.current.can_transition(ctx):
      for state in self.states:
        if state.should_transition(self.current, ctx):
          next_state = state
          break
    self.previous = self.current
    self.entered = next_state is not self.current
    if self.entered: self.current = next_state
    setattr(ctx, "entered", self.entered)
    self.current.run_state(ctx, pub)
