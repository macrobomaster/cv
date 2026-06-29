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

  def observe(self, ctx:Any):
    pass

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
      state.observe(ctx)
    next_state = self.current
    if self.current.can_transition(ctx):
      for state in self.states:
        if state.should_transition(self.current, ctx):
          next_state = state
          break
    self.previous = self.current
    self.entered = next_state is not self.current
    if self.entered: self.current = next_state
    setattr(ctx, "entered", self.entered)
    self.current.run(ctx, pub)
