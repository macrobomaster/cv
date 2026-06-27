"""Shared numpy geometry / angle helpers for the daemons (plated, decisiond, slamd).

Only the elementary axis rotations live here — each consumer composes them for its
own frame convention (plated's gimbal-inertial is y-up: yaw=roty, pitch=rotx;
slamd's world is z-up: yaw=rotz, pitch=roty), so the primitives stay shared without
baking a frame into them. Returns float64; cast at the call site if needed.
"""
import math

import numpy as np

def wrap_pi(x:float) -> float:
  return (x + math.pi) % (2 * math.pi) - math.pi

def rotx(a:float) -> np.ndarray:
  c, s = math.cos(a), math.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(a:float) -> np.ndarray:
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(a:float) -> np.ndarray:
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
