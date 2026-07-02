"""Shared gimbal_state ring buffer with capture-time interpolation.

commsd publishes gimbal_state non-conflated at ~200 Hz. Consumers that need the
gimbal pose at a past instant (plated/slamd de-rotating a frame captured a few ms
ago) call interpolate(t); the control loop (decisiond) takes latest(). Subscribe
non-conflated and drain() every tick so no sample is dropped.
"""
from collections import deque
from typing import Optional

from .geometry import wrap_pi

DEQUE_MAX = 200          # ~1 s at 200 Hz; tolerates large t_capture lag
STALE_GAP = 0.030        # s — nearest sample farther than this from t ⇒ flagged stale

class GimbalBuffer:
  def __init__(self):
    self.samples: deque = deque(maxlen=DEQUE_MAX)

  def push(self, msg:dict):
    self.samples.append((msg["t_stamp"], msg["yaw_gi"], msg["pitch_gi"],
                         msg["yaw_rate_gi"], msg["pitch_rate_gi"]))

  def latest(self) -> Optional[tuple[float, float, float, float]]:
    """(yaw, pitch, yaw_rate, pitch_rate) of the newest sample, or None if empty."""
    if not self.samples: return None
    _, y, p, yr, pr = self.samples[-1]
    return y, p, yr, pr

  def interpolate(self, t:float) -> Optional[tuple[float, float, bool]]:
    """(yaw, pitch, stale) linearly interpolated to t, or None if the buffer is empty.
    Clamps to the ends; stale=True when the nearest sample is farther than STALE_GAP."""
    if not self.samples: return None
    if t <= self.samples[0][0]:
      ts, y, p, _, _ = self.samples[0]
      return y, p, (ts - t) > STALE_GAP
    if t >= self.samples[-1][0]:
      ts, y, p, _, _ = self.samples[-1]
      return y, p, (t - ts) > STALE_GAP
    # linear bracket — N is small (≤DEQUE_MAX), linear scan is fine
    prev_ts, prev_y, prev_p = self.samples[0][:3]
    for s in list(self.samples)[1:]:
      ts, y, p = s[:3]
      if prev_ts <= t <= ts:
        a = (t - prev_ts) / (ts - prev_ts)
        # yaw is ±π-wrapped: interpolate the SHORTEST arc, else a pair straddling the branch cut (+179°→−179°)
        # linearly averages to ~0° and throws the de-rotation off by ~180°. pitch doesn't wrap.
        return wrap_pi(prev_y + a * wrap_pi(y - prev_y)), prev_p + a * (p - prev_p), False
      prev_ts, prev_y, prev_p = ts, y, p
    return None
