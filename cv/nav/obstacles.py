"""Persisting + decaying dynamic obstacles (enemy robots) for navd's planner.

plated tracks the enemy ONE robot at a time and only enemies (not allies/self), so navd
records each detected enemy world position with a timestamp and keeps it alive for a short
TTL — a robot you've looked away from lingers as an obstacle, then fades. A detection within
MERGE_DIST of an existing track updates it (the same robot moved); a farther one starts a new
track (a different enemy). `active(now)` returns (x, y, age) per live track so the caller can
grow the painted radius with staleness (a stale position is less certain → inflate more).
"""

class RobotObstacles:
  def __init__(self, ttl:float = 2.0, merge_dist:float = 0.5):
    self.ttl = ttl
    self.merge_dist2 = merge_dist * merge_dist
    self.tracks: list[list[float]] = []        # [x, y, t_seen]

  def update(self, x:float, y:float, now:float):
    for tr in self.tracks:
      if (tr[0] - x) ** 2 + (tr[1] - y) ** 2 <= self.merge_dist2:
        tr[0], tr[1], tr[2] = x, y, now        # same robot moved
        return
    self.tracks.append([x, y, now])

  def active(self, now:float):
    """Live tracks as (x, y, age_seconds); also prunes ones past TTL."""
    self.tracks = [tr for tr in self.tracks if now - tr[2] < self.ttl]
    return [(tr[0], tr[1], now - tr[2]) for tr in self.tracks]
