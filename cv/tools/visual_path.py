"""Standalone Rerun visualizer for the blended waypoint path.

Default (simulated):  python -m cv.tools.visual_path
Live mode:            python -m cv.tools.visual_path <robot-ip>
"""
import sys, time, math
from collections import deque

import rerun as rr

from ..system.decisiond.decisiond import WaypointFollower, _build_path, LineSegment, ArcSegment
from ..system.core import messaging
from ..system.core.helpers import FrequencyKeeper

# path config — must match decisiond.run()
WAYPOINTS = [(0, 0), (6.2, 0), (6.2, 6.2), (0, 6.2), (0, 0)]
SPEED = 1.03
BLEND_RADIUS = 0.5

# visualization
ARC_SAMPLE_STEP = 0.05  # radians (~3 degrees)
TRAIL_LEN = 200
PAUSE_AFTER_PATH = 2.0  # seconds before looping

# -- path geometry helpers --

def sample_path_points(segments, start):
  """Sample dense (x, z) points along the full blended path."""
  points = []
  pos = list(start)

  for seg in segments:
    if isinstance(seg, LineSegment):
      points.append((pos[0], pos[1]))
      pos[0] += seg.direction[0] * seg.length
      pos[1] += seg.direction[1] * seg.length
    elif isinstance(seg, ArcSegment):
      n = max(2, int(abs(seg.sweep) / ARC_SAMPLE_STEP))
      for j in range(n + 1):
        frac = j / n
        angle = seg.start_angle + frac * seg.sweep
        points.append((
          seg.center[0] + seg.radius * math.cos(angle),
          seg.center[1] + seg.radius * math.sin(angle),
        ))
      end_angle = seg.start_angle + seg.sweep
      pos[0] = seg.center[0] + seg.radius * math.cos(end_angle)
      pos[1] = seg.center[1] + seg.radius * math.sin(end_angle)

  points.append((pos[0], pos[1]))
  return points

def position_at(segments, start, elapsed):
  """Compute (x, z) position at a given elapsed time along the path."""
  pos = list(start)
  t = elapsed

  for seg in segments:
    if t <= seg.duration:
      if isinstance(seg, LineSegment):
        frac = t / seg.duration if seg.duration > 0 else 0
        return (pos[0] + seg.direction[0] * seg.length * frac,
                pos[1] + seg.direction[1] * seg.length * frac)
      else:
        frac = t / seg.duration if seg.duration > 0 else 0
        angle = seg.start_angle + frac * seg.sweep
        return (seg.center[0] + seg.radius * math.cos(angle),
                seg.center[1] + seg.radius * math.sin(angle))
    t -= seg.duration
    if isinstance(seg, LineSegment):
      pos[0] += seg.direction[0] * seg.length
      pos[1] += seg.direction[1] * seg.length
    else:
      end_angle = seg.start_angle + seg.sweep
      pos[0] = seg.center[0] + seg.radius * math.cos(end_angle)
      pos[1] = seg.center[1] + seg.radius * math.sin(end_angle)

  return (pos[0], pos[1])

def velocity_at(segments, elapsed):
  """Compute (vx, vz) velocity at a given elapsed time."""
  t = elapsed
  for seg in segments:
    if t <= seg.duration:
      return seg.velocity(t)
    t -= seg.duration
  return (0.0, 0.0)

# -- main --

def main():
  addr = sys.argv[1] if len(sys.argv) > 1 else None
  live = addr is not None

  rr.init("waypoint_path")
  rr.connect_grpc()

  # build path
  segments = _build_path(WAYPOINTS, SPEED, BLEND_RADIUS)
  total_duration = sum(s.duration for s in segments)
  start = WAYPOINTS[0]

  # -- static: planned path --
  path_points = sample_path_points(segments, start)
  rr.log("path/planned", rr.LineStrips2D([path_points], colors=[[100, 100, 255]], radii=[0.03]), static=True)

  # -- static: waypoint markers --
  wp_points = [(x, z) for x, z in WAYPOINTS]
  wp_labels = [f"WP{i} ({x},{z})" for i, (x, z) in enumerate(WAYPOINTS)]
  rr.log("path/waypoints", rr.Points2D(wp_points, colors=[[255, 200, 0]], radii=[0.08], labels=wp_labels), static=True)

  # -- static: segment boundaries (where lines meet arcs) --
  boundary_points = []
  pos = list(start)
  for seg in segments:
    boundary_points.append((pos[0], pos[1]))
    if isinstance(seg, LineSegment):
      pos[0] += seg.direction[0] * seg.length
      pos[1] += seg.direction[1] * seg.length
    else:
      end_angle = seg.start_angle + seg.sweep
      pos[0] = seg.center[0] + seg.radius * math.cos(end_angle)
      pos[1] = seg.center[1] + seg.radius * math.sin(end_angle)
  boundary_points.append((pos[0], pos[1]))
  rr.log("path/blend_points", rr.Points2D(boundary_points, colors=[[100, 100, 255]], radii=[0.04]), static=True)

  # -- live mode setup --
  sub = None
  real_pos = [0.0, 0.0]
  real_trail = deque(maxlen=TRAIL_LEN)
  if live:
    sub = messaging.Sub(["chassis_velocity"], addr=addr)
    print(f"live mode: subscribing to chassis_velocity at {addr}")

  # -- animation loop --
  sim_trail = deque(maxlen=TRAIL_LEN)
  fk = FrequencyKeeper(60)
  loop_start = time.monotonic()
  last_t = loop_start

  while True:
    now = time.monotonic()
    wall_dt = now - last_t
    last_t = now
    rr.set_time_seconds("time", now)

    # simulated cursor (always runs, loops after path + pause)
    loop_period = total_duration + PAUSE_AFTER_PATH
    elapsed_in_loop = (now - loop_start) % loop_period
    sim_t = min(elapsed_in_loop, total_duration)

    px, pz = position_at(segments, start, sim_t)
    vx, vz = velocity_at(segments, sim_t)

    rr.log("path/cursor", rr.Points2D([(px, pz)], colors=[[255, 50, 50]], radii=[0.12]))

    # velocity arrow (scaled down for visibility)
    if abs(vx) + abs(vz) > 1e-6:
      rr.log("path/velocity", rr.Arrows2D(origins=[(px, pz)], vectors=[(vx * 0.5, vz * 0.5)], colors=[[50, 255, 50]]))

    sim_trail.append((px, pz))
    if len(sim_trail) > 1:
      rr.log("path/trail", rr.LineStrips2D([list(sim_trail)], colors=[[255, 100, 100]], radii=[0.02]))

    rr.log("progress", rr.Scalar(sim_t / total_duration * 100))

    # live cursor (only if connected to robot)
    if live:
      sub.update(timeout=0)
      cv = sub["chassis_velocity"]
      if sub.updated["chassis_velocity"] and cv is not None:
        real_pos[0] += cv["x"] * wall_dt
        real_pos[1] += cv["z"] * wall_dt
        real_trail.append(tuple(real_pos))

      rr.log("path/real_cursor", rr.Points2D([tuple(real_pos)], colors=[[50, 255, 50]], radii=[0.12]))
      if len(real_trail) > 1:
        rr.log("path/real_trail", rr.LineStrips2D([list(real_trail)], colors=[[50, 200, 50]], radii=[0.02]))

    fk.step()

if __name__ == "__main__":
  main()
