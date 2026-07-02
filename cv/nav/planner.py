"""Any-angle path planning (A* + string-pull) on an OccupancyGrid for navd.

A* (8-connected) finds a grid path; a greedy line-of-sight "string-pull" then
collapses it to sparse ANY-ANGLE corner waypoints (no grid staircase). This is far
cheaper than per-expansion Theta* (no line-of-sight in the hot loop — only once per
kept vertex at the end) and gives essentially the same path; pure pursuit rounds the
corners anyway. Output is world-XY points start→goal — hand straight to PurePursuit.

Deterministic and sub-ms on a small field (~11k cells), so the navd loop just
replans each cycle — "incremental" enough here. The pure plan(grid, start, goal)
interface leaves room to swap in D* Lite if a bigger map ever wants search reuse.
"""
import math, heapq

_SQRT2 = math.sqrt(2.0)
# (dx, dy, step-cost) — orthogonals first so ties favor straight moves
_NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2)]

def line_of_sight(grid, a, b) -> bool:
  """True if the segment between cell centers a→b stays in free cells (supercover)."""
  (ax, ay), (bx, by) = a, b
  if not grid.is_free(ax, ay) or not grid.is_free(bx, by): return False
  dx, dy = bx - ax, by - ay
  nx, ny = abs(dx), abs(dy)
  sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
  x, y, ix, iy = ax, ay, 0, 0
  while ix < nx or iy < ny:
    dec = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
    if dec == 0:    x += sx; y += sy; ix += 1; iy += 1   # exact diagonal
    elif dec < 0:   x += sx; ix += 1
    else:           y += sy; iy += 1
    if not grid.is_free(x, y): return False
  return True

def _snap(grid, c, max_r:int = 25):
  """Nearest free cell to c (spiral out), or None — for a start/goal that landed in
  an (inflated) obstacle, e.g. the robot hugging a wall or a goal too near one."""
  if grid.is_free(*c): return c
  cx, cy = c
  for r in range(1, max_r + 1):
    for dx in range(-r, r + 1):
      for dy in range(-r, r + 1):
        if max(abs(dx), abs(dy)) == r and grid.is_free(cx + dx, cy + dy):
          return (cx + dx, cy + dy)
  return None

def _string_pull(grid, cells):
  """Collapse a grid path to sparse any-angle vertices: from each kept vertex, jump
  to the farthest still-visible cell ahead."""
  if len(cells) <= 2: return cells
  out, i, n = [cells[0]], 0, len(cells)
  while i < n - 1:
    j = n - 1
    while j > i + 1 and not line_of_sight(grid, cells[i], cells[j]): j -= 1
    out.append(cells[j]); i = j
  return out

def plan(grid, start_xy, goal_xy):
  """A*+string-pull path (list of world (x,y) start→goal), or None if unreachable."""
  start = _snap(grid, grid.world_to_cell(*start_xy))
  goal = _snap(grid, grid.world_to_cell(*goal_xy))
  if start is None or goal is None: return None
  gx, gy = goal
  # Inline the free check over local refs (no method-call / in_bounds overhead) — the A* hot loop
  # runs this millions of times at fine resolution, so it dominates the plan time.
  occ, nx, ny = grid.occ, grid.nx, grid.ny
  free = lambda ix, iy: 0 <= ix < nx and 0 <= iy < ny and not occ[iy, ix]

  parent = {start: None}
  gscore = {start: 0.0}
  open_heap = [(math.hypot(start[0] - gx, start[1] - gy), start)]
  closed = set()
  found = start == goal
  while open_heap:
    _, c = heapq.heappop(open_heap)
    if c == goal: found = True; break
    if c in closed: continue
    closed.add(c)
    cx, cy, gc = c[0], c[1], gscore[c]
    for ox, oy, step in _NB:
      nb = (cx + ox, cy + oy)
      if nb in closed or not free(nb[0], nb[1]): continue
      if ox and oy and (not free(cx + ox, cy) or not free(cx, cy + oy)): continue  # no corner-cut
      ng = gc + step
      if ng < gscore.get(nb, math.inf):
        gscore[nb] = ng; parent[nb] = c
        heapq.heappush(open_heap, (ng + math.hypot(nb[0] - gx, nb[1] - gy), nb))
  if not found: return None

  cells = []
  c = goal
  while c is not None:
    cells.append(c); c = parent[c]
  cells.reverse()
  pts = [list(grid.cell_to_world(ix, iy)) for ix, iy in _string_pull(grid, cells)]
  pts[0] = [float(start_xy[0]), float(start_xy[1])]       # exact robot start for the follower
  if grid.world_to_cell(*goal_xy) == goal:               # goal wasn't snapped out of an obstacle
    pts[-1] = [float(goal_xy[0]), float(goal_xy[1])]
  return pts
