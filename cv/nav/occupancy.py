"""2D occupancy grid + obstacle inflation for navd path planning.

World frame is slam's RH z-up; planning happens in the horizontal x-y plane
(metres). The grid stores RAW obstacles (static walls + dynamic robots); call
`inflated(robot_radius)` to get a configuration-space copy (obstacles dilated by
the robot radius) so the planner can treat the robot as a point.

Static walls come from a map config; dynamic obstacles (detected robots) are
painted as circles each replan onto a copy of the static grid.
"""
import math

import numpy as np

# Robot footprint RADIUS (m, RoboMaster half-diagonal) — the C-space inflation radius.
# Single source: navd plans with it and path_editor previews/exports with it, so they agree.
ROBOT_RADIUS = 0.28

class OccupancyGrid:
  def __init__(self, x0:float, y0:float, x1:float, y1:float, res:float):
    self.x0, self.y0, self.res = float(x0), float(y0), float(res)
    self.nx = max(1, int(math.ceil((x1 - x0) / res)))
    self.ny = max(1, int(math.ceil((y1 - y0) / res)))
    self.occ = np.zeros((self.ny, self.nx), dtype=bool)   # [iy, ix], True = blocked

  # --- world <-> cell ---
  def world_to_cell(self, x:float, y:float):
    return (int(math.floor((x - self.x0) / self.res)), int(math.floor((y - self.y0) / self.res)))
  def cell_to_world(self, ix:int, iy:int):
    return (self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res)
  def in_bounds(self, ix:int, iy:int) -> bool:
    return 0 <= ix < self.nx and 0 <= iy < self.ny
  def is_free(self, ix:int, iy:int) -> bool:
    return self.in_bounds(ix, iy) and not self.occ[iy, ix]

  # --- obstacle painting (world coords) ---
  def add_rect(self, x0:float, y0:float, x1:float, y1:float):
    ax, ay = self.world_to_cell(min(x0, x1), min(y0, y1))
    bx, by = self.world_to_cell(max(x0, x1), max(y0, y1))
    ax, ay, bx, by = max(0, ax), max(0, ay), min(self.nx - 1, bx), min(self.ny - 1, by)
    if ax <= bx and ay <= by: self.occ[ay:by + 1, ax:bx + 1] = True

  def add_circle(self, cx:float, cy:float, r:float):
    ax, ay = self.world_to_cell(cx - r, cy - r)
    bx, by = self.world_to_cell(cx + r, cy + r)
    ax, ay, bx, by = max(0, ax), max(0, ay), min(self.nx - 1, bx), min(self.ny - 1, by)
    if ax > bx or ay > by: return
    iy, ix = np.ogrid[ay:by + 1, ax:bx + 1]
    wx = self.x0 + (ix + 0.5) * self.res
    wy = self.y0 + (iy + 0.5) * self.res
    self.occ[ay:by + 1, ax:bx + 1] |= ((wx - cx) ** 2 + (wy - cy) ** 2 <= r * r)

  def add_poly(self, pts):
    import cv2
    cells = np.array([[self.world_to_cell(x, y) for x, y in pts]], dtype=np.int32)
    mask = np.zeros((self.ny, self.nx), np.uint8)
    cv2.fillPoly(mask, cells, 1)
    self.occ |= mask.astype(bool)

  def copy(self) -> "OccupancyGrid":
    g = OccupancyGrid.__new__(OccupancyGrid)
    g.x0, g.y0, g.res, g.nx, g.ny = self.x0, self.y0, self.res, self.nx, self.ny
    g.occ = self.occ.copy()
    return g

  def inflated(self, radius:float) -> "OccupancyGrid":
    """Copy with obstacles dilated by `radius` (m) → C-space for a point robot."""
    from scipy.ndimage import binary_dilation
    g = self.copy()
    rc = int(math.ceil(radius / self.res))
    if rc > 0 and self.occ.any():
      yy, xx = np.ogrid[-rc:rc + 1, -rc:rc + 1]
      g.occ = binary_dilation(self.occ, structure=(xx * xx + yy * yy <= rc * rc))
    return g
