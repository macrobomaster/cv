"""Ground-plane inverse-perspective-mapping (IPM) + world occupancy accumulation.

The camera rides the gimbal YAW stage only (fixed pitch — see slam.common / plated),
so the camera→ground geometry is a CONSTANT up to the live world heading. GroundIPM
precomputes, once, the floor point each lower-band pixel maps to in the camera-
forward-aligned frame; project() then just rotates+translates those points by the
live (heading, robot position) into the world XY plane.

IPM is only valid for the GROUND-CONTACT pixel of an obstacle — pixels above the
contact line are the obstacle body and project as if lying on the floor (i.e. too
far). The caller must therefore hand project() a mask of contact pixels, not every
non-floor pixel.

Occupancy is the union of obstacle ground-contact cells over a short TTL: a forward
camera sees only a near wedge, so out-of-view cells must persist briefly then decay.
The grid matches nav's OccupancyGrid (FIELD_BOUNDS, res) so navd can OR it in.
"""
import math

import numpy as np

from ..slam.common import K_INV, R_CAM, T_CAM, CAM_HEIGHT
from ..nav.occupancy import OccupancyGrid

class GroundIPM:
  """Per-pixel ground projection for the camera's lower image band. Precomputes the
  floor point (camera-forward-aligned, z-up) each band pixel hits; the geometry is
  fixed (yaw-only camera), so only the live heading/position vary per frame."""
  def __init__(self, img_h:int, img_w:int, band_top:float=0.5, min_range:float=0.2, max_range:float=4.0):
    self.v0 = int(band_top * img_h)                       # only the lower band can see the floor
    us = np.arange(img_w, dtype=np.float64)
    vs = np.arange(self.v0, img_h, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)                          # (Hb, W)
    self.band_shape = uu.shape
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 0)   # (3, N) homogeneous pixels
    d_cam = K_INV.astype(np.float64) @ pix                # camera RDF rays
    d0 = R_CAM.astype(np.float64) @ d_cam                 # camera-forward-aligned, z-up
    dz = d0[2]
    down = dz < -1e-6                                     # ray must hit the floor (points down)
    # floor at z = -CAM_HEIGHT below the optical centre (T_CAM[z]=0): P = T_CAM + t·d0, P.z=-CAM_HEIGHT
    t = np.zeros_like(dz)
    t[down] = -float(CAM_HEIGHT) / dz[down]
    pxy = (T_CAM[:2].astype(np.float64)[:, None] + t[None, :] * d0[:2]).T   # (N, 2) ground xy @ yaw0
    rng = np.hypot(pxy[:, 0], pxy[:, 1])
    self.valid = down & (rng >= min_range) & (rng <= max_range)
    pxy[~self.valid] = 0.0
    self.pxy = pxy                                        # (N, 2)

  def project(self, contact_mask:np.ndarray, heading:float, p_xy) -> np.ndarray:
    """World-XY (M, 2) for the band's obstacle CONTACT pixels. `contact_mask` is a
    band-shaped (Hb, W) bool; `heading` is the camera forward azimuth in world;
    `p_xy` the robot world position. Rotates the precomputed yaw0 ground points into
    the world and offsets by the robot position (mirrors navd's psi0 rotation)."""
    sel = contact_mask.ravel() & self.valid
    if not sel.any():
      return np.empty((0, 2), np.float64)
    pts = self.pxy[sel]                                   # (M, 2) camera-forward-aligned
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(p_xy, np.float64)

class OccupancyAccumulator:
  """Persistent world occupancy: stamps obstacle cells with the time last seen and
  reports a bool grid of cells seen within TTL, so out-of-view / departed obstacles
  fade. Grid geometry matches nav's OccupancyGrid for a direct OR into navd."""
  def __init__(self, bounds, res:float=0.10, ttl:float=2.0):
    self.grid = OccupancyGrid(*bounds, res)
    self.ttl = ttl
    self.seen = np.full((self.grid.ny, self.grid.nx), -1e18, dtype=np.float64)   # last-seen monotonic time

  def stamp(self, pts_xy:np.ndarray, now:float):
    if len(pts_xy) == 0: return
    g = self.grid
    ix = np.floor((pts_xy[:, 0] - g.x0) / g.res).astype(np.int64)
    iy = np.floor((pts_xy[:, 1] - g.y0) / g.res).astype(np.int64)
    m = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
    self.seen[iy[m], ix[m]] = now

  def occupied(self, now:float) -> np.ndarray:
    return (now - self.seen) < self.ttl
