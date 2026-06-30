"""Camera → 2D occupancy daemon (classical ground-plane IPM).

The detection chain (autoaim→plated) only ever reports ONE enemy-colored plate, so
navd is blind to allies, extra enemies, and non-robot obstacles. occupancyd is
detection-independent: it segments floor-vs-obstacle in the camera's lower band,
projects each obstacle's ground-CONTACT pixel onto the known flat floor via IPM
(yaw-only fixed-pitch camera → constant homography, see cv/occupancy/ipm), and
accumulates the obstacle cells in a persistent decaying WORLD grid that navd ORs into
its planner.

The floor classifier here is the classical appearance test (Phase 1); a tiny tinygrad
seg net can later replace just `floor_mask()` behind the same pipeline (Phase 2).

Subs:  camera_feed:  {ct, st, fid, slot}            (RGB frame, canonical pinhole, shm ring)
       slam_pose:    {t, p_w, q_wb, n_tags, ...}     (robot world pose; world valid once n_tags>0)
       gimbal_state: {t_stamp, yaw_gi, ...}          (~200Hz; interpolated to the frame ct)
Pubs:  cam_occupancy:{t, x0, y0, res, nx, ny, occ:<bool grid bytes>, stale}
"""
import gc, math, time

import cv2
import numpy as np
from tinygrad.helpers import getenv

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_put
from ..common.frame_ring import FrameRing, frame_view, CAMERA_RING
from ..common.gimbal import GimbalBuffer
from ...slam import common
from ...occupancy.ipm import GroundIPM, OccupancyAccumulator

OPENCV_THREADS = getenv("OPENCV_THREADS", 1)   # don't oversubscribe; camerad is RT-pinned
OCCUPANCY_HZ = getenv("OCCUPANCY_HZ", 8)        # cap; navd replans at 5Hz so this is plenty
OCC_TTL = 2.0                                   # s, out-of-view / departed obstacles fade (mirrors RobotObstacles.ttl)
BAND_TOP = 0.5                                  # fraction of image height; only the lower band sees the floor
IPM_MIN_RANGE = 0.2                             # m, ignore cells too close (own footprint) / too far (unreliable)
IPM_MAX_RANGE = 4.0
FLOOR_THRESH = 28.0                             # Lab distance from the floor prior above which a pixel is "obstacle"
FLOOR_STRIP_FRAC = 0.12                         # bottom-centre strip sampled as the floor-colour prior

def gimbal_heading(q_wb):
  """Horizontal unit forward vector of the gimbal in world from q_wb=[w,x,y,z], or
  None if it points ~straight up/down. (Same as navd's; the SLAM cam forward azimuth.)"""
  w, x, y, z = q_wb
  fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
  n = math.hypot(fx, fy)
  return (fx / n, fy / n) if n > 1e-6 else None

def floor_mask(band:np.ndarray) -> np.ndarray:
  """Classical floor/not-floor on the lower image band (RGB). Samples the bottom-centre
  strip as the floor-colour prior (the field mat is controlled) and flags pixels whose
  Lab colour deviates from it as obstacle. Phase-2 swaps THIS for a tinygrad seg net."""
  hb, w = band.shape[:2]
  lab = cv2.cvtColor(band, cv2.COLOR_RGB2LAB).astype(np.float32)
  sh = max(1, int(FLOOR_STRIP_FRAC * hb))
  strip = lab[hb - sh:, w // 4: 3 * w // 4].reshape(-1, 3)
  prior = np.median(strip, axis=0)
  dist = np.linalg.norm(lab - prior, axis=2)         # (hb, w)
  floor = dist < FLOOR_THRESH
  k = np.ones((5, 5), np.uint8)                       # open then close to denoise specks
  floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_OPEN, k)
  floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, k).astype(bool)
  return floor

def contact_mask(floor:np.ndarray) -> np.ndarray:
  """Per-column ground-CONTACT pixel: scanning up from the band bottom, the first
  obstacle row above the free floor run. Only the contact pixel is valid for IPM —
  body pixels above it would project as if lying further out on the floor."""
  obs = ~floor
  flip = obs[::-1]                                    # row 0 = image bottom
  has = flip.any(0)                                   # columns with any obstacle
  first = flip.argmax(0)                              # first obstacle scanning up from the bottom
  out = np.zeros_like(obs)
  cols = np.nonzero(has)[0]
  out[obs.shape[0] - 1 - first[cols], cols] = True
  return out

def run():
  gc.disable()
  cv2.setNumThreads(OPENCV_THREADS)
  pub = messaging.Pub(["cam_occupancy"])
  sub = messaging.Sub(["camera_feed", "slam_pose"], poll="camera_feed")
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()
  ring = FrameRing(CAMERA_RING, (common.IMG_H, common.IMG_W, 3))

  ipm = GroundIPM(common.IMG_H, common.IMG_W, band_top=BAND_TOP, min_range=IPM_MIN_RANGE, max_range=IPM_MAX_RANGE)
  acc = OccupancyAccumulator(common.FIELD_BOUNDS, res=0.10, ttl=OCC_TTL)
  g = acc.grid
  logger.info(f"occupancyd: grid {g.nx}x{g.ny} @ {g.res}m, band v0={ipm.v0}, CAM_HEIGHT={common.CAM_HEIGHT}m")

  last_occ_t = last_wd = last_diag = 0.0
  kv_put("watchdog", "occupancyd", time.monotonic())

  while True:
    sub.update(timeout=200)
    now = time.monotonic()
    for m in gimbal_sub.drain("gimbal_state"): gimbal_buf.push(m)
    if now - last_wd > 1.0:
      kv_put("watchdog", "occupancyd", now); last_wd = now

    cam = sub["camera_feed"]
    if cam is None or not sub.updated["camera_feed"]: continue
    if now - last_occ_t < 1.0 / OCCUPANCY_HZ: continue
    last_occ_t = now

    # World projection needs the robot pose; the world origin is meaningless until the
    # first absolute tag fix (same gate navd uses).
    pose = sub["slam_pose"]
    if pose is None or pose["n_tags"] == 0: continue
    fwd = gimbal_heading(pose["q_wb"])
    gp_pose = gimbal_buf.interpolate(pose["t"])
    gp_ct = gimbal_buf.interpolate(float(cam["ct"]))
    if fwd is None or gp_pose is None or gp_ct is None: continue
    # psi0 = chassis world heading (slew-invariant); add the gimbal yaw at the FRAME's
    # capture time → the camera's world-forward azimuth at ct (mirrors navd's look-at).
    psi0 = math.atan2(fwd[1], fwd[0]) - gp_pose[0]
    cam_world_yaw = psi0 + gp_ct[0]

    rgb = frame_view(ring, cam)
    if rgb is None: continue
    band = rgb[ipm.v0:]
    contact = contact_mask(floor_mask(band))
    pts = ipm.project(contact, cam_world_yaw, np.asarray(pose["p_w"], np.float64)[:2])
    acc.stamp(pts, now)

    occ = acc.occupied(now)
    pub.send("cam_occupancy", {
      "t": float(cam["ct"]), "x0": g.x0, "y0": g.y0, "res": g.res, "nx": g.nx, "ny": g.ny,
      "occ": occ.tobytes(), "stale": bool(gp_ct[2])})

    if now - last_diag > 1.0:
      logger.info(f"occupancyd: {len(pts)} contact pts, {int(occ.sum())} occ cells"
                  + (" [stale gimbal]" if gp_ct[2] else "")); last_diag = now
