"""Camera → 2D occupancy daemon (classical ground-plane IPM).

The detection chain (autoaim→plated) only ever reports ONE enemy-colored plate, so
navd is blind to allies, extra enemies, and non-robot obstacles. occupancyd is
detection-independent: it segments floor-vs-obstacle in the camera's lower band,
projects each obstacle's ground-CONTACT pixel onto the known flat floor via IPM
(yaw-only fixed-pitch camera → constant homography, see cv/occupancy/ipm), and
accumulates the obstacle cells in a persistent decaying WORLD grid that navd ORs into
its planner.

The floor classifier here is the classical appearance test (Phase 1); a tiny tinygrad
seg net can later replace just `FloorClassifier` behind the same pipeline (Phase 2).

Subs:  camera_feed:  {ct, st, fid, slot}            (RGB frame, canonical pinhole, shm ring)
       slam_pose:    {t, p_w, q_wb, n_tags, ...}     (robot world pose; world valid once n_tags>0)
       gimbal_state: {t_stamp, yaw_gi, ...}          (~200Hz; interpolated to the frame ct)
Pubs:  cam_occupancy:{t, x0, y0, res, nx, ny, occ:<bool grid bytes>, stale}
"""
import os
# Cap native math-lib threads BEFORE cv2/numpy load: occupancyd does only light per-frame
# CPU work at a few Hz, so OpenBLAS/OpenMP defaulting to one thread PER CORE just
# oversubscribes — the idle worker threads busy-spin → spurious ~Ncore% CPU. One is plenty.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
DEBUG = getenv("DEBUG", 0)                       # >=1: publish a 2D contact overlay for visual_slam
OCC_TTL = 2.0                                   # s, out-of-view / departed obstacles fade (mirrors RobotObstacles.ttl)
BAND_TOP = 0.5                                  # fraction of image height; only the lower band sees the floor
IPM_MIN_RANGE = 0.2                             # m, ignore cells too close (own footprint) / too far (unreliable)
IPM_MAX_RANGE = 4.0
FLOOR_STRIP_FRAC = 0.12                         # bottom-centre strip sampled as the floor-colour prior
FLOOR_CHROMA_THRESH = 18.0                      # LAB a/b deviation → "obstacle" (coloured; shadow/glare-invariant). cv2 LAB units (0-255)
FLOOR_DARK_THRESH = 45.0                        # LAB L BELOW the floor prior → dark achromatic obstacle. Brighter (glare/specular) is NOT flagged
FLOOR_PRIOR_EMA = 0.1                           # per-frame smoothing of the floor prior (resists a transient obstacle in the sample strip)

def gimbal_heading(q_wb):
  """Horizontal unit forward vector of the gimbal in world from q_wb=[w,x,y,z], or
  None if it points ~straight up/down. (Same as navd's; the SLAM cam forward azimuth.)"""
  w, x, y, z = q_wb
  fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
  n = math.hypot(fx, fy)
  return (fx / n, fy / n) if n > 1e-6 else None

class FloorClassifier:
  """Floor-vs-obstacle on a uniform (grey) field mat. A band pixel is OBSTACLE if it is
  CHROMATIC — its LAB a/b deviate from the floor prior (catches coloured robot parts;
  invariant to shadow/glare since those mostly move L, not a/b) — OR markedly DARKER than
  the floor (catches black chassis/wheels). Brighter-than-floor (glare/specular) is NOT
  flagged. The floor prior is an EMA of the bottom-centre strip's median, updated only when
  the strip still looks like floor, so an obstacle parked in the strip can't poison it.
  Phase-2 swaps THIS class for a tinygrad seg net behind the same (band)->floor-mask API."""
  def __init__(self):
    self.prior = None                                  # (L, a, b) float, cv2 LAB units

  def __call__(self, band:np.ndarray) -> np.ndarray:
    hb, w = band.shape[:2]
    lab = cv2.cvtColor(band, cv2.COLOR_RGB2LAB).astype(np.float32)
    sh = max(1, int(FLOOR_STRIP_FRAC * hb))
    med = np.median(lab[hb - sh:, w // 4: 3 * w // 4].reshape(-1, 3), axis=0)
    if self.prior is None:
      self.prior = med
    elif np.hypot(med[1] - self.prior[1], med[2] - self.prior[2]) < FLOOR_CHROMA_THRESH \
        and abs(med[0] - self.prior[0]) < FLOOR_DARK_THRESH:   # strip still consistent with floor → adapt
      self.prior = (1 - FLOOR_PRIOR_EMA) * self.prior + FLOOR_PRIOR_EMA * med

    chroma = np.hypot(lab[..., 1] - self.prior[1], lab[..., 2] - self.prior[2])   # colour deviation
    darker = self.prior[0] - lab[..., 0]                                          # how much darker than floor
    obstacle = (chroma > FLOOR_CHROMA_THRESH) | (darker > FLOOR_DARK_THRESH)
    k = np.ones((5, 5), np.uint8)                       # open then close to denoise specks
    floor = cv2.morphologyEx((~obstacle).astype(np.uint8), cv2.MORPH_OPEN, k)
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
  # Mark alive IMMEDIATELY (before constructing sockets / FrameRing / GroundIPM) so a slow
  # startup under load can't miss the 10s watchdog and trigger a kill→respawn restart loop
  # (each respawn re-imports cv2/numpy → sustained high CPU with no output).
  kv_put("watchdog", "occupancyd", time.monotonic())
  logger.info("occupancyd: starting")
  t0 = time.monotonic()

  pub = messaging.Pub(["cam_occupancy", "cam_occupancy_debug"])
  sub = messaging.Sub(["camera_feed", "slam_pose"], poll="camera_feed")
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)
  gimbal_buf = GimbalBuffer()
  ring = FrameRing(CAMERA_RING, (common.IMG_H, common.IMG_W, 3))

  ipm = GroundIPM(common.IMG_H, common.IMG_W, band_top=BAND_TOP, min_range=IPM_MIN_RANGE, max_range=IPM_MAX_RANGE)
  acc = OccupancyAccumulator(common.FIELD_BOUNDS, res=0.10, ttl=OCC_TTL)
  floorcls = FloorClassifier()
  g = acc.grid
  logger.info(f"occupancyd: grid {g.nx}x{g.ny} @ {g.res}m, band v0={ipm.v0}, "
              f"CAM_HEIGHT={common.CAM_HEIGHT}m ({time.monotonic() - t0:.1f}s setup)")

  last_occ_t = last_wd = last_diag = 0.0
  n_cam = n_rgb = n_contact = 0                          # per-diag-window counters
  reason = "no camera_feed yet"                          # latest world-projection status

  while True:
    sub.update(timeout=200)
    now = time.monotonic()
    for m in gimbal_sub.drain("gimbal_state"): gimbal_buf.push(m)
    if now - last_wd > 1.0:
      kv_put("watchdog", "occupancyd", now); last_wd = now
    # Unconditional heartbeat at the TOP so it always reports, even when the loop is
    # continuing early (no frames, shm not attachable, gated on pose).
    if now - last_diag > 1.0:
      logger.info(f"occupancyd: cam_feed={n_cam}/s rgb_ok={n_rgb}/s contact_px={n_contact} "
                  f"pose_alive={sub.alive.get('slam_pose')} | world: {reason}")
      last_diag = now; n_cam = n_rgb = 0

    cam = sub["camera_feed"]
    if cam is None or not sub.updated["camera_feed"]: continue
    n_cam += 1
    if now - last_occ_t < 1.0 / OCCUPANCY_HZ: continue
    last_occ_t = now

    rgb = frame_view(ring, cam)
    if rgb is None:
      reason = "frame_view None — shm slot not attachable (occupancyd not on camerad's host? camerad down?)"; continue
    n_rgb += 1
    contact = contact_mask(floorcls(rgb[ipm.v0:]))       # (Hb, W) ground-contact pixels
    n_contact = int(contact.sum())

    # DEBUG: 2D overlay of the detected obstacle contact line, in FULL-image pixel coords —
    # visible in visual_slam's camera view INDEPENDENT of localization, so the floor
    # classifier can be verified before SLAM ever anchors.
    if DEBUG:
      cy, cx = np.nonzero(contact)
      pub.send("cam_occupancy_debug",
               {"t": float(cam["ct"]), "contact": np.stack([cx, cy + ipm.v0], 1).astype(np.int32).tolist()})

    # World projection needs a localized pose: the world origin is meaningless until the
    # first absolute tag fix (same gate navd uses). Until then we still emit the 2D overlay
    # above but can't place obstacles in the world — log WHY so it isn't silently dark.
    pose = sub["slam_pose"]
    fwd = gimbal_heading(pose["q_wb"]) if pose is not None else None
    gp_pose = gimbal_buf.interpolate(pose["t"]) if pose is not None else None
    gp_ct = gimbal_buf.interpolate(float(cam["ct"]))
    if pose is None or pose["n_tags"] == 0:
      reason = "no slam_pose" if pose is None else "n_tags=0 (SLAM not anchored to a tag yet)"
    elif fwd is None or gp_pose is None or gp_ct is None:
      reason = "no gimbal_state"
    else:
      # psi0 = chassis world heading (slew-invariant); add the gimbal yaw at the FRAME's
      # capture time → the camera's world-forward azimuth at ct (mirrors navd's look-at).
      cam_world_yaw = math.atan2(fwd[1], fwd[0]) - gp_pose[0] + gp_ct[0]
      p_xy = np.asarray(pose["p_w"], np.float64)[:2]
      pts = ipm.project(contact, cam_world_yaw, p_xy)
      inb = acc.stamp(pts, now)
      occ = acc.occupied(now)
      pub.send("cam_occupancy", {
        "t": float(cam["ct"]), "x0": g.x0, "y0": g.y0, "res": g.res, "nx": g.nx, "ny": g.ny,
        "occ": occ.tobytes(), "stale": bool(gp_ct[2])})
      c = pts.mean(0) if len(pts) else (float("nan"), float("nan"))
      reason = (f"yaw={math.degrees(cam_world_yaw):.0f}deg robot=({p_xy[0]:.1f},{p_xy[1]:.1f}) "
                f"pts~({c[0]:.1f},{c[1]:.1f}) inbounds={inb}/{len(pts)} → {int(occ.sum())} cells"
                + (" [stale gimbal]" if gp_ct[2] else ""))
