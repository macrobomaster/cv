import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ..common.geometry import rotz
from ..common.gimbal import GimbalBuffer
from ...autoaim.common import (CANONICAL_CAMERA_MATRIX, CANONICAL_DIST_COEFFS,
                               IMG_H, IMG_W, plate_screw_points)
from ...slam.common import R_CAM, T_CAM   # calibrated yaw-only camera↔yaw-stage mount (z-up, shared)

# loaded camera calibration for warp
def _load_real_calib():
  path = Path(__file__).parent.parent.parent.parent / "weights" / "camera_calib.npz"
  if path.exists():
    data = np.load(path)
    logger.info(f"loaded real camera calibration from {path}")
    calib_w = int(data["image_size"][0]) if "image_size" in data.files else 512
    return data["mtx"].astype(np.float32), data["dist"].astype(np.float32), calib_w
  logger.warning("no weights/camera_calib.npz!")
  return CANONICAL_CAMERA_MATRIX.copy(), CANONICAL_DIST_COEFFS.copy(), IMG_W

REAL_CAMERA_MATRIX, REAL_DIST_COEFFS, REAL_CALIB_W = _load_real_calib()
# PnP object points are the screw-hole rectangle, switched by armor number (small vs large/hero) via
# plate_screw_points() — same geometry syndata trains on. Cached per number (only 2 distinct).
_PLATE_POINTS_CACHE: dict = {}
def plate_points(number:int) -> np.ndarray:
  if number not in _PLATE_POINTS_CACHE:
    _PLATE_POINTS_CACHE[number] = plate_screw_points(number)
  return _PLATE_POINTS_CACHE[number]

# --- plated tuning constants ---

T_LOST = 0.300                  # s of consecutive invalid → LOST + retarget on next valid
T_HANDOFF = 0.150               # s — withhold class after a NEW-target retarget (state not yet settled)

# Constant-velocity plate tracker process noise (per second)
Q_POS = (0.05) ** 2             # m^2/s, plate-position random walk
Q_VEL = (1.0) ** 2              # (m/s)^2/s, plate-velocity random walk; (1)²≈real chassis accel, raise
                                # toward (2) for faster chase. Sits between chassis-maneuver and spin
                                # frequency so a fast orbit is AVERAGED (not chased) — measure it on a spinner.

# Initial covariance at bootstrap
INIT_C = (0.30) ** 2
INIT_V = (1.0) ** 2
P_CAP = 1e3                     # diagonal covariance ceiling (numerical safety)

# Measurement noise — ANISOTROPIC in the CAMERA frame. PnP's depth (camera +z) is the sloppy axis (the
# measured ~0.1 m scatter is almost all depth); the corner centroid pins the LATERAL/bearing tightly.
# cov is camera-frame here; run() rotates it to gimbal-inertial (depth → range, lateral → bearing/height).
MEAS_NOISE_LAT = (0.02) ** 2    # m^2, lateral (camera x,y) floor — bearing is well-localized
MEAS_NOISE_DEPTH = (0.10) ** 2  # m^2, depth (camera z) floor — the measured raw-PnP scatter (~0.1 m)
_MEAS_FLOOR = np.diag([MEAS_NOISE_LAT, MEAS_NOISE_LAT, MEAS_NOISE_DEPTH])
MEAS_NOISE_STALE_MULT = 100.0
H_EMA = 0.3                     # plate-height smoothing

# Data association / retarget
ASSOC_POS_MAHAL = 30.0          # position gate for "same track" (else jump: handoff snap or new-robot retarget)
ASSOC_POS_FLOOR = (0.06) ** 2   # m^2 — floor on the association innov-cov so a CONVERGED filter (tiny
                                # P+R) can't hair-trigger the gate on normal PnP noise (~5-15cm).
                                # With MAHAL=30 this caps the gate at a ~0.33 m "same-plate" radius.
RETARGET_JUMP_FRAMES = 2        # consecutive gate failures before accepting the jump

# gimbal-inertial is z-up RH (shared with slam): +x forward, +y left, +z up; yaw turns about +z (rotz).
# The autoaim camera is on the gimbal YAW stage only (it does NOT pitch), so camera→gimbal-inertial is
# yaw-only: pos_gi = rotz(yaw_gi) @ (R_CAM @ pos_cam + T_CAM), reusing slam's calibrated yaw-only mount.

# --- Plate-centric constant-velocity tracker ------------------------------------------------------
# State x = [px, py, vx, vy]: the visible plate's position + velocity in the gimbal-inertial horizontal
# plane (z-up RH: x-forward, y-left). Height z is tracked separately (EMA) — it's measured directly and
# doesn't couple into the CV model. The plate position is directly, well observed by PnP, so we filter
# it instead of a derived spin center. Linear F, linear H → exact Kalman. A spinning robot's plate orbits:
# with Q tuned below the spin frequency the filter AVERAGES the orbit toward the center; decisiond
# suppresses the (tangential) velocity lead while the plate oscillates so the aim doesn't fling.

IDX_PX, IDX_PY, IDX_VX, IDX_VY = range(4)

class PlateTracker:
  def __init__(self):
    self.t = 0.0
    self.x = np.zeros(4)
    self.P = np.eye(4)
    self.initialized = False
    self.h: Optional[float] = None   # smoothed plate height (world-z)

  def bootstrap(self, p_xy:np.ndarray, h:float, t:float):
    self.x = np.array([p_xy[0], p_xy[1], 0.0, 0.0])   # assume stationary on first sight
    self.P = np.diag([INIT_C, INIT_C, INIT_V, INIT_V])
    self.h = h
    self.t = t
    self.initialized = True

  def _F_Q(self, dt:float):
    F = np.eye(4)
    F[IDX_PX, IDX_VX] = dt
    F[IDX_PY, IDX_VY] = dt
    Q = np.diag([Q_POS*dt, Q_POS*dt, Q_VEL*dt, Q_VEL*dt])
    return F, Q

  def predict_to(self, t:float):
    dt = t - self.t
    if dt <= 0: return
    F, Q = self._F_Q(dt)
    self.x = F @ self.x
    self.P = F @ self.P @ F.T + Q
    np.fill_diagonal(self.P, np.minimum(np.diag(self.P), P_CAP))
    self.t = t

  def pos_mahal(self, p_xy:np.ndarray, R_pos:np.ndarray) -> float:
    """Pre-update position Mahalanobis distance — the jump gate. Floored innov-cov so a converged filter
    (tiny P+R) can't hair-trigger a retarget on ordinary PnP noise (~5-15cm)."""
    innov = p_xy - self.x[:2]
    S = self.P[:2, :2] + R_pos + ASSOC_POS_FLOOR * np.eye(2)
    return float(innov @ np.linalg.solve(S, innov))

  def update(self, p_xy:np.ndarray, h:float, R_pos:np.ndarray):
    H = np.zeros((2, 4))
    H[0, IDX_PX] = 1.0
    H[1, IDX_PY] = 1.0
    innov = p_xy - self.x[:2]
    S = self.P[:2, :2] + R_pos
    K = self.P @ H.T @ np.linalg.inv(S)
    self.x = self.x + K @ innov
    self.P = self.P - K @ S @ K.T
    self.P = 0.5 * (self.P + self.P.T)
    np.fill_diagonal(self.P, np.minimum(np.diag(self.P), P_CAP))
    self.h = h if self.h is None else (1 - H_EMA) * self.h + H_EMA * h

  def plate_state_at(self, t:float):
    """Forward-predict (pos_gi, vel_gi, cov_pos) to t without mutating the filter."""
    dt = max(0.0, t - self.t)
    F, Q = self._F_Q(dt)
    x = F @ self.x
    P = F @ self.P @ F.T + Q
    h = self.h if self.h is not None else 0.0
    pos = np.array([x[IDX_PX], x[IDX_PY], h])                # z-up: (x, y, height=z)
    vel = np.array([x[IDX_VX], x[IDX_VY], 0.0])              # horizontal only
    cov = np.diag([P[IDX_PX, IDX_PX], P[IDX_PY, IDX_PY], (0.02) ** 2])
    cov[0, 1] = cov[1, 0] = P[IDX_PX, IDX_PY]
    return pos, vel, cov

# --- Top-level plated state machine ---------------------------------------------------------------

class PlatedState:
  def __init__(self):
    self.tracker = PlateTracker()
    self.target_id = 0
    self.last_meta: Optional[tuple] = None
    self.last_valid_t: Optional[float] = None
    self.handoff_until_t = 0.0
    self.consecutive_jumps = 0
    self.last_meas: Optional[np.ndarray] = None   # raw PnP plate position (pre-filter), for viz
    self._retarget_n = 0                          # NEW-target retargets since last diag log
    self._retarget_log_t = 0.0

  def _retarget(self, p_xy, h, t, reason, new_target=True):
    self.tracker.bootstrap(p_xy, h, t)
    self.consecutive_jumps = 0
    if not new_target:
      return   # same-target snap (handoff / re-appear): keep target_id and stay TRACKING
    self.target_id += 1
    self.handoff_until_t = t + T_HANDOFF
    # Throttled log so a flickering cause shows its RATE + reason, not a per-frame flood.
    self._retarget_n += 1
    if self._retarget_log_t == 0.0: self._retarget_log_t = t
    if t - self._retarget_log_t > 1.0:
      logger.info(f"plated retarget → #{self.target_id} ({reason}); {self._retarget_n} in {t-self._retarget_log_t:.1f}s")
      self._retarget_log_t = t
      self._retarget_n = 0

  def push_measurement(self, t_capture, pos_gi, meta, R_pos_xy):
    self.last_meas = pos_gi
    p_xy = np.array([pos_gi[0], pos_gi[1]])     # z-up: horizontal plane is x-y
    h = float(pos_gi[2])                         # z-up: height is the world-z component

    if not self.tracker.initialized:
      self._retarget(p_xy, h, t_capture, "init")
      self.last_meta = meta
      self.last_valid_t = t_capture
      return

    if self.last_valid_t is not None and (t_capture - self.last_valid_t) > T_LOST:
      self._retarget(p_xy, h, t_capture, f"lost {t_capture - self.last_valid_t:.2f}s")
      self.last_meta = meta
      self.last_valid_t = t_capture
      return

    self.tracker.predict_to(t_capture)
    mahal = self.tracker.pos_mahal(p_xy, R_pos_xy)
    meta_changed = self.last_meta is not None and meta != self.last_meta

    # Position jump. A changed color/number ⇒ a genuinely NEW robot ⇒ retarget (bump target_id, which
    # resets decisiond's shoot gate). A SAME-meta jump is almost always the same robot handing off to its
    # next plate — SNAP to the new plate but KEEP the track identity, so decisiond keeps firing through
    # handoffs instead of resetting its in-tolerance count every quarter turn. (A same-color second robot
    # is indistinguishable from one plate; treated as a handoff — a rare, benign conflation.)
    if mahal > ASSOC_POS_MAHAL:
      self.consecutive_jumps += 1
      if meta_changed:
        self._retarget(p_xy, h, t_capture, f"meta+jump mahal={mahal:.0f}", new_target=True)
        self.last_meta = meta
        self.last_valid_t = t_capture
      elif self.consecutive_jumps >= RETARGET_JUMP_FRAMES:
        self._retarget(p_xy, h, t_capture, f"handoff mahal={mahal:.0f}", new_target=False)
        self.last_meta = meta
        self.last_valid_t = t_capture
      return
    self.consecutive_jumps = 0
    self.tracker.update(p_xy, h, R_pos_xy)
    self.last_meta = meta                          # adopt current label (handles benign meta flicker)
    self.last_valid_t = t_capture

  def push_invalid(self, t:float):
    pass  # LOST is derived from last_valid_t in publish()

  def publish(self, t_now:float) -> Optional[dict]:
    if not self.tracker.initialized:
      return None
    if self.last_valid_t is None or (t_now - self.last_valid_t) > T_LOST:
      cls = "LOST"
    elif t_now < self.handoff_until_t:
      cls = "UNKNOWN"
    else:
      cls = "TRACKING"
    pos, vel, cov = self.tracker.plate_state_at(t_now)
    return {
      "t_state": t_now,
      "pos_gi": pos.tolist(),
      "vel_gi": vel.tolist(),
      "cov_pos": cov.tolist(),
      "pos_meas": self.last_meas.tolist() if self.last_meas is not None else None,
      "class": cls,          # TRACKING | UNKNOWN | LOST
      "target_id": self.target_id,
    }

# --- PnP --------------------------------------------------------------------------------------------

def _pnp(corners_2d:np.ndarray, obj_points:np.ndarray) -> Optional[np.ndarray]:
  ok, _, tvec = cv2.solvePnP(obj_points, corners_2d, CANONICAL_CAMERA_MATRIX,
                             CANONICAL_DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE)
  return tvec.flatten() if ok else None

def _pnp_pose(corners_2d:np.ndarray, obj_points:np.ndarray):
  ok, rvec, tvec = cv2.solvePnP(obj_points, corners_2d, CANONICAL_CAMERA_MATRIX,
                                CANONICAL_DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE)
  if not ok: return None, None
  return tvec.flatten(), rvec

_SCALE_PX = np.array([IMG_W, IMG_H], dtype=np.float32)

def _pnp_pos_cov(corners:list, c_lo:Optional[list], c_hi:Optional[list], number:int):
  """(pos_cam, normal_cam, cov_cam). Position + plate outward normal from the mean corners; cov by
  propagating the per-corner DFL quantiles through PnP as DETERMINISTIC sigma points (each coord to
  its q_lo/q_hi, PnP each, accumulate ¼Σ(y_hi−y_lo)(y_hi−y_lo)ᵀ). Deterministic → stable R; base floor
  added. cov falls back to the base floor if quantiles are missing."""
  obj = plate_points(number)
  mean = np.array(corners, dtype=np.float32).reshape(4, 2) * _SCALE_PX
  pos, rvec = _pnp_pose(mean, obj)
  if pos is None:
    return None, None, _MEAS_FLOOR
  normal = cv2.Rodrigues(rvec)[0][:, 2]              # plate model +z = outward normal, camera frame
  if normal[2] > 0: normal = -normal                # face the camera (optical axis is +z)
  if c_lo is None or c_hi is None:
    return pos, normal, _MEAS_FLOOR
  lo = np.array(c_lo, dtype=np.float32).reshape(4, 2) * _SCALE_PX
  hi = np.array(c_hi, dtype=np.float32).reshape(4, 2) * _SCALE_PX
  cov = np.zeros((3, 3))
  for r in range(4):
    for c in range(2):
      pl = mean.copy(); pl[r, c] = lo[r, c]
      ph = mean.copy(); ph[r, c] = hi[r, c]
      yl, yh = _pnp(pl, obj), _pnp(ph, obj)
      if yl is None or yh is None: continue
      d = yh - yl
      cov += 0.25 * np.outer(d, d)
  return pos, normal, cov + _MEAS_FLOOR

def run():
  pub = messaging.Pub(["plate"])
  autoaim_sub = messaging.Sub(["autoaim"])
  # gimbal_state is high-rate (≥200Hz). Non-conflated + drained each tick so we keep every sample
  # for interpolation at t_capture.
  gimbal_sub = messaging.Sub(["gimbal_state"], conflate=False)

  gimbal_buf = GimbalBuffer()
  state = PlatedState()
  warned_no_gimbal = False

  while True:
    autoaim_sub.update(timeout=10)
    for msg in gimbal_sub.drain("gimbal_state"):
      gimbal_buf.push(msg)

    if not autoaim_sub.updated["autoaim"]:
      continue
    autoaim = autoaim_sub["autoaim"]
    if autoaim is None: continue

    t_capture = autoaim["t_capture"]
    meta = (autoaim["color"], autoaim["number"])

    if not autoaim["valid"]:
      state.push_invalid(t_capture)
      out = state.publish(time.monotonic())
      if out is not None: pub.send("plate", out)
      continue

    pos_cam, _, cov_cam = _pnp_pos_cov(autoaim["corners"], autoaim.get("corner_lo"),
                                       autoaim.get("corner_hi"), autoaim["number"])
    if pos_cam is None:
      state.push_invalid(t_capture)
      out = state.publish(time.monotonic())
      if out is not None: pub.send("plate", out)
      continue

    gp = gimbal_buf.interpolate(t_capture)
    if gp is None:
      yaw_gi_cap, stale = 0.0, True
      if not warned_no_gimbal:
        logger.warning("plated: no gimbal_state samples; running in degraded mode")
        warned_no_gimbal = True
    else:
      yaw_gi_cap, _, stale = gp                       # pitch unused — camera is yaw-only

    # camera→gimbal-inertial (z-up RH). The camera rides the YAW stage only, so this is YAW-ONLY (no
    # pitch). T_CAM (optical centre off the yaw axis) rotates with the stage: pos_gi = Rz·(R_CAM·pos_cam
    # + T_CAM), reusing slam's calibrated yaw-only mount. pos_gi is the target relative to the yaw axis.
    G = rotz(yaw_gi_cap)
    rot = G @ R_CAM
    pos_gi = G @ (R_CAM @ pos_cam + T_CAM)
    cov_gi = rot @ cov_cam @ rot.T * (MEAS_NOISE_STALE_MULT if stale else 1.0)
    R_pos_xy = cov_gi[np.ix_([0, 1], [0, 1])]

    state.push_measurement(t_capture, pos_gi, meta, R_pos_xy)

    out = state.publish(time.monotonic())
    if out is not None: pub.send("plate", out)
