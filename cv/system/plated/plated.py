import math
import time
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ...autoaim.common import (R_MOUNT, T_MOUNT, CANONICAL_CAMERA_MATRIX, CANONICAL_DIST_COEFFS,
                               IMG_H, IMG_W, plate_screw_points)

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

GIMBAL_DEQUE_MAX = 200          # ~1s at 200Hz; lets us tolerate large t_capture lag
GIMBAL_STALE_GAP = 0.030        # s — if no bracket sample within this, flag stale

T_LOST = 0.300                  # s of consecutive invalid → LOST + retarget on next valid
T_HANDOFF = 0.150               # s — withhold class after retarget (state not yet settled)

# Robot geometry priors
R_DEFAULT = 0.15                # m, plate radius from spin axis (bootstrap; refined when rotating)
R_RANGE = (0.04, 0.40)          # m, sanity clamp on the estimated radius
SPIN_FACING_HALF = math.radians(50)  # half-width of a plate's facing window (for decisiond)

# Robot-body EKF process noise (per second)
Q_CENTER = (0.05) ** 2          # m^2/s, center random-walk
Q_VEL = (3.0) ** 2              # (m/s)^2/s, center-velocity random walk (allows accel)
Q_THETA = (0.05) ** 2           # rad^2/s, heading random-walk
Q_OMEGA = (12.0) ** 2           # (rad/s)^2/s, heading-rate random walk (covers a spin spin-up in ~1s)
Q_R = (0.02) ** 2              # m^2/s, radius drifts slowly

# Initial covariance at bootstrap — center/heading/omega/radius are poorly known from one plate.
INIT_C = (0.30) ** 2
INIT_V = (1.0) ** 2
INIT_TH = (math.radians(40)) ** 2
INIT_W = (10.0) ** 2
INIT_R = (0.12) ** 2
P_CAP = 1e3                     # diagonal covariance ceiling (numerical safety in under-observed dirs)

# Measurement noise
MEAS_NOISE_BASE = (0.01) ** 2   # m^2, PnP position floor
MEAS_NOISE_STALE_MULT = 100.0
PSI_NOISE = (math.radians(25)) ** 2  # rad^2, plate facing-yaw is the NOISY part of PnP — trust loosely
H_EMA = 0.3                     # per-plate height smoothing

# Data association / retarget
ASSOC_PHASE_TOL = math.radians(40)   # facing-yaw must land within this of a plate slot to associate
ASSOC_POS_MAHAL = 30.0          # position gate for "same robot, this plate"
RETARGET_JUMP_FRAMES = 2        # consecutive gate failures before declaring a new robot

# Classification thresholds (read off the state)
OMEGA_SPIN = 1.5                # rad/s, enter SPIN
OMEGA_STATIC = 0.4              # rad/s, below this rotation is negligible
V_STATIC = 0.15                 # m/s, below this translation is negligible
SPIN_OMEGA_SNR = 1.0            # |omega| must exceed this * sigma_omega to trust it as spinning

# --- Math helpers ---

def R_yaw(angle:float) -> np.ndarray:
  c, s = math.cos(angle), math.sin(angle)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def R_pitch(angle:float) -> np.ndarray:
  c, s = math.cos(angle), math.sin(angle)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def wrap_pi(x:float) -> float:
  return (x + math.pi) % (2 * math.pi) - math.pi

# --- Gimbal sample interpolation ---

class GimbalBuffer:
  def __init__(self):
    self.samples: deque = deque(maxlen=GIMBAL_DEQUE_MAX)

  def push(self, msg:dict):
    self.samples.append((msg["t_stamp"], msg["yaw_gi"], msg["pitch_gi"]))

  def interpolate(self, t:float) -> Optional[tuple[float, float, bool]]:
    if not self.samples: return None
    if t <= self.samples[0][0]:
      ts, y, p = self.samples[0]
      return y, p, (ts - t) > GIMBAL_STALE_GAP
    if t >= self.samples[-1][0]:
      ts, y, p = self.samples[-1]
      return y, p, (t - ts) > GIMBAL_STALE_GAP
    # linear bracket — N is small (≤200), linear scan is fine
    prev_ts, prev_y, prev_p = self.samples[0]
    for ts, y, p in list(self.samples)[1:]:
      if prev_ts <= t <= ts:
        a = (t - prev_ts) / (ts - prev_ts)
        return prev_y + a * (y - prev_y), prev_p + a * (p - prev_p), False
      prev_ts, prev_y, prev_p = ts, y, p
    return None

# --- Robot-body EKF -------------------------------------------------------------------------------
# State x = [cx, cz, vx, vz, theta, omega, r]:
#   (cx,cz)  spin-axis position in the horizontal plane (gimbal-inertial x-forward, z-left), m
#   (vx,vz)  spin-axis velocity, m/s
#   theta    body heading — the outward bearing of plate 0, rad
#   omega    heading rate, rad/s  (≈0 static, large when spinning)
#   r        shared plate radius from the axis, m
# Per-plate heights h_k are tracked as side params (plate-y is measured directly). A robot is 4 plates
# at theta + k·90°; the visible plate's position stays well-observed even when center/theta/r don't,
# so aiming never degrades — the geometry only sharpens (and helps association/spin) once rotation is
# seen. STATIC / LINEAR / SPIN are just regions of (|v|, |omega|).

IDX_CX, IDX_CZ, IDX_VX, IDX_VZ, IDX_TH, IDX_W, IDX_R = range(7)

class RobotEKF:
  def __init__(self):
    self.t = 0.0
    self.x = np.zeros(7)
    self.P = np.eye(7)
    self.initialized = False
    self.h: list[Optional[float]] = [None, None, None, None]   # per-plate heights
    self.last_k = 0                                            # most-recently-seen plate slot

  def bootstrap(self, p_xz:np.ndarray, y:float, psi:float, t:float):
    # First sighting → call it plate 0: theta = its outward bearing, center one radius inward.
    cx = p_xz[0] - R_DEFAULT * math.cos(psi)
    cz = p_xz[1] - R_DEFAULT * math.sin(psi)
    self.x = np.array([cx, cz, 0.0, 0.0, psi, 0.0, R_DEFAULT])
    self.P = np.diag([INIT_C, INIT_C, INIT_V, INIT_V, INIT_TH, INIT_W, INIT_R])
    self.h = [None, None, None, None]
    self.h[0] = y
    self.last_k = 0
    self.t = t
    self.initialized = True

  def _F_Q(self, dt:float):
    F = np.eye(7)
    F[IDX_CX, IDX_VX] = dt
    F[IDX_CZ, IDX_VZ] = dt
    F[IDX_TH, IDX_W] = dt
    Q = np.diag([Q_CENTER*dt, Q_CENTER*dt, Q_VEL*dt, Q_VEL*dt, Q_THETA*dt, Q_OMEGA*dt, Q_R*dt])
    return F, Q

  def predict_to(self, t:float):
    dt = t - self.t
    if dt <= 0: return
    F, Q = self._F_Q(dt)
    self.x = F @ self.x
    self.x[IDX_TH] = wrap_pi(self.x[IDX_TH])
    self.P = F @ self.P @ F.T + Q
    np.fill_diagonal(self.P, np.minimum(np.diag(self.P), P_CAP))
    self.t = t

  def associate(self, psi_obs:float) -> tuple[int, float]:
    """Which plate slot does this facing-yaw belong to, and the phase residual."""
    k = int(round(wrap_pi(psi_obs - self.x[IDX_TH]) / (math.pi / 2))) % 4
    phase_res = abs(wrap_pi(psi_obs - self.x[IDX_TH] - k * (math.pi / 2)))
    return k, phase_res

  def _h_H(self, k:int):
    """Predicted (px, pz, psi) for plate k and the 3x7 Jacobian."""
    cx, cz, _, _, theta, _, r = self.x
    phi = theta + k * (math.pi / 2)
    c, s = math.cos(phi), math.sin(phi)
    z_pred = np.array([cx + r * c, cz + r * s, phi])
    H = np.zeros((3, 7))
    H[0, IDX_CX] = 1; H[0, IDX_TH] = -r * s; H[0, IDX_R] = c
    H[1, IDX_CZ] = 1; H[1, IDX_TH] = r * c;  H[1, IDX_R] = s
    H[2, IDX_TH] = 1
    return z_pred, H

  def pos_mahal(self, k:int, p_xz:np.ndarray, R_pos:np.ndarray) -> float:
    z_pred, H = self._h_H(k)
    innov = p_xz - z_pred[:2]
    S = H[:2] @ self.P @ H[:2].T + R_pos
    return float(innov @ np.linalg.solve(S, innov))

  def update(self, k:int, p_xz:np.ndarray, y:float, psi_obs:float, R_pos:np.ndarray, R_psi:float):
    z_pred, H = self._h_H(k)
    innov = np.array([p_xz[0] - z_pred[0], p_xz[1] - z_pred[1], wrap_pi(psi_obs - z_pred[2])])
    R = np.zeros((3, 3)); R[:2, :2] = R_pos; R[2, 2] = R_psi
    S = H @ self.P @ H.T + R
    K = self.P @ H.T @ np.linalg.inv(S)
    self.x = self.x + K @ innov
    self.x[IDX_TH] = wrap_pi(self.x[IDX_TH])
    self.x[IDX_R] = min(R_RANGE[1], max(R_RANGE[0], self.x[IDX_R]))
    self.P = (np.eye(7) - K @ H) @ self.P
    np.fill_diagonal(self.P, np.minimum(np.diag(self.P), P_CAP))
    self.h[k] = y if self.h[k] is None else (1 - H_EMA) * self.h[k] + H_EMA * y
    self.last_k = k

  def n_seen(self) -> int:
    return sum(1 for hk in self.h if hk is not None)

  def _mean_h(self) -> float:
    seen = [hk for hk in self.h if hk is not None]
    return float(np.mean(seen)) if seen else 0.0

  def plate_state_at(self, k:int, t:float):
    """Forward-predict plate k's (pos_gi, vel_gi, cov_pos) to t without mutating the filter."""
    dt = max(0.0, t - self.t)
    F, Q = self._F_Q(dt)
    x = F @ self.x
    P = F @ self.P @ F.T + Q
    cx, cz, vx, vz, theta, omega, r = x
    phi = theta + k * (math.pi / 2)
    c, s = math.cos(phi), math.sin(phi)
    h = self.h[k] if self.h[k] is not None else self._mean_h()
    pos = np.array([cx + r * c, h, cz + r * s])
    vel = np.array([vx - r * omega * s, 0.0, vz + r * omega * c])
    Hp = np.zeros((2, 7))
    Hp[0, IDX_CX] = 1; Hp[0, IDX_TH] = -r * s; Hp[0, IDX_R] = c
    Hp[1, IDX_CZ] = 1; Hp[1, IDX_TH] = r * c;  Hp[1, IDX_R] = s
    cov_xz = Hp @ P @ Hp.T
    cov = np.diag([cov_xz[0, 0], (0.02) ** 2, cov_xz[1, 1]])  # height variance ~ const
    cov[0, 2] = cov[2, 0] = cov_xz[0, 1]
    return pos, vel, cov

  def classify(self) -> str:
    omega = self.x[IDX_W]
    sigma_w = math.sqrt(max(self.P[IDX_W, IDX_W], 0.0))
    speed = math.hypot(self.x[IDX_VX], self.x[IDX_VZ])
    if abs(omega) > OMEGA_SPIN and abs(omega) > SPIN_OMEGA_SNR * sigma_w and self.n_seen() >= 2:
      return "SPIN"
    if speed < V_STATIC and abs(omega) < OMEGA_STATIC:
      return "STATIC"
    return "LINEAR"

  def to_spin_dict(self) -> dict:
    cx, cz, vx, vz, theta, omega, r = self.x
    mean_h = self._mean_h()
    return {
      "c_0": [float(cx), mean_h, float(cz)],
      "v_c": [float(vx), float(vz)],
      "omega": float(omega),
      "theta_body_0": float(theta),
      "t_ref": float(self.t),
      "plates": [{"r": float(r), "h": float(self.h[k] if self.h[k] is not None else mean_h),
                  "known": self.h[k] is not None} for k in range(4)],
      "theta_facing": float(SPIN_FACING_HALF),
    }

# --- Top-level plated state machine ---------------------------------------------------------------

class PlatedState:
  def __init__(self):
    self.ekf = RobotEKF()
    self.target_id = 0
    self.last_meta: Optional[tuple] = None
    self.last_valid_t: Optional[float] = None
    self.handoff_until_t = 0.0
    self.consecutive_jumps = 0
    self.last_meas: Optional[np.ndarray] = None   # raw PnP plate position (pre-EKF), for viz
    self.last_psi: float = 0.0                    # raw measured plate facing-yaw, for viz/validation

  def _retarget(self, p_xz, y, psi, t):
    self.ekf.bootstrap(p_xz, y, psi, t)
    self.target_id += 1
    self.handoff_until_t = t + T_HANDOFF
    self.consecutive_jumps = 0

  def push_measurement(self, t_capture, pos_gi, psi, meta, R_pos_xz, R_psi):
    self.last_meas = pos_gi
    self.last_psi = wrap_pi(psi)
    p_xz = np.array([pos_gi[0], pos_gi[2]])
    y = float(pos_gi[1])

    if not self.ekf.initialized:
      self._retarget(p_xz, y, psi, t_capture)
      self.last_meta = meta
      self.last_valid_t = t_capture
      return

    # A different color/number is unambiguously a different robot.
    if self.last_meta is not None and meta != self.last_meta:
      self._retarget(p_xz, y, psi, t_capture)
      self.last_meta = meta
      self.last_valid_t = t_capture
      return

    if self.last_valid_t is not None and (t_capture - self.last_valid_t) > T_LOST:
      self._retarget(p_xz, y, psi, t_capture)
      self.last_meta = meta
      self.last_valid_t = t_capture
      return

    self.ekf.predict_to(t_capture)
    k, phase_res = self.ekf.associate(psi)
    mahal = self.ekf.pos_mahal(k, p_xz, R_pos_xz)
    # No consistent plate slot (bad facing-yaw AND off-position) → likely a new robot.
    if phase_res > ASSOC_PHASE_TOL and mahal > ASSOC_POS_MAHAL:
      self.consecutive_jumps += 1
      if self.consecutive_jumps >= RETARGET_JUMP_FRAMES:
        self._retarget(p_xz, y, psi, t_capture)
        self.last_meta = meta
        self.last_valid_t = t_capture
      return
    self.consecutive_jumps = 0
    self.ekf.update(k, p_xz, y, psi, R_pos_xz, R_psi)
    self.last_meta = meta
    self.last_valid_t = t_capture

  def push_invalid(self, t:float):
    pass  # LOST is derived from last_valid_t in publish()

  def publish(self, t_now:float) -> Optional[dict]:
    if not self.ekf.initialized:
      return None
    if self.last_valid_t is None or (t_now - self.last_valid_t) > T_LOST:
      cls = "LOST"
    elif t_now < self.handoff_until_t:
      cls = "UNKNOWN"
    else:
      cls = self.ekf.classify()
    pos, vel, cov = self.ekf.plate_state_at(self.ekf.last_k, t_now)
    return {
      "t_state": t_now,
      "pos_gi": pos.tolist(),
      "vel_gi": vel.tolist(),
      "cov_pos": cov.tolist(),
      "pos_meas": self.last_meas.tolist() if self.last_meas is not None else None,
      "psi_meas": float(self.last_psi),
      "visible_k": int(self.ekf.last_k),
      "class": cls,
      "spin": self.ekf.to_spin_dict(),
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
    return None, None, np.eye(3) * MEAS_NOISE_BASE
  normal = cv2.Rodrigues(rvec)[0][:, 2]              # plate model +z = outward normal, camera frame
  if normal[2] > 0: normal = -normal                # face the camera (optical axis is +z)
  if c_lo is None or c_hi is None:
    return pos, normal, np.eye(3) * MEAS_NOISE_BASE
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
  return pos, normal, cov + np.eye(3) * MEAS_NOISE_BASE

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

    pos_cam, normal_cam, R_cam = _pnp_pos_cov(autoaim["corners"], autoaim.get("corner_lo"),
                                              autoaim.get("corner_hi"), autoaim["number"])
    if pos_cam is None:
      state.push_invalid(t_capture)
      out = state.publish(time.monotonic())
      if out is not None: pub.send("plate", out)
      continue

    gp = gimbal_buf.interpolate(t_capture)
    if gp is None:
      yaw_gi_cap, pitch_gi_cap, stale = 0.0, 0.0, True
      if not warned_no_gimbal:
        logger.warning("plated: no gimbal_state samples; running in degraded mode")
        warned_no_gimbal = True
    else:
      yaw_gi_cap, pitch_gi_cap, stale = gp

    # camera→gimbal-inertial. T_MOUNT (camera optical center in the gimbal-end-effector frame) rotates
    # WITH the gimbal: pos_gi = G·(R_MOUNT·pos_cam + T_MOUNT). (No-op while T_MOUNT=0.)
    G = R_yaw(yaw_gi_cap) @ R_pitch(pitch_gi_cap)
    rot = G @ R_MOUNT
    pos_gi = G @ (R_MOUNT @ pos_cam + T_MOUNT)
    normal_gi = rot @ normal_cam
    psi = math.atan2(normal_gi[2], normal_gi[0])     # plate outward bearing in the horizontal plane
    cov_gi = rot @ R_cam @ rot.T * (MEAS_NOISE_STALE_MULT if stale else 1.0)
    R_pos_xz = cov_gi[np.ix_([0, 2], [0, 2])]
    R_psi = PSI_NOISE * (MEAS_NOISE_STALE_MULT if stale else 1.0)

    state.push_measurement(t_capture, pos_gi, psi, meta, R_pos_xz, R_psi)

    out = state.publish(time.monotonic())
    if out is not None: pub.send("plate", out)
