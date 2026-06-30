import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ..common.geometry import rotz, wrap_pi
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
MEAS_NOISE_BASE = (0.1) ** 2    # m^2, PnP position floor — set to the MEASURED raw-PnP scatter (~0.1 m).
                                # Was (0.01)² = 100× overconfident, so the UKF over-trusted noisy corners
                                # and over-fit them into v_c / spurious jumps. Retune if scatter changes.
MEAS_NOISE_STALE_MULT = 100.0
PSI_NOISE = (math.radians(25)) ** 2  # rad^2, plate facing-yaw is the NOISY part of PnP — trust loosely
PSI_FLIP_GATE = math.radians(90)     # |ψ innovation| above this ⇒ likely a PnP normal flip → drop ψ this frame
H_EMA = 0.3                     # per-plate height smoothing

# Data association / retarget
ASSOC_POS_MAHAL = 30.0          # min-over-slots position gate for "same robot, this plate" (else retarget)
ASSOC_POS_FLOOR = (0.06) ** 2   # m^2 — floor on the association innov-cov so a CONVERGED filter (tiny
                                # Pzz+R_pos) can't hair-trigger the gate on normal PnP noise (~5-15cm).
                                # With MAHAL=30 this caps the gate at a ~0.33 m "same-plate" radius.
RETARGET_JUMP_FRAMES = 2        # consecutive gate failures before declaring a new robot

# Classification thresholds (read off the state)
OMEGA_SPIN = 1.5                # rad/s, enter SPIN
OMEGA_STATIC = 0.4              # rad/s, below this rotation is negligible
V_STATIC = 0.15                 # m/s, below this translation is negligible
SPIN_OMEGA_SNR = 1.0            # |omega| must exceed this * sigma_omega to trust it as spinning

# --- Math helpers ---

# gimbal-inertial is z-up RH (shared with slam): +x forward, +y left, +z up; yaw turns about +z (rotz).
# The autoaim camera is on the gimbal YAW stage only (it does NOT pitch), so camera→gimbal-inertial is
# yaw-only: pos_gi = rotz(yaw_gi) @ (R_CAM @ pos_cam + T_CAM), reusing slam's calibrated yaw-only mount.

# --- UKF weights / matrix sqrt --------------------------------------------------------------------
# The robot-body process model is linear, so the predict stays an exact Kalman propagation; only the
# trig plate-measurement is nonlinear. We replace the EKF measurement Jacobian with an unscented update
# so curvature over large theta/r uncertainty (bootstrap/acquisition) is captured where the linearized
# H wasn't. alpha=1, beta=2, kappa=0 → lambda=0: sigma spread sqrt(n), every covariance weight >= 0
# (no indefinite P from the update), and it reduces EXACTLY to the linear KF when the measurement is
# linear — so on the position channel nothing regresses; only the trig coupling improves.
_UKF_N = 7
_UKF_ALPHA, _UKF_BETA, _UKF_KAPPA = 1.0, 2.0, 0.0
_UKF_LAMBDA = _UKF_ALPHA ** 2 * (_UKF_N + _UKF_KAPPA) - _UKF_N
_UKF_GAMMA = math.sqrt(_UKF_N + _UKF_LAMBDA)
_UKF_WM = np.full(2 * _UKF_N + 1, 1.0 / (2 * (_UKF_N + _UKF_LAMBDA)))
_UKF_WC = _UKF_WM.copy()
_UKF_WM[0] = _UKF_LAMBDA / (_UKF_N + _UKF_LAMBDA)
_UKF_WC[0] = _UKF_LAMBDA / (_UKF_N + _UKF_LAMBDA) + (1 - _UKF_ALPHA ** 2 + _UKF_BETA)

def _chol_psd(M:np.ndarray) -> np.ndarray:
  """Matrix sqrt (columns = sigma directions) via lower Cholesky; jitter-retry then eigen-clip if M
  drifts non-PD from update round-off."""
  M = 0.5 * (M + M.T)
  for jit in (0.0, 1e-9, 1e-6, 1e-3):
    try:
      return np.linalg.cholesky(M + jit * np.eye(M.shape[0]))
    except np.linalg.LinAlgError:
      continue
  w, V = np.linalg.eigh(M)
  return V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))

# --- Robot-body UKF -------------------------------------------------------------------------------
# Linear-predict / unscented-update filter (predict is an exact KF step; only the trig measurement is
# unscented — see the UKF-weights note above).
# State x = [cx, cy, vx, vy, theta, omega, r]:
#   (cx,cy)  spin-axis position in the horizontal plane (gimbal-inertial z-up: x-forward, y-left), m
#   (vx,vy)  spin-axis velocity, m/s
#   theta    body heading — the outward bearing of plate 0, rad
#   omega    heading rate, rad/s  (≈0 static, large when spinning)
#   r        shared plate radius from the axis, m
# Per-plate heights h_k are tracked as side params (plate height = world-z, measured directly). A robot is 4 plates
# at theta + k·90°; the visible plate's position stays well-observed even when center/theta/r don't,
# so aiming never degrades — the geometry only sharpens (and helps association/spin) once rotation is
# seen. STATIC / LINEAR / SPIN are just regions of (|v|, |omega|).

IDX_CX, IDX_CY, IDX_VX, IDX_VY, IDX_TH, IDX_W, IDX_R = range(7)

class RobotUKF:
  def __init__(self):
    self.t = 0.0
    self.x = np.zeros(7)
    self.P = np.eye(7)
    self.initialized = False
    self.h: list[Optional[float]] = [None, None, None, None]   # per-plate heights
    self.last_k = 0                                            # most-recently-seen plate slot

  def bootstrap(self, p_xy:np.ndarray, h:float, psi:float, t:float):
    # First sighting → call it plate 0: theta = its outward bearing, center one radius inward.
    cx = p_xy[0] - R_DEFAULT * math.cos(psi)
    cy = p_xy[1] - R_DEFAULT * math.sin(psi)
    self.x = np.array([cx, cy, 0.0, 0.0, psi, 0.0, R_DEFAULT])
    self.P = np.diag([INIT_C, INIT_C, INIT_V, INIT_V, INIT_TH, INIT_W, INIT_R])
    self.h = [None, None, None, None]
    self.h[0] = h
    self.last_k = 0
    self.t = t
    self.initialized = True

  def _F_Q(self, dt:float):
    F = np.eye(7)
    F[IDX_CX, IDX_VX] = dt
    F[IDX_CY, IDX_VY] = dt
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

  def associate(self, p_xy:np.ndarray, R_pos:np.ndarray) -> tuple[int, float]:
    """Position-based association: the 4 slots are distinct points (a square around the center), so the
    visible plate is the slot whose PREDICTED position is closest. Returns (k, that slot's Mahalanobis
    distance — the min also gates retargeting). Independent of ψ, so a PnP flip can't mis-assign a slot."""
    mahals = [self.pos_mahal(k, p_xy, R_pos) for k in range(4)]
    k = int(np.argmin(mahals))
    return k, mahals[k]

  def _unscented(self, x:np.ndarray, P:np.ndarray, k:int):
    """Sigma-point propagation of plate-k's measurement h(x) = [px, pz, psi]. Returns
    (z_hat[3], Pzz[3,3] WITHOUT measurement noise, Pxz[7,3]). psi (= theta + k·90°) is an angle, so it
    is meaned/differenced circularly; positions are linear."""
    A = _UKF_GAMMA * _chol_psd(P)
    X = np.vstack([x, x + A.T, x - A.T])                                   # (2n+1, 7) sigma points
    phi = X[:, IDX_TH] + k * (math.pi / 2)
    Z = np.stack([X[:, IDX_CX] + X[:, IDX_R] * np.cos(phi),
                  X[:, IDX_CY] + X[:, IDX_R] * np.sin(phi), phi], axis=1)  # (2n+1, 3)
    z_hat = np.array([_UKF_WM @ Z[:, 0], _UKF_WM @ Z[:, 1],
                      math.atan2(_UKF_WM @ np.sin(Z[:, 2]), _UKF_WM @ np.cos(Z[:, 2]))])
    dz = Z - z_hat; dz[:, 2] = (dz[:, 2] + math.pi) % (2 * math.pi) - math.pi
    dx = X - x;     dx[:, IDX_TH] = (dx[:, IDX_TH] + math.pi) % (2 * math.pi) - math.pi
    Pzz = (dz.T * _UKF_WC) @ dz
    Pxz = (dx.T * _UKF_WC) @ dz
    return z_hat, Pzz, Pxz

  def pos_mahal(self, k:int, p_xy:np.ndarray, R_pos:np.ndarray) -> float:
    z_hat, Pzz, _ = self._unscented(self.x, self.P, k)
    innov = p_xy - z_hat[:2]
    # Floor the innovation cov: a converged filter has tiny Pzz+R_pos, which would turn ordinary PnP
    # noise into a huge Mahalanobis distance and spuriously retarget a plate it's tracking fine.
    S = Pzz[:2, :2] + R_pos + ASSOC_POS_FLOOR * np.eye(2)
    return float(innov @ np.linalg.solve(S, innov))

  def update(self, k:int, p_xy:np.ndarray, h:float, psi_obs:float, R_pos:np.ndarray, R_psi:float):
    z_hat, Pzz, Pxz = self._unscented(self.x, self.P, k)
    innov_psi = wrap_pi(psi_obs - z_hat[2])
    if abs(innov_psi) > PSI_FLIP_GATE:
      # |ψ innovation| too large ⇒ likely a PnP normal flip. Drop the facing channel this frame and keep
      # the tight position update, so the flip can't corrupt theta (position association already chose k).
      S = Pzz[:2, :2] + R_pos
      K = Pxz[:, :2] @ np.linalg.inv(S)
      innov = p_xy - z_hat[:2]
    else:
      R = np.zeros((3, 3)); R[:2, :2] = R_pos; R[2, 2] = R_psi
      S = Pzz + R
      K = Pxz @ np.linalg.inv(S)
      innov = np.array([p_xy[0] - z_hat[0], p_xy[1] - z_hat[1], innov_psi])
    self.x = self.x + K @ innov
    self.x[IDX_TH] = wrap_pi(self.x[IDX_TH])
    self.x[IDX_R] = min(R_RANGE[1], max(R_RANGE[0], self.x[IDX_R]))
    self.P = self.P - K @ S @ K.T
    self.P = 0.5 * (self.P + self.P.T)
    np.fill_diagonal(self.P, np.minimum(np.diag(self.P), P_CAP))
    self.h[k] = h if self.h[k] is None else (1 - H_EMA) * self.h[k] + H_EMA * h
    self.last_k = k

  def n_seen(self) -> int:
    return sum(1 for hk in self.h if hk is not None)

  def _mean_h(self) -> float:
    seen = [hk for hk in self.h if hk is not None]
    return float(np.mean(seen)) if seen else 0.0

  def plate_state_at(self, k:int, t:float):
    """Forward-predict plate k's (pos_gi, vel_gi, cov_pos) to t without mutating the filter. Predict is
    exact-linear; pos/vel are the deterministic model at the predicted mean, and the position cov is the
    unscented map of the trig measurement (no measurement noise added)."""
    dt = max(0.0, t - self.t)
    F, Q = self._F_Q(dt)
    x = F @ self.x
    x[IDX_TH] = wrap_pi(x[IDX_TH])
    P = F @ self.P @ F.T + Q
    cx, cy, vx, vy, theta, omega, r = x
    phi = theta + k * (math.pi / 2)
    c, s = math.cos(phi), math.sin(phi)
    h = self.h[k] if self.h[k] is not None else self._mean_h()
    pos = np.array([cx + r * c, cy + r * s, h])                 # z-up: (x, y, height=z)
    vel = np.array([vx - r * omega * s, vy + r * omega * c, 0.0])
    _, Pzz, _ = self._unscented(x, P, k)
    cov = np.diag([Pzz[0, 0], Pzz[1, 1], (0.02) ** 2])          # horizontal x,y; height variance ~ const
    cov[0, 1] = cov[1, 0] = Pzz[0, 1]
    return pos, vel, cov

  def classify(self) -> str:
    omega = self.x[IDX_W]
    sigma_w = math.sqrt(max(self.P[IDX_W, IDX_W], 0.0))
    speed = math.hypot(self.x[IDX_VX], self.x[IDX_VY])
    if abs(omega) > OMEGA_SPIN and abs(omega) > SPIN_OMEGA_SNR * sigma_w and self.n_seen() >= 2:
      return "SPIN"
    if speed < V_STATIC and abs(omega) < OMEGA_STATIC:
      return "STATIC"
    return "LINEAR"

  def to_spin_dict(self) -> dict:
    cx, cy, vx, vy, theta, omega, r = self.x
    mean_h = self._mean_h()
    return {
      "c_0": [float(cx), float(cy), mean_h],          # z-up: (x, y, height=z)
      "v_c": [float(vx), float(vy)],                  # horizontal velocity (x, y)
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
    self.ukf = RobotUKF()
    self.target_id = 0
    self.last_meta: Optional[tuple] = None
    self.last_valid_t: Optional[float] = None
    self.handoff_until_t = 0.0
    self.consecutive_jumps = 0
    self.last_meas: Optional[np.ndarray] = None   # raw PnP plate position (pre-EKF), for viz
    self.last_psi: float = 0.0                    # raw measured plate facing-yaw, for viz/validation
    self._retarget_n = 0                          # retargets since last diag log
    self._retarget_log_t = 0.0

  def _retarget(self, p_xy, h, psi, t, reason):
    self.ukf.bootstrap(p_xy, h, psi, t)
    self.target_id += 1
    self.handoff_until_t = t + T_HANDOFF
    self.consecutive_jumps = 0
    # Each retarget forces 150ms of class=UNKNOWN, so frequent retargets read as "stuck UNKNOWN while
    # tracking". Throttled log so a flickering cause shows its RATE + reason, not a per-frame flood.
    self._retarget_n += 1
    if self._retarget_log_t == 0.0: self._retarget_log_t = t
    if t - self._retarget_log_t > 1.0:
      logger.info(f"plated retarget → #{self.target_id} ({reason}); {self._retarget_n} in {t-self._retarget_log_t:.1f}s")
      self._retarget_log_t = t; self._retarget_n = 0

  def push_measurement(self, t_capture, pos_gi, psi, meta, R_pos_xy, R_psi):
    self.last_meas = pos_gi
    self.last_psi = wrap_pi(psi)
    p_xy = np.array([pos_gi[0], pos_gi[1]])     # z-up: horizontal plane is x-y
    h = float(pos_gi[2])                         # z-up: height is the world-z component

    if not self.ukf.initialized:
      self._retarget(p_xy, h, psi, t_capture, "init")
      self.last_meta = meta
      self.last_valid_t = t_capture
      return

    if self.last_valid_t is not None and (t_capture - self.last_valid_t) > T_LOST:
      self._retarget(p_xy, h, psi, t_capture, f"lost {t_capture - self.last_valid_t:.2f}s")
      self.last_meta = meta
      self.last_valid_t = t_capture
      return

    self.ukf.predict_to(t_capture)
    k, mahal = self.ukf.associate(p_xy, R_pos_xy)
    meta_changed = self.last_meta is not None and meta != self.last_meta

    # A new robot appears at a DIFFERENT position. A changed color/number CORROBORATES that but isn't
    # sufficient alone — a trained model still flips a label on the odd frame, and a flip on a plate
    # that's still in the same place is just noise (keep the track, adopt the label). So retarget only on
    # a position jump: persistent (RETARGET_JUMP_FRAMES) by itself, or immediate if meta ALSO changed.
    if mahal > ASSOC_POS_MAHAL:
      self.consecutive_jumps += 1
      if meta_changed or self.consecutive_jumps >= RETARGET_JUMP_FRAMES:
        self._retarget(p_xy, h, psi, t_capture,
                       f"{'meta+' if meta_changed else ''}jump mahal={mahal:.0f}>{ASSOC_POS_MAHAL:.0f}")
        self.last_meta = meta
        self.last_valid_t = t_capture
      return
    self.consecutive_jumps = 0
    self.ukf.update(k, p_xy, h, psi, R_pos_xy, R_psi)
    self.last_meta = meta                          # adopt current label (handles benign meta flicker)
    self.last_valid_t = t_capture

  def push_invalid(self, t:float):
    pass  # LOST is derived from last_valid_t in publish()

  def publish(self, t_now:float) -> Optional[dict]:
    if not self.ukf.initialized:
      return None
    if self.last_valid_t is None or (t_now - self.last_valid_t) > T_LOST:
      cls = "LOST"
    elif t_now < self.handoff_until_t:
      cls = "UNKNOWN"
    else:
      cls = self.ukf.classify()
    pos, vel, cov = self.ukf.plate_state_at(self.ukf.last_k, t_now)
    return {
      "t_state": t_now,
      "pos_gi": pos.tolist(),
      "vel_gi": vel.tolist(),
      "cov_pos": cov.tolist(),
      "pos_meas": self.last_meas.tolist() if self.last_meas is not None else None,
      "psi_meas": float(self.last_psi),
      "visible_k": int(self.ukf.last_k),
      "class": cls,
      "spin": self.ukf.to_spin_dict(),
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

    pos_cam, normal_cam, cov_cam = _pnp_pos_cov(autoaim["corners"], autoaim.get("corner_lo"),
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
    normal_gi = rot @ normal_cam
    psi = math.atan2(normal_gi[1], normal_gi[0])     # plate outward bearing in the x-y horizontal plane
    cov_gi = rot @ cov_cam @ rot.T * (MEAS_NOISE_STALE_MULT if stale else 1.0)
    R_pos_xy = cov_gi[np.ix_([0, 1], [0, 1])]
    R_psi = PSI_NOISE * (MEAS_NOISE_STALE_MULT if stale else 1.0)

    state.push_measurement(t_capture, pos_gi, psi, meta, R_pos_xy, R_psi)

    out = state.publish(time.monotonic())
    if out is not None: pub.send("plate", out)
