import math
import time
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ...autoaim.common import R_MOUNT, T_MOUNT, CANONICAL_CAMERA_MATRIX, CANONICAL_DIST_COEFFS, IMG_H, IMG_W

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
PLATE_WIDTH, PLATE_HEIGHT = 0.095, 0.104
# Corner order matches autoaim's syndata keypoint order: TL, TR, BL, BR (image-space convention,
# y-down; 3D y-axis is flipped vs image so TL has -y).
PLATE_POINTS = np.array([
  [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # TL
  [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0], # TR
  [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], # BL
  [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0], # BR
], dtype=np.float32)

# --- plated tuning constants ---

GIMBAL_DEQUE_MAX = 200          # ~1s at 200Hz; lets us tolerate large t_capture lag
GIMBAL_STALE_GAP = 0.030        # s — if no bracket sample within this, flag stale

CLASSIFY_INTERVAL = 0.100       # s — classifier re-runs at most this fast
T_LOST = 0.300                  # s of consecutive invalid → LOST + retarget on next valid
T_HANDOFF = 0.150               # s — withhold class after retarget

D_RETARGET_MAHAL = 30.0         # Mahalanobis distance for "this is a different target" jump
RETARGET_JUMP_FRAMES = 2        # require N consecutive frames of large jump before retargeting

BURST_GAP_MERGE = 0.05          # s — short invalid runs don't terminate a burst
BURST_MIN_PTS = 5
BURST_MIN_ARC = math.radians(20)
BURST_MAX_KEEP = 10             # rolling buffer of recent bursts

SPIN_OMEGA_MIN = 1.5            # rad/s for enter-SPIN
SPIN_OMEGA_EXIT = 0.5           # rad/s for exit-SPIN
SPIN_R_RANGE = (0.04, 0.40)
SPIN_PHASE_TOL = math.radians(22.5)
SPIN_PLATE_EMA = 0.3
SPIN_FACING_HALF = math.radians(50)

STATIC_VAR_THRESH = (0.015) ** 2  # 15mm stdev across the window → STATIC

EKF_Q_POS = (0.05) ** 2         # m^2/s — position random-walk noise
EKF_Q_VEL = (3.0) ** 2          # (m/s)^2/s — velocity random-walk noise (allows plate-jumps)
EKF_INIT_COV_POS = (0.10) ** 2
EKF_INIT_COV_VEL = (1.0) ** 2

MEAS_NOISE_BASE = (0.01) ** 2
MEAS_NOISE_STALE_MULT = 100.0
# Measurement covariance is propagated from the model's per-corner DFL std through PnP by
# DETERMINISTIC sigma points: perturb each corner coord by ±its std, PnP each, and accumulate
# cov = ¼ Σ (y₊ − y₋)(y₊ − y₋)ᵀ  (= J Σ Jᵀ via central differences). Captures PnP's anisotropic,
# geometry-dependent uncertainty (depth ≫ lateral, blows up edge-on/far) that a scalar/isotropic R
# can't — and, being deterministic, doesn't inject frame-to-frame jitter into R the way random MC
# would. MEAS_NOISE_BASE is added as a floor.

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

# --- 6-state constant-velocity EKF on (pos_gi, vel_gi) ---

class PoseEKF:
  def __init__(self):
    self.t = 0.0
    self.x = np.zeros(6)
    self.P = np.eye(6)
    self.initialized = False

  def reset(self, pos:np.ndarray, t:float):
    self.t = t
    self.x = np.concatenate([pos, np.zeros(3)])
    self.P = np.diag([EKF_INIT_COV_POS]*3 + [EKF_INIT_COV_VEL]*3)
    self.initialized = True

  def _F_Q(self, dt:float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = EKF_Q_POS * dt * np.eye(3)
    Q[3:, 3:] = EKF_Q_VEL * dt * np.eye(3)
    return F, Q

  def predict_to(self, t:float):
    dt = t - self.t
    if dt <= 0: return
    F, Q = self._F_Q(dt)
    self.x = F @ self.x
    self.P = F @ self.P @ F.T + Q
    self.t = t

  def update(self, pos:np.ndarray, R_meas:np.ndarray):
    H = np.zeros((3, 6)); H[:3, :3] = np.eye(3)
    y = pos - H @ self.x
    S = H @ self.P @ H.T + R_meas
    K = self.P @ H.T @ np.linalg.inv(S)
    self.x = self.x + K @ y
    self.P = (np.eye(6) - K @ H) @ self.P

  def state_at(self, t:float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Forward-predict for output without mutating internal state.
    dt = max(0.0, t - self.t)
    F, Q = self._F_Q(dt)
    x_pred = F @ self.x
    P_pred = F @ self.P @ F.T + Q
    return x_pred[:3].copy(), x_pred[3:].copy(), P_pred[:3, :3].copy()

  def gating_mahal(self, pos:np.ndarray, R_meas:np.ndarray, t:float) -> float:
    dt = max(0.0, t - self.t)
    F, Q = self._F_Q(dt)
    x_pred = F @ self.x
    P_pred = F @ self.P @ F.T + Q
    H = np.zeros((3, 6)); H[:3, :3] = np.eye(3)
    innov = pos - H @ x_pred
    S = H @ P_pred @ H.T + R_meas
    return float(innov @ np.linalg.solve(S, innov))

# --- Burst circle fit ---

def fit_burst_circle(observations:list) -> Optional[dict]:
  if len(observations) < BURST_MIN_PTS: return None
  ts = np.array([o[0] for o in observations])
  pts = np.array([o[1] for o in observations])
  xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

  # Kåsa algebraic circle fit: minimize ||x² + z² - (2 cx x + 2 cz z + (r² - cx² - cz²))||.
  A = np.column_stack([2*xs, 2*zs, np.ones_like(xs)])
  b = xs**2 + zs**2
  try:
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
  except np.linalg.LinAlgError:
    return None
  cx, cz, w = sol
  r_sq = w + cx**2 + cz**2
  if r_sq <= 0: return None
  r = math.sqrt(r_sq)
  if not (SPIN_R_RANGE[0] <= r <= SPIN_R_RANGE[1]): return None

  thetas = np.unwrap(np.arctan2(zs - cz, xs - cx))
  arc = float(abs(thetas[-1] - thetas[0]))
  if arc < BURST_MIN_ARC: return None

  A2 = np.column_stack([ts, np.ones_like(ts)])
  sol2, *_ = np.linalg.lstsq(A2, thetas, rcond=None)
  omega, theta_0 = float(sol2[0]), float(sol2[1])

  return {
    "c": np.array([float(cx), float(np.median(ys)), float(cz)]),
    "r": float(r),
    "h": float(np.median(ys)),
    "omega": omega,
    "theta_0": theta_0,
    "t_ref": float(ts[0]),
    "theta_ref": float(thetas[0]),
    "arc": arc,
  }

# --- Burst detector ---

class BurstWindow:
  def __init__(self):
    self.current: list = []
    self.completed: deque = deque(maxlen=BURST_MAX_KEEP)
    self.last_valid_t: Optional[float] = None

  def push(self, t:float, valid:bool, pos_gi:Optional[np.ndarray]):
    if not valid:
      return
    if self.last_valid_t is not None and (t - self.last_valid_t) > BURST_GAP_MERGE:
      if len(self.current) >= BURST_MIN_PTS:
        self.completed.append(self.current)
      self.current = []
    self.current.append((t, pos_gi.copy()))
    self.last_valid_t = t

  def iter_for_fit(self):
    yield from self.completed
    if len(self.current) >= BURST_MIN_PTS:
      yield self.current

  def reset(self):
    self.current = []
    self.completed.clear()
    self.last_valid_t = None

# --- 4-plate spin model ---

class SpinModel:
  def __init__(self):
    self.c_0 = np.zeros(3)
    self.v_c = np.zeros(2)      # (vx, vz)
    self.omega = 0.0
    self.theta_body_0 = 0.0
    self.t_ref = 0.0
    self.plates = [{"r": 0.0, "h": 0.0, "known": False} for _ in range(4)]
    # For estimating v_c across bursts: (t_burst_start, c_burst_xz)
    self.burst_centers: deque = deque(maxlen=BURST_MAX_KEEP)
    self.fitted_burst_ids: set = set()

  def has_any_plate(self) -> bool:
    return any(p["known"] for p in self.plates)

  def n_known_plates(self) -> int:
    return sum(1 for p in self.plates if p["known"])

  def update_from_burst(self, fit:dict, burst_id:int) -> bool:
    if burst_id in self.fitted_burst_ids: return False
    self.fitted_burst_ids.add(burst_id)

    if not self.has_any_plate():
      self.c_0 = fit["c"].copy()
      self.omega = fit["omega"]
      self.t_ref = fit["t_ref"]
      self.theta_body_0 = fit["theta_ref"]
      self.plates[0] = {"r": fit["r"], "h": fit["h"], "known": True}
      self.burst_centers.append((fit["t_ref"], np.array([fit["c"][0], fit["c"][2]])))
      return True

    # Phase-match BEFORE mutating shared state.
    dt = fit["t_ref"] - self.t_ref
    theta_body_pred = self.omega * dt + self.theta_body_0
    delta = fit["theta_ref"] - theta_body_pred
    k = round(delta / (math.pi / 2)) % 4
    phase_err = abs(wrap_pi(delta - k * (math.pi / 2)))
    if phase_err > SPIN_PHASE_TOL:
      return False

    p = self.plates[k]
    if p["known"]:
      p["r"] = (1 - SPIN_PLATE_EMA) * p["r"] + SPIN_PLATE_EMA * fit["r"]
      p["h"] = (1 - SPIN_PLATE_EMA) * p["h"] + SPIN_PLATE_EMA * fit["h"]
    else:
      p["r"] = fit["r"]
      p["h"] = fit["h"]
      p["known"] = True

    self.omega = (1 - SPIN_PLATE_EMA) * self.omega + SPIN_PLATE_EMA * fit["omega"]
    # Re-anchor theta_body_0 so it's the body-frame phase at the new t_ref (= fit.t_ref, set
    # below by _refit_v_c). At the new anchor, plate k would be at fit.theta_ref, so plate 0 is
    # at fit.theta_ref - k·π/2.
    self.theta_body_0 = wrap_pi(fit["theta_ref"] - k * (math.pi / 2))
    self.burst_centers.append((fit["t_ref"], np.array([fit["c"][0], fit["c"][2]])))
    self._refit_v_c()
    return True

  def _refit_v_c(self):
    if len(self.burst_centers) < 2:
      self.v_c = np.zeros(2)
      return
    ts = np.array([t for t, _ in self.burst_centers])
    cs = np.array([c for _, c in self.burst_centers])
    A = np.column_stack([ts, np.ones_like(ts)])
    sol_x, *_ = np.linalg.lstsq(A, cs[:, 0], rcond=None)
    sol_z, *_ = np.linalg.lstsq(A, cs[:, 1], rcond=None)
    self.v_c = np.array([float(sol_x[0]), float(sol_z[0])])
    # Anchor c_0 at the most-recent burst.
    last_t, last_c = self.burst_centers[-1]
    self.c_0 = np.array([float(last_c[0]), self.c_0[1], float(last_c[1])])
    self.t_ref = float(last_t)

  def center_at(self, t:float) -> np.ndarray:
    dt = t - self.t_ref
    return np.array([
      self.c_0[0] + self.v_c[0] * dt,
      self.c_0[1],
      self.c_0[2] + self.v_c[1] * dt,
    ])

  def plate_position(self, k:int, t:float) -> Optional[np.ndarray]:
    p = self.plates[k]
    if not p["known"]:
      known = [pp for pp in self.plates if pp["known"]]
      if not known: return None
      r = float(np.mean([pp["r"] for pp in known]))
      h = float(np.mean([pp["h"] for pp in known]))
    else:
      r, h = p["r"], p["h"]
    dt = t - self.t_ref
    c = self.center_at(t)
    theta_k = self.omega * dt + self.theta_body_0 + k * (math.pi / 2)
    return np.array([c[0] + r * math.cos(theta_k), h, c[2] + r * math.sin(theta_k)])

  def to_dict(self) -> dict:
    return {
      "c_0": self.c_0.tolist(),
      "v_c": self.v_c.tolist(),
      "omega": float(self.omega),
      "theta_body_0": float(self.theta_body_0),
      "t_ref": float(self.t_ref),
      "plates": [{"r": float(p["r"]), "h": float(p["h"]), "known": bool(p["known"])} for p in self.plates],
      "theta_facing": float(SPIN_FACING_HALF),
    }

# --- Target classifier ---

class TargetClassifier:
  def __init__(self):
    self.window: deque = deque(maxlen=300)  # (t, valid, pos_gi or None)
    self.burst_window = BurstWindow()
    self.spin_model = SpinModel()
    self.state = "UNKNOWN"
    self.last_classify_t = 0.0
    self.last_valid_t: Optional[float] = None
    self.burst_id_counter = 0
    self.completed_seen = 0

  def push(self, t:float, valid:bool, pos_gi:Optional[np.ndarray]):
    self.window.append((t, valid, pos_gi.copy() if pos_gi is not None else None))
    if valid:
      self.last_valid_t = t
    self.burst_window.push(t, valid, pos_gi)

  def classify(self, t:float) -> str:
    if t - self.last_classify_t < CLASSIFY_INTERVAL:
      return self.state
    self.last_classify_t = t

    # LOST takes priority.
    if self.last_valid_t is None or (t - self.last_valid_t) > T_LOST:
      self.state = "LOST"
      return self.state

    # Fit any newly-completed bursts.
    while self.completed_seen < len(self.burst_window.completed):
      burst = self.burst_window.completed[self.completed_seen]
      self.completed_seen += 1
      fit = fit_burst_circle(burst)
      if fit is not None:
        self.spin_model.update_from_burst(fit, burst_id=self.burst_id_counter)
        self.burst_id_counter += 1

    valid_obs = [o for o in self.window if o[1]]
    if len(valid_obs) < 5:
      return self.state  # not enough data; hold previous

    pts = np.array([o[2] for o in valid_obs])

    # SPIN: at least 2 plates filled and |omega| above threshold.
    if self.spin_model.n_known_plates() >= 2 and abs(self.spin_model.omega) > SPIN_OMEGA_MIN:
      self.state = "SPIN"
      return self.state

    # Exit SPIN if we were there and conditions degrade.
    if self.state == "SPIN" and abs(self.spin_model.omega) < SPIN_OMEGA_EXIT:
      self.state = "LINEAR"

    var = float(np.var(pts, axis=0).sum())
    if var < STATIC_VAR_THRESH:
      self.state = "STATIC"
    else:
      self.state = "LINEAR"
    return self.state

  def reset(self):
    self.window.clear()
    self.burst_window.reset()
    self.spin_model = SpinModel()
    self.state = "UNKNOWN"
    self.last_classify_t = 0.0
    self.last_valid_t = None
    self.burst_id_counter = 0
    self.completed_seen = 0

# --- Top-level plated state machine ---

class PlatedState:
  def __init__(self):
    self.ekf = PoseEKF()
    self.classifier = TargetClassifier()
    self.target_id = 0
    self.last_meta: Optional[tuple] = None
    self.handoff_until_t = 0.0
    self.consecutive_jumps = 0
    self.last_meas: Optional[np.ndarray] = None   # most recent raw PnP measurement (pre-EKF), for viz

  def _retarget(self, pos_gi:np.ndarray, t:float):
    self.ekf.reset(pos_gi, t)
    self.classifier.reset()
    self.target_id += 1
    self.handoff_until_t = t + T_HANDOFF
    self.consecutive_jumps = 0

  def push_measurement(self, t_capture:float, pos_gi:np.ndarray, meta:tuple, R_meas:np.ndarray):
    self.last_meas = pos_gi   # raw measurement this frame (before any accept/reject/filter)
    if not self.ekf.initialized:
      self._retarget(pos_gi, t_capture)
      self.classifier.push(t_capture, True, pos_gi)
      self.last_meta = meta
      return

    # Metadata-driven retarget: identifies a different robot.
    if self.last_meta is not None and meta != self.last_meta:
      self._retarget(pos_gi, t_capture)
      self.classifier.push(t_capture, True, pos_gi)
      self.last_meta = meta
      return

    # Long-loss retarget.
    if self.classifier.last_valid_t is not None and (t_capture - self.classifier.last_valid_t) > T_LOST:
      self._retarget(pos_gi, t_capture)
      self.classifier.push(t_capture, True, pos_gi)
      self.last_meta = meta
      return

    # Mahalanobis jump retarget: require N consecutive frames before committing.
    mahal = self.ekf.gating_mahal(pos_gi, R_meas, t_capture)
    if mahal > D_RETARGET_MAHAL:
      self.consecutive_jumps += 1
      if self.consecutive_jumps >= RETARGET_JUMP_FRAMES:
        self._retarget(pos_gi, t_capture)
        self.classifier.push(t_capture, True, pos_gi)
        self.last_meta = meta
        return
      # Treat as outlier — drop the measurement but keep state.
      return
    self.consecutive_jumps = 0

    self.ekf.predict_to(t_capture)
    self.ekf.update(pos_gi, R_meas)
    self.classifier.push(t_capture, True, pos_gi)
    self.last_meta = meta

  def push_invalid(self, t:float):
    self.classifier.push(t, False, None)

  def publish(self, t_now:float) -> Optional[dict]:
    if not self.ekf.initialized:
      return None
    pos, vel, cov = self.ekf.state_at(t_now)
    if t_now < self.handoff_until_t:
      cls = "UNKNOWN"
    else:
      cls = self.classifier.classify(t_now)
    spin = self.classifier.spin_model.to_dict() if cls == "SPIN" else None
    return {
      "t_state": t_now,
      "pos_gi": pos.tolist(),
      "vel_gi": vel.tolist(),
      "cov_pos": cov.tolist(),
      "pos_meas": self.last_meas.tolist() if self.last_meas is not None else None,
      "class": cls,
      "spin": spin,
      "target_id": self.target_id,
    }

# --- Daemon entry point ---

def _pnp(corners_2d:np.ndarray) -> Optional[np.ndarray]:
  ok, _, tvec = cv2.solvePnP(PLATE_POINTS, corners_2d, CANONICAL_CAMERA_MATRIX,
                             CANONICAL_DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE)
  return tvec.flatten() if ok else None

_SCALE_PX = np.array([IMG_W, IMG_H], dtype=np.float32)

def _pnp_pos_cov(corners:list, c_lo:Optional[list], c_hi:Optional[list]) -> tuple[Optional[np.ndarray], np.ndarray]:
  """Position from the mean corners; covariance by propagating the per-corner DFL quantiles through
  PnP as DETERMINISTIC sigma points: set each coord to its q_lo / q_hi (others at the mean), PnP
  each, accumulate cov = ¼ Σ (y_hi − y_lo)(y_hi − y_lo)ᵀ. Deterministic → stable R; base floor added.
  Returns (pos_cam, cov_cam); cov falls back to the base floor if quantiles are missing."""
  mean = np.array(corners, dtype=np.float32).reshape(4, 2) * _SCALE_PX
  pos = _pnp(mean)
  if pos is None or c_lo is None or c_hi is None:
    return pos, np.eye(3) * MEAS_NOISE_BASE
  lo = np.array(c_lo, dtype=np.float32).reshape(4, 2) * _SCALE_PX
  hi = np.array(c_hi, dtype=np.float32).reshape(4, 2) * _SCALE_PX
  cov = np.zeros((3, 3))
  for r in range(4):
    for c in range(2):
      pl = mean.copy(); pl[r, c] = lo[r, c]
      ph = mean.copy(); ph[r, c] = hi[r, c]
      yl, yh = _pnp(pl), _pnp(ph)
      if yl is None or yh is None: continue
      d = yh - yl
      cov += 0.25 * np.outer(d, d)
  return pos, cov + np.eye(3) * MEAS_NOISE_BASE

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

    pos_cam, R_cam = _pnp_pos_cov(autoaim["corners"], autoaim.get("corner_lo"), autoaim.get("corner_hi"))
    if pos_cam is None:
      state.push_invalid(t_capture)
      out = state.publish(time.monotonic())
      if out is not None: pub.send("plate", out)
      continue

    gp = gimbal_buf.interpolate(t_capture)
    if gp is None:
      # No gimbal_state at all — degraded mode: assume zero pose, inflate cov.
      yaw_gi_cap, pitch_gi_cap, stale = 0.0, 0.0, True
      if not warned_no_gimbal:
        logger.warning("plated: no gimbal_state samples; running in degraded mode")
        warned_no_gimbal = True
    else:
      yaw_gi_cap, pitch_gi_cap, stale = gp

    # rotate position and the PnP covariance from camera frame into the gimbal-inertial frame
    rot = R_yaw(yaw_gi_cap) @ R_pitch(pitch_gi_cap) @ R_MOUNT
    pos_gi = rot @ pos_cam + T_MOUNT
    R_meas = rot @ R_cam @ rot.T * (MEAS_NOISE_STALE_MULT if stale else 1.0)

    state.push_measurement(t_capture, pos_gi, meta, R_meas)

    out = state.publish(time.monotonic())
    if out is not None: pub.send("plate", out)
