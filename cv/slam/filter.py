"""Map-localization pose filter — a plain 10-state EKF (numpy).

The robot localizes against a KNOWN field (AprilTags at surveyed locations), so
this is localization-to-a-map, not SLAM — no feature triangulation, no map
building. Orientation is a known input from the gimbal, which removes the SO(3)
nonlinearity that normally forces an EKF/UKF; what's left is nearly linear, so a
first-order EKF is exact enough (a UKF would be wasted sigma points). Contrast
plated's RobotUKF, whose trig measurement genuinely needs the unscented update.

State (held here): p_w(3), v_w(3), b_a(3). Error state (10): [δp, δv, δb_a, δψ].
δψ is the gimbal-yaw drift bias; its nominal (psi_corr) lives in slamd as
yaw_offset, which builds the rotation matrices.

Sensor division of labour:
  predict             — IMU accel (short-term, high-rate, slip-aware)
  update_with_velocity — wheel odometry (observes velocity directly) + planar vz=0
  update_with_position / update_with_yaw — AprilTags (drift-free absolute anchor)
The wheels kill the IMU's velocity drift and the tags kill its position drift, so
the IMU is never integrated unattended (pure IMU odometry would drift as t²).
"""
from dataclasses import dataclass

import numpy as np

from . import common

PSI = 9                                                   # δψ error index
ERR_DIM = 10                                              # [δp, δv, δb_a, δψ]

_GRAV = (common.GRAVITY.astype(np.float64) if common.ACCEL_INCLUDES_GRAVITY
         else np.zeros(3))
_ACC_N2, _ACCB_N2 = common.ACCEL_NOISE**2, common.ACCEL_BIAS_RW**2
_VEL_PN2 = common.VEL_PROCESS_NOISE**2
_VEL_OBSERVES_YAW = common.VEL_OBSERVES_YAW
_YAW_RW2 = common.YAW_DRIFT_RW**2
_R_POS = common.TAG_POS_NOISE**2
_R_YAW = common.TAG_YAW_NOISE**2
_R_VEL = common.WHEEL_VEL_NOISE**2
_R_VZ = common.VERT_VEL_NOISE**2
_VEL_MAX2 = common.WHEEL_SPEED_MAX**2
_I3 = np.eye(3)

@dataclass
class PoseEKF:
  p_w: np.ndarray; v_w: np.ndarray; b_a: np.ndarray
  P: np.ndarray

  @staticmethod
  def init(p_var=1e-4, v_var=1e-3, ba_var=1e-3, psi_var=0.09) -> "PoseEKF":
    diag = [p_var]*3 + [v_var]*3 + [ba_var]*3 + [psi_var]
    return PoseEKF(p_w=np.zeros(3), v_w=np.zeros(3), b_a=np.zeros(3),
                   P=np.diag(np.array(diag, np.float64)))

  # -- predict ----------------------------------------------------------------
  def predict_batch(self, accels:np.ndarray, dts:np.ndarray, R_wb:np.ndarray) -> None:
    """Integrate a batch of accel samples over one frame interval.
    Nominal p/v per-sample (exact); covariance once for the whole interval."""
    if len(dts) == 0: return
    accels = np.asarray(accels, np.float64).reshape(-1, 3)
    dts = np.asarray(dts, np.float64).reshape(-1)
    R = np.asarray(R_wb, np.float64)
    ba = self.b_a
    p, v = self.p_w.copy(), self.v_w.copy()
    for a, dt in zip(accels, dts):
      a_w = R @ (a - ba) + _GRAV
      p = p + v*dt + 0.5*a_w*dt*dt
      v = v + a_w*dt
    self.p_w, self.v_w = p, v

    total_dt = float(dts.sum())
    ap = R @ (accels.mean(0) - ba)                 # a_w-g, for the δψ coupling
    zc = np.array([-ap[1], ap[0], 0.0])            # ẑ × ap
    F = np.eye(ERR_DIM)
    F[0:3, 3:6]  = _I3 * total_dt
    F[0:3, 6:9]  = -0.5 * R * total_dt*total_dt
    F[0:3, PSI]  = 0.5 * zc * total_dt*total_dt
    F[3:6, 6:9]  = -R * total_dt
    F[3:6, PSI]  = zc * total_dt
    # process noise (accel white → p,v; accel-bias RW; explicit velocity RW; yaw RW)
    G = np.zeros((ERR_DIM, 6))
    G[0:3, 0:3] = -0.5 * R * total_dt*total_dt
    G[3:6, 0:3] = -R * total_dt
    G[6:9, 3:6] = _I3
    sig = np.array([_ACC_N2]*3 + [_ACCB_N2]*3) * total_dt
    Q = G @ np.diag(sig) @ G.T
    Q[3:6, 3:6] += _VEL_PN2 * total_dt * _I3   # keeps wheel/planar velocity pins authoritative
    Q[PSI, PSI] += _YAW_RW2 * total_dt
    self.P = F @ self.P @ F.T + Q

  def predict(self, accel:np.ndarray, dt:float, R_wb:np.ndarray) -> None:
    self.predict_batch(np.asarray(accel, np.float64)[None, :], np.array([dt]), R_wb)

  # -- updates ----------------------------------------------------------------
  def _apply(self, H:np.ndarray, r:np.ndarray, R:np.ndarray) -> float:
    """EKF update with Joseph form; injects dx, returns the δψ increment."""
    PHt = self.P @ H.T
    S = H @ PHt + R
    K = PHt @ np.linalg.inv(S)                          # K = P Hᵀ S⁻¹
    dx = K @ r
    IKH = np.eye(ERR_DIM) - K @ H
    self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
    self.p_w += dx[0:3]; self.v_w += dx[3:6]; self.b_a += dx[6:9]
    return float(dx[PSI])

  def update_with_velocity(self, vx:float, vy:float, psi:float) -> float:
    """Fuse a 2-D horizontal chassis velocity (wheel odometry, m/s) referenced to
    the gimbal heading `psi` (world yaw): vx=forward, vy=left. Observes v_w
    directly (the accelerometer can't see constant velocity), killing the coast
    in both stationary (v≈0) and moving cases — no separate ZUPT branch.

    Row 2 pins world-vertical v_w.z to 0 (planar-motion constraint): wheels only
    observe the horizontal plane, so without it the vertical channel drifts and
    b_a.z stays unobservable. VERT_VEL_NOISE is loose enough for ramps.

    Slip is treated as measurement noise (WHEEL_VEL_NOISE); only impossible
    readings (sensor faults) are rejected. NOT Mahalanobis-gated — the accel
    can't observe DC velocity, so P[v] understates the true uncertainty and a
    gate would reject the wheel readings that make velocity observable.
    TODO(skid): when wheels read ~0 but |accel| says we're still decelerating
    (hard stop → slide), down-weight this update so the IMU carries the slide."""
    if vx*vx + vy*vy > _VEL_MAX2: return 0.0
    c, s = np.cos(psi), np.sin(psi)
    e_fwd  = np.array([c, s, 0.0])           # gimbal forward in world = Rz(psi)·x̂
    e_left = np.array([-s, c, 0.0])          # gimbal left    in world = Rz(psi)·ŷ
    h1, h2, h3 = float(e_fwd @ self.v_w), float(e_left @ self.v_w), float(self.v_w[2])
    H = np.zeros((3, ERR_DIM))
    H[0, 3:6] = e_fwd; H[1, 3:6] = e_left; H[2, 5] = 1.0   # row 2: world-vertical v_w.z
    if _VEL_OBSERVES_YAW:
      H[0, PSI] = h2; H[1, PSI] = -h1        # ∂(Rz(psi)ᵀ v_w)/∂ψ → δψ coupling (horizontal only)
    r = np.array([vx - h1, vy - h2, 0.0 - h3])             # vz measured 0 (planar ground robot)
    return self._apply(H, r, np.diag([_R_VEL, _R_VEL, _R_VZ]))

  def update_with_position(self, p_meas:np.ndarray) -> float:
    """Absolute position fix from an AprilTag (known field map)."""
    H = np.zeros((3, ERR_DIM)); H[:, 0:3] = _I3
    r = np.asarray(p_meas, np.float64) - self.p_w
    return self._apply(H, r, _R_POS * _I3)

  def update_with_yaw(self, r_yaw:float) -> float:
    """Absolute yaw fix from an AprilTag → corrects the δψ gimbal-yaw drift bias."""
    H = np.zeros((1, ERR_DIM)); H[0, PSI] = 1.0
    return self._apply(H, np.array([r_yaw]), np.array([[_R_YAW]]))
