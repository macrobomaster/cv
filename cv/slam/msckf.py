"""Translation + yaw-bias MSCKF with gimbal-supplied orientation — numpy.

Ported from tinygrad: the filter matrices are tiny (≈55×55) and the dense
linear algebra (Householder QR + triangular solve) compiled to ~175 micro
kernels per feature update on tinygrad-CPU (~3.6 ms), where numpy's single
LAPACK calls do the same in ~0.06 ms (≈50×). numpy *is* C (LAPACK/BLAS); the
per-op Python glue is negligible at this size. The autoaim model stays
tinygrad — this is just the filter.

Orientation is a known input (gimbal): pitch is gravity-referenced, yaw is a
small drift-bias state δψ corrected by AprilTags.

Nominal state (held here): p_w(3), v_w(3), b_a(3) + clone camera positions
p_c(3)×N. Error state (10 + 3N): [δp, δv, δb_a, δψ, (δp_c)×N]; δψ's nominal
(psi_corr) lives in slamd, which builds the rotation matrices.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from . import common

PSI = 9                                                   # δψ error index
IMU_ERR_DIM = 10                                          # [p,v,b_a,δψ]
CLONE_ERR_DIM = 3
N_CLONES   = common.N_CLONES
CLONE_BLOCK = CLONE_ERR_DIM * N_CLONES
ERR_DIM    = IMU_ERR_DIM + CLONE_BLOCK
K_MIN      = common.MIN_FEATURE_OBS                        # min obs for null-space projection

_GRAV = (common.GRAVITY.astype(np.float64) if common.ACCEL_INCLUDES_GRAVITY
         else np.zeros(3))
_ACC_N2, _ACCB_N2 = common.ACCEL_NOISE**2, common.ACCEL_BIAS_RW**2
_VEL_PN2 = common.VEL_PROCESS_NOISE**2
_YAW_RW2 = common.YAW_DRIFT_RW**2
_R_PIX = common.PIXEL_NOISE**2
_R_POS = common.TAG_POS_NOISE**2
_R_YAW = common.TAG_YAW_NOISE**2
_FX, _FY, _CX, _CY = common.FX, common.FY, common.CX, common.CY
_R_VEL = common.WHEEL_VEL_NOISE**2
_R_VZ = common.VERT_VEL_NOISE**2
_VEL_MAX2 = common.WHEEL_SPEED_MAX**2
_I3 = np.eye(3)
_I2 = np.eye(2)
# Mahalanobis gate thresholds (chi-square inverse-CDF) by measurement DOF,
# computed once. A measurement whose rᵀS⁻¹r exceeds its DOF's threshold is an
# outlier (wheel slip, degenerate triangulation) and is dropped.
_CHI2 = {d: float(chi2.ppf(common.GATE_CONFIDENCE, d)) for d in range(1, 2*N_CLONES + 1)}

@dataclass
class MsckfState:
  p_w: np.ndarray; v_w: np.ndarray; b_a: np.ndarray
  p_cl: np.ndarray                   # (N_CLONES, 3) camera positions
  R_cl: list                         # world<-camera per clone (known)
  t_cl: list; fid_cl: list
  P: np.ndarray

  @staticmethod
  def init(p_var=1e-4, v_var=1e-3, ba_var=1e-3, psi_var=0.09) -> "MsckfState":
    diag = ([p_var]*3 + [v_var]*3 + [ba_var]*3 + [psi_var] + [p_var]*3*N_CLONES)
    return MsckfState(
      p_w=np.zeros(3), v_w=np.zeros(3), b_a=np.zeros(3),
      p_cl=np.zeros((N_CLONES, 3)),
      R_cl=[np.eye(3) for _ in range(N_CLONES)],
      t_cl=[0.0]*N_CLONES, fid_cl=[-1]*N_CLONES,
      P=np.diag(np.array(diag, np.float64)),
    )

  # -- predict ----------------------------------------------------------------
  def predict_batch(self, accels:np.ndarray, dts:np.ndarray, R_wb:np.ndarray) -> None:
    """Integrate a batch of accel samples over one camera-frame interval.
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
    # process noise (accel white → p,v; accel-bias RW; yaw RW)
    G = np.zeros((IMU_ERR_DIM, 6))
    G[0:3, 0:3] = -0.5 * R * total_dt*total_dt
    G[3:6, 0:3] = -R * total_dt
    G[6:9, 3:6] = _I3
    sig = np.array([_ACC_N2]*3 + [_ACCB_N2]*3) * total_dt
    Q = np.zeros((ERR_DIM, ERR_DIM))
    Q[:IMU_ERR_DIM, :IMU_ERR_DIM] = G @ np.diag(sig) @ G.T
    Q[3:6, 3:6] += _VEL_PN2 * total_dt * _I3   # explicit velocity RW: keeps wheel/planar pins authoritative
    Q[PSI, PSI] += _YAW_RW2 * total_dt
    self.P = F @ self.P @ F.T + Q

  def predict(self, accel:np.ndarray, dt:float, R_wb:np.ndarray) -> None:
    self.predict_batch(np.asarray(accel, np.float64)[None, :], np.array([dt]), R_wb)

  # -- augment ----------------------------------------------------------------
  def augment(self, t:float, frame_id:int, R_wc:np.ndarray, p_offset:np.ndarray) -> None:
    """Clone the camera position p_w + p_offset; clone error ≈ δp (rows 0:3)."""
    p_cam = self.p_w + np.asarray(p_offset, np.float64)
    self.p_cl = np.vstack([self.p_cl[1:], p_cam])
    self.R_cl = self.R_cl[1:] + [np.asarray(R_wc, np.float64)]
    self.t_cl = self.t_cl[1:] + [t]
    self.fid_cl = self.fid_cl[1:] + [frame_id]
    # P: drop oldest clone (rows/cols), append a new clone correlated to δp.
    keep = list(range(IMU_ERR_DIM)) + list(range(IMU_ERR_DIM + CLONE_ERR_DIM, ERR_DIM))
    P = self.P[np.ix_(keep, keep)]                     # (ERR_DIM-3, ERR_DIM-3)
    J = self.P[0:3, keep]                              # cross-cov of new clone (=δp) with kept state
    self_cov = self.P[0:3, 0:3]
    newP = np.zeros((ERR_DIM, ERR_DIM))
    n = ERR_DIM - CLONE_ERR_DIM
    newP[:n, :n] = P
    newP[:n, n:] = J.T
    newP[n:, :n] = J
    newP[n:, n:] = self_cov
    self.P = newP

  # -- updates ----------------------------------------------------------------
  def _apply(self, H:np.ndarray, r:np.ndarray, R:np.ndarray, gate:float|None=None) -> float:
    """EKF update with Joseph form; injects dx, returns the δψ increment.
    If `gate` is given, the Mahalanobis distance rᵀS⁻¹r is checked first and the
    update is REJECTED (no state change, return 0) when it exceeds the gate — an
    outlier like wheel slip or a degenerate (zero-baseline) triangulation."""
    PHt = self.P @ H.T
    S = H @ PHt + R
    Sinv = np.linalg.inv(S)
    if gate is not None and float(r @ Sinv @ r) > gate: return 0.0
    K = PHt @ Sinv                                     # K = P Hᵀ S⁻¹
    dx = K @ r
    IKH = np.eye(ERR_DIM) - K @ H
    self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
    self.p_w += dx[0:3]; self.v_w += dx[3:6]; self.b_a += dx[6:9]
    self.p_cl += dx[IMU_ERR_DIM:].reshape(N_CLONES, 3)
    return float(dx[PSI])

  def update_with_features(self, points_world:list, observations:list) -> float:
    dpsi = 0.0
    for pw, obs in zip(points_world, observations):
      pw = np.asarray(pw, np.float64)
      Hx, Hf, res = [], [], []
      for slot, uv in obs:
        R_cw = self.R_cl[slot].T
        p_c = R_cw @ (pw - self.p_cl[slot])
        z = p_c[2]
        if z <= 1e-3: continue
        iz = 1.0/z
        Jpc = np.array([[_FX*iz, 0, -_FX*p_c[0]*iz*iz],
                        [0, _FY*iz, -_FY*p_c[1]*iz*iz]])
        JR = Jpc @ R_cw                                # (2,3)
        h = np.zeros((2, ERR_DIM))
        h[:, IMU_ERR_DIM + 3*slot: IMU_ERR_DIM + 3*slot + 3] = JR
        Hx.append(h); Hf.append(-JR)
        res.append(uv - np.array([_FX*p_c[0]*iz + _CX, _FY*p_c[1]*iz + _CY]))
      K = len(res)
      if K < K_MIN: continue
      Hx = np.vstack(Hx); Hf = np.vstack(Hf); res = np.concatenate(res)
      # null-space projection: kill dependence on the 3D point (not in state)
      Q, _ = np.linalg.qr(Hf, mode="complete")         # (2K, 2K)
      N = Q[:, 3:]                                      # (2K, 2K-3)
      H_o = N.T @ Hx; r_o = N.T @ res
      # Gated: a degenerate triangulation (no parallax) or a moving-scene point
      # that slipped past the front-end produces a large residual vs the
      # IMU-propagated state — drop it instead of injecting a huge correction.
      dpsi += self._apply(H_o, r_o, _R_PIX * np.eye(H_o.shape[0]), gate=_CHI2[H_o.shape[0]])
    return dpsi

  def update_with_velocity(self, vx:float, vy:float, psi:float) -> float:
    """Fuse a 2-D horizontal chassis velocity (wheel odometry, m/s) referenced to
    the gimbal heading `psi` (world yaw): vx=forward, vy=left. This observes v_w
    directly (the accelerometer can't see constant velocity), killing the coast
    in both the stationary (v≈0) and moving cases with no separate ZUPT branch.
    The measurement passes through psi, so it also constrains δψ.

    Also pins the world-vertical velocity v_w.z to 0 (planar-motion constraint):
    the robot is on the ground, and wheels only observe the horizontal plane, so
    without this the vertical channel drifts (the "drifts upward" bug) and b_a.z
    stays unobservable. VERT_VEL_NOISE is loose enough to allow ramps.

    Slip is treated as measurement noise (WHEEL_VEL_NOISE); only physically-
    impossible readings (sensor faults) are rejected. NOT Mahalanobis-gated — the
    accelerometer can't observe DC velocity, so P[v] understates the true
    uncertainty and a chi-square gate would reject the good wheel readings that
    make velocity observable in the first place."""
    if vx*vx + vy*vy > _VEL_MAX2: return 0.0
    c, s = np.cos(psi), np.sin(psi)
    e_fwd  = np.array([c, s, 0.0])           # gimbal forward in world = Rz(psi)·x̂
    e_left = np.array([-s, c, 0.0])          # gimbal left    in world = Rz(psi)·ŷ
    h1, h2, h3 = float(e_fwd @ self.v_w), float(e_left @ self.v_w), float(self.v_w[2])
    H = np.zeros((3, ERR_DIM))
    H[0, 3:6] = e_fwd; H[1, 3:6] = e_left; H[2, 5] = 1.0   # row 2: world-vertical v_w.z
    H[0, PSI] = h2; H[1, PSI] = -h1          # ∂(Rz(psi)ᵀ v_w)/∂ψ → δψ coupling (horizontal only)
    r = np.array([vx - h1, vy - h2, 0.0 - h3])             # vz measured 0 (planar ground robot)
    return self._apply(H, r, np.diag([_R_VEL, _R_VEL, _R_VZ]))

  def update_with_position(self, p_meas:np.ndarray) -> float:
    H = np.zeros((3, ERR_DIM)); H[:, 0:3] = _I3
    r = np.asarray(p_meas, np.float64) - self.p_w
    return self._apply(H, r, _R_POS * _I3)

  def update_with_yaw(self, r_yaw:float) -> float:
    H = np.zeros((1, ERR_DIM)); H[0, PSI] = 1.0
    return self._apply(H, np.array([r_yaw]), np.array([[_R_YAW]]))
