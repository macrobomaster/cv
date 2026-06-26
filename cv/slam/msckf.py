"""Translation + yaw-bias MSCKF with gimbal-supplied orientation, in tinygrad.

The gimbal IMU is fused to absolute, gravity-referenced orientation. Its PITCH
is gravity-referenced (drift-free) so we take it as given; its YAW is
gyro-integrated and drifts, so the filter carries a single yaw-bias state δψ
(a small rotation about world-up) that's random-walked in predict and
corrected by AprilTag yaw (a proper Kalman update — no hand-tuned blend).

Nominal state (9 + 3·N_CLONES) held here:
  IMU:    [p_w(3), v_w(3), b_a(3)]
  Clones: [p_c_i(3)] × N_CLONES                       (camera positions)
The yaw-bias *nominal* (psi_corr) lives in slamd, which owns the gimbal
angles and builds the rotation matrices; this filter only carries δψ in the
ERROR state / covariance and returns the δψ increment from each update for
slamd to fold into psi_corr.

Error state (10 + 3·N_CLONES):
  [δp(3), δv(3), δb_a(3), δψ(1), (δp_c_i(3)) × N_CLONES]

Each clone's camera orientation is known (gimbal-derived, stored host-side).
δψ couples to position through the accel rotation in predict, so position and
yaw are correlated and the tag position / feature updates also touch yaw via
the covariance cross-terms.
"""
from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes

_DEV = "CPU"

def _np_to_tensor(arr:np.ndarray) -> Tensor:
  arr = np.ascontiguousarray(arr, dtype=np.float32)
  return Tensor.from_blob(arr.ctypes.data, arr.shape, dtype=dtypes.float32, device=_DEV)

from . import calib

PSI = 9                                                   # δψ error index
IMU_ERR_DIM = 10                                          # [p(3), v(3), b_a(3), δψ(1)]
CLONE_ERR_DIM = 3
N_CLONES   = calib.N_CLONES
CLONE_BLOCK = CLONE_ERR_DIM * N_CLONES
ERR_DIM    = IMU_ERR_DIM + CLONE_BLOCK
MAX_K      = 10
K_MIN      = 3

# ---- statics (CPU) ----
# Gravity added back only if the accelerometer reports raw specific force;
# zero it if the IMU already removed gravity (else the robot free-falls).
_G_W      = Tensor(calib.GRAVITY.tolist() if calib.ACCEL_INCLUDES_GRAVITY else [0.0, 0.0, 0.0],
                   device=_DEV)
_SIGMA2   = Tensor([calib.ACCEL_NOISE**2]*3 + [calib.ACCEL_BIAS_RW**2]*3, device=_DEV)
_YAW_RW   = calib.YAW_DRIFT_RW ** 2
_R_YAW    = calib.TAG_YAW_NOISE ** 2
_I3       = Tensor.eye(3).to(_DEV)
_Z3       = Tensor.zeros(3, 3, device=_DEV)
_I_ERR    = Tensor.eye(ERR_DIM).to(_DEV)
_I_CL     = Tensor.eye(CLONE_BLOCK).to(_DEV)
_ZERO31   = Tensor.zeros(3, 1, device=_DEV)
_ROW19    = Tensor.zeros(1, 9, device=_DEV)
_ONE11    = Tensor.ones(1, 1, device=_DEV)
_ROW16    = Tensor.zeros(1, 6, device=_DEV)
# one-hot outer at δψ (10x10), to inject yaw random-walk into Q
_e_psi    = Tensor(np.eye(IMU_ERR_DIM, dtype=np.float32)[PSI], device=_DEV)
_E_PSI    = (_e_psi.reshape(IMU_ERR_DIM, 1) @ _e_psi.reshape(1, IMU_ERR_DIM)).contiguous()
_Z_IMU_CL = Tensor.zeros(IMU_ERR_DIM, CLONE_BLOCK, device=_DEV)
_Z_CL_IMU = Tensor.zeros(CLONE_BLOCK, IMU_ERR_DIM, device=_DEV)
_R_PIXEL = calib.PIXEL_NOISE ** 2
_FX, _FY = calib.FX, calib.FY
_CX, _CY = calib.CX, calib.CY
_R_POS_MAT = (Tensor([calib.TAG_POS_NOISE**2]*3, device=_DEV).diag()).contiguous()
_R_POS_ROW = Tensor([calib.TAG_POS_NOISE**2]*3, device=_DEV).reshape(1, 3).contiguous()

def _hcat(ts): return ts[0].cat(*ts[1:], dim=-1)
def _vcat(ts): return ts[0].cat(*ts[1:], dim=-2)

# ============================================================================
# Kernels
# ============================================================================

@TinyJit
def _predict_kernel(p_w:Tensor, v_w:Tensor, b_a:Tensor, P:Tensor,
                    accel:Tensor, dt:Tensor, R_wb:Tensor):
  ap = (R_wb @ (accel - b_a).unsqueeze(-1)).squeeze(-1)               # a_w - g
  a_w = ap + _G_W
  p_new = p_w + v_w * dt + 0.5 * a_w * dt * dt
  v_new = v_w + a_w * dt
  # ẑ × ap  (sensitivity of world accel to a small yaw error about world-up)
  zc = (-ap[1:2]).cat(ap[0:1], Tensor.zeros(1, device=_DEV), dim=0)   # (3,)
  # F9 over [δp, δv, δb_a]
  F9 = _vcat([
    _hcat([_I3, _I3*dt, -0.5 * R_wb * (dt*dt)]),
    _hcat([_Z3, _I3,    -R_wb * dt           ]),
    _hcat([_Z3, _Z3,    _I3                  ]),
  ])
  col_psi = _vcat([(0.5*dt*dt) * zc.reshape(3,1), dt * zc.reshape(3,1), _ZERO31])  # (9,1)
  F_imu = _vcat([_hcat([F9, col_psi]), _hcat([_ROW19, _ONE11])])      # (10,10)
  G9 = _vcat([
    _hcat([-0.5 * R_wb * (dt*dt), _Z3]),
    _hcat([-R_wb * dt,            _Z3]),
    _hcat([_Z3,                   _I3]),
  ])
  G = _vcat([G9, _ROW16])                                             # (10,6)
  GQGt = (G * (_SIGMA2 * dt).unsqueeze(0)) @ G.transpose(-2, -1) + _E_PSI * (_YAW_RW * dt)
  F = F_imu.cat(_Z_IMU_CL, dim=1).cat(_Z_CL_IMU.cat(_I_CL, dim=1), dim=0)
  Q_d = GQGt.cat(_Z_IMU_CL, dim=1).cat(Tensor.zeros(CLONE_BLOCK, ERR_DIM, device=_DEV), dim=0)
  P_new = F @ P @ F.transpose(-2, -1) + Q_d
  return p_new, v_new, P_new

@TinyJit
def _augment_kernel(p_w:Tensor, p_cl:Tensor, P:Tensor, p_offset:Tensor):
  # Clone the camera POSITION = p_w + p_offset (deterministic offset given the
  # known orientation), so the clone error ≈ δp (rows 0:3). The tiny δψ
  # coupling through p_offset is neglected (mount lever is small).
  p_cam = p_w + p_offset
  new_p_cl = p_cl[1:].cat(p_cam.unsqueeze(0), dim=0)
  J_rows = P[0:3, :]
  J_kept = J_rows[:, :IMU_ERR_DIM].cat(J_rows[:, IMU_ERR_DIM+CLONE_ERR_DIM:], dim=1)
  self_cov = P[0:3, 0:3]
  tl = P[:IMU_ERR_DIM, :IMU_ERR_DIM]
  tr = P[:IMU_ERR_DIM, IMU_ERR_DIM+CLONE_ERR_DIM:]
  bl = P[IMU_ERR_DIM+CLONE_ERR_DIM:, :IMU_ERR_DIM]
  br = P[IMU_ERR_DIM+CLONE_ERR_DIM:, IMU_ERR_DIM+CLONE_ERR_DIM:]
  P_shift = tl.cat(tr, dim=1).cat(bl.cat(br, dim=1), dim=0)
  new_P = P_shift.cat(J_kept.transpose(-2, -1), dim=1).cat(
          J_kept.cat(self_cov, dim=1), dim=0)
  return new_p_cl, new_P

def _back_substitute(R:Tensor, b:Tensor) -> Tensor:
  n = int(R.shape[-1])
  rows = [None]*n
  for i in range(n-1, -1, -1):
    acc = b[i:i+1, :]
    for j in range(i+1, n): acc = acc - R[i:i+1, j:j+1] * rows[j]
    rows[i] = acc / R[i:i+1, i:i+1]
  return rows[0].cat(*rows[1:], dim=0)

def _inject(p_w, v_w, b_a, p_cl, dx):
  """Apply Euclidean increments; return the δψ increment for slamd's psi_corr."""
  p_new  = p_w + dx[0:3]
  v_new  = v_w + dx[3:6]
  ba_new = b_a + dx[6:9]
  dpsi   = dx[PSI:PSI+1]
  dxc = dx[IMU_ERR_DIM:].reshape(N_CLONES, 3)
  return p_new, v_new, ba_new, p_cl + dxc, dpsi

@TinyJit
def _feature_update_kernel(p_w:Tensor, v_w:Tensor, b_a:Tensor, p_cl:Tensor, P:Tensor,
                           uvs:Tensor, pw:Tensor, slot_oh:Tensor, mask:Tensor, R_cw:Tensor):
  p_obs = slot_oh @ p_cl
  diff  = (pw.reshape(1, 3) - p_obs).unsqueeze(-1)
  p_c   = (R_cw @ diff).squeeze(-1)
  x = p_c[:, 0:1]; y = p_c[:, 1:2]; z = p_c[:, 2:3]
  z_safe = z + (z.abs() < 1e-3).where(1e-3, 0.0)
  iz = 1.0 / z_safe
  z_pred = (_FX * x * iz + _CX).cat(_FY * y * iz + _CY, dim=1)
  mask2 = mask.repeat_interleave(2).unsqueeze(-1)
  r = ((uvs - z_pred) * mask.unsqueeze(-1)).reshape(MAX_K * 2)
  zero = Tensor.zeros_like(x)
  Jpc_r0 = (_FX * iz).cat(zero, -_FX * x * iz * iz, dim=1)
  Jpc_r1 = zero.cat(_FY * iz, -_FY * y * iz * iz, dim=1)
  Jpc = Jpc_r0.unsqueeze(1).cat(Jpc_r1.unsqueeze(1), dim=1)
  H_clone = Jpc @ R_cw
  H_f = (-H_clone).reshape(MAX_K * 2, 3) * mask2
  jc_flat = H_clone.reshape(MAX_K, 6)
  ko = (slot_oh.unsqueeze(-1) * jc_flat.unsqueeze(1)).reshape(MAX_K, N_CLONES, 2, 3)
  H_x_clone = ko.permute(0, 2, 1, 3).reshape(MAX_K * 2, CLONE_BLOCK)
  H_x = (Tensor.zeros(MAX_K * 2, IMU_ERR_DIM, device=_DEV).cat(H_x_clone, dim=1)) * mask2
  Q, _ = H_f.qr()
  Nproj = Q[:, 3:]
  H_o = Nproj.transpose(-2, -1) @ H_x
  r_o = Nproj.transpose(-2, -1) @ r
  S  = H_o @ P @ H_o.transpose(-2, -1) + _R_PIXEL * Tensor.eye(MAX_K*2 - 3).to(_DEV)
  PHt = P @ H_o.transpose(-2, -1)
  Q_s, R_s = S.qr()
  Xt = _back_substitute(R_s, Q_s.transpose(-2, -1) @ PHt.transpose(-2, -1))
  K  = Xt.transpose(-2, -1)
  dx = (K @ r_o.unsqueeze(-1)).squeeze(-1)
  IKH = _I_ERR - K @ H_o
  P_new = IKH @ P @ IKH.transpose(-2, -1) + (K * _R_PIXEL) @ K.transpose(-2, -1)
  p_new, v_new, ba_new, pcl_new, dpsi = _inject(p_w, v_w, b_a, p_cl, dx)
  return p_new, v_new, ba_new, pcl_new, P_new, dpsi

_H_POS = Tensor(np.concatenate(
  [np.eye(3, dtype=np.float32), np.zeros((3, ERR_DIM-3), np.float32)], axis=1),
  device=_DEV).contiguous()

@TinyJit
def _pos_update_kernel(p_w:Tensor, v_w:Tensor, b_a:Tensor, p_cl:Tensor, P:Tensor,
                       p_meas:Tensor):
  r = p_meas - p_w
  H = _H_POS
  S = H @ P @ H.transpose(-2, -1) + _R_POS_MAT
  PHt = P @ H.transpose(-2, -1)
  Q_s, R_s = S.qr()
  Xt = _back_substitute(R_s, Q_s.transpose(-2, -1) @ PHt.transpose(-2, -1))
  K  = Xt.transpose(-2, -1)
  dx = (K @ r.unsqueeze(-1)).squeeze(-1)
  IKH = _I_ERR - K @ H
  P_new = IKH @ P @ IKH.transpose(-2, -1) + (K * _R_POS_ROW) @ K.transpose(-2, -1)
  p_new, v_new, ba_new, pcl_new, dpsi = _inject(p_w, v_w, b_a, p_cl, dx)
  return p_new, v_new, ba_new, pcl_new, P_new, dpsi

_H_YAW = Tensor(np.eye(ERR_DIM, dtype=np.float32)[PSI:PSI+1], device=_DEV).contiguous()  # (1, ERR_DIM)

@TinyJit
def _yaw_update_kernel(p_w:Tensor, v_w:Tensor, b_a:Tensor, p_cl:Tensor, P:Tensor,
                       r_yaw:Tensor):
  """Scalar Kalman update of the yaw-bias δψ from an absolute yaw residual
  r_yaw = wrap(yaw_tag - (gimbal_yaw + psi_corr)). H = e_δψ, so S = P[ψ,ψ]+R."""
  Pcol = P[:, PSI:PSI+1]                                              # (ERR_DIM, 1)
  s = Pcol[PSI:PSI+1, :] + _R_YAW                                     # (1,1)
  K = Pcol / s                                                        # (ERR_DIM, 1)
  dx = (K * r_yaw).squeeze(-1)                                        # (ERR_DIM,)
  # Joseph form for the rank-1 update: P = (I - K H) P (I - K H)^T + K R K^T
  KH = K @ _H_YAW                                                     # (ERR_DIM, ERR_DIM)
  IKH = _I_ERR - KH
  P_new = IKH @ P @ IKH.transpose(-2, -1) + (K * _R_YAW) @ K.transpose(-2, -1)
  p_new, v_new, ba_new, pcl_new, dpsi = _inject(p_w, v_w, b_a, p_cl, dx)
  return p_new, v_new, ba_new, pcl_new, P_new, dpsi

# ============================================================================
# State container
# ============================================================================

@dataclass
class MsckfState:
  p_w: Tensor; v_w: Tensor; b_a: Tensor
  p_cl: Tensor
  R_cl: list
  t_cl: list; fid_cl: list
  P: Tensor

  @staticmethod
  def init(p_var:float=1e-4, v_var:float=1e-3, ba_var:float=1e-3,
           psi_var:float=0.09) -> "MsckfState":   # ~0.3 rad: gimbal yaw may have an
                                                   # unknown field-frame offset at boot;
                                                   # the first tags snap δψ in.
    p_w = Tensor.zeros(3, device=_DEV)
    v_w = Tensor.zeros(3, device=_DEV)
    b_a = Tensor.zeros(3, device=_DEV)
    p_cl = Tensor.zeros(N_CLONES, 3, device=_DEV)
    diag = [p_var]*3 + [v_var]*3 + [ba_var]*3 + [psi_var] + [p_var]*3*N_CLONES
    P = Tensor(diag, device=_DEV).diag()
    return MsckfState(
      p_w=p_w.contiguous(), v_w=v_w.contiguous(), b_a=b_a.contiguous(),
      p_cl=p_cl.contiguous(),
      R_cl=[np.eye(3, dtype=np.float32)]*N_CLONES,
      t_cl=[0.0]*N_CLONES, fid_cl=[-1]*N_CLONES,
      P=P.contiguous(),
    )

  def predict(self, accel:np.ndarray, dt:float, R_wb:np.ndarray) -> None:
    a   = _np_to_tensor(np.ascontiguousarray(accel, np.float32))
    dt_t = _np_to_tensor(np.array([dt], np.float32))
    R   = _np_to_tensor(np.ascontiguousarray(R_wb, np.float32))
    self.p_w, self.v_w, self.P = _predict_kernel(
      self.p_w, self.v_w, self.b_a, self.P, a, dt_t, R)

  def augment(self, t:float, frame_id:int, R_wc:np.ndarray, p_offset:np.ndarray) -> None:
    off = _np_to_tensor(np.ascontiguousarray(p_offset, np.float32))
    self.p_cl, self.P = _augment_kernel(self.p_w, self.p_cl, self.P, off)
    self.R_cl = self.R_cl[1:] + [np.ascontiguousarray(R_wc, np.float32)]
    self.t_cl = self.t_cl[1:] + [t]
    self.fid_cl = self.fid_cl[1:] + [frame_id]

  def update_with_features(self, points_world:list, observations:list) -> float:
    """Returns the accumulated δψ increment (for slamd's psi_corr)."""
    dpsi_total = 0.0
    for pw, obs in zip(points_world, observations):
      if len(obs) < K_MIN: continue
      obs_use = obs[-MAX_K:] if len(obs) > MAX_K else obs
      uvs_np  = np.zeros((MAX_K, 2),        np.float32)
      slot_oh = np.zeros((MAX_K, N_CLONES), np.float32)
      mask_np = np.zeros(MAX_K,             np.float32)
      Rcw_np  = np.tile(np.eye(3, dtype=np.float32), (MAX_K, 1, 1))
      for k, (slot, uv) in enumerate(obs_use):
        uvs_np[k] = uv; slot_oh[k, slot] = 1.0; mask_np[k] = 1.0
        Rcw_np[k] = self.R_cl[slot].T
      pw32 = np.ascontiguousarray(pw, np.float32)
      (self.p_w, self.v_w, self.b_a, self.p_cl, self.P, dpsi) = _feature_update_kernel(
        self.p_w, self.v_w, self.b_a, self.p_cl, self.P,
        _np_to_tensor(uvs_np), _np_to_tensor(pw32),
        _np_to_tensor(slot_oh), _np_to_tensor(mask_np), _np_to_tensor(Rcw_np))
      dpsi_total += float(dpsi.numpy()[0])
    return dpsi_total

  def update_with_position(self, p_meas:np.ndarray) -> float:
    p32 = np.ascontiguousarray(p_meas, np.float32)
    (self.p_w, self.v_w, self.b_a, self.p_cl, self.P, dpsi) = _pos_update_kernel(
      self.p_w, self.v_w, self.b_a, self.p_cl, self.P, _np_to_tensor(p32))
    return float(dpsi.numpy()[0])

  def update_with_yaw(self, r_yaw:float) -> float:
    """Kalman update of δψ from an absolute yaw residual. Returns δψ increment."""
    r = _np_to_tensor(np.array([r_yaw], np.float32))
    (self.p_w, self.v_w, self.b_a, self.p_cl, self.P, dpsi) = _yaw_update_kernel(
      self.p_w, self.v_w, self.b_a, self.p_cl, self.P, r)
    return float(dpsi.numpy()[0])
