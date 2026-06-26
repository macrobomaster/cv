"""Translation-only MSCKF with externally-supplied orientation, in tinygrad.

The gimbal IMU is already fused to absolute, gravity-referenced orientation
(yaw, pitch; roll≈0). Re-integrating raw gyro inside the filter would only
re-introduce the drift the gimbal firmware already removed, so we DON'T:
orientation is a *known input* (R_wb from gimbal_state), and the filter
estimates only translation. This is far simpler and more robust than full
6DoF VIO, and the world comes out gravity-aligned for free.

Nominal state (9 + 3·N_CLONES):
  IMU:    [p_w(3), v_w(3), b_a(3)]                  = 9
  Clones: [p_c_i(3)] × N_CLONES                      = 3·N   (camera positions)

Error state has the same dims — all Euclidean, no rotation error:
  [δp(3), δv(3), δb_a(3), (δp_c_i(3)) × N_CLONES]

Each clone's camera ORIENTATION (R_wc) is a known quantity (gimbal-derived at
augment time), stored host-side alongside the clone, not estimated. The
sliding window is always full: augment drops slot 0, shifts left, appends the
new camera position at slot N-1.
"""
from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes

# MSCKF is pinned to CPU regardless of Device.DEFAULT — tiny matrices that
# don't benefit from GPU, and CPU lets us share numpy memory via from_blob.
_DEV = "CPU"

def _np_to_tensor(arr:np.ndarray) -> Tensor:
  """Zero-copy view of a contiguous float32 numpy array via from_blob.
  Caller must keep `arr` alive past the JIT call."""
  arr = np.ascontiguousarray(arr, dtype=np.float32)
  return Tensor.from_blob(arr.ctypes.data, arr.shape, dtype=dtypes.float32, device=_DEV)

from . import calib

IMU_ERR_DIM = 9                                           # [p(3), v(3), b_a(3)]
CLONE_ERR_DIM = 3                                         # [p_c(3)]
N_CLONES   = calib.N_CLONES
CLONE_BLOCK = CLONE_ERR_DIM * N_CLONES                    # 3N
ERR_DIM    = IMU_ERR_DIM + CLONE_BLOCK                    # 9 + 3N
MAX_K      = 10                                            # max observations per feature
K_MIN      = 3                                             # min for null-space projection

# ============================================================================
# Module-level statics (CPU). Plain floats fold into kernel bodies.
# ============================================================================
_G_W      = Tensor([0.0, 0.0, -9.81], device=_DEV)
# process noise variance: accel white noise (3) then accel-bias random walk (3)
_SIGMA2   = Tensor([calib.ACCEL_NOISE**2]*3 + [calib.ACCEL_BIAS_RW**2]*3, device=_DEV)
_I3       = Tensor.eye(3).to(_DEV)
_Z3       = Tensor.zeros(3, 3, device=_DEV)
_I_ERR    = Tensor.eye(ERR_DIM).to(_DEV)
_Z_IMU_CL = Tensor.zeros(IMU_ERR_DIM, CLONE_BLOCK, device=_DEV)
_Z_CL_IMU = Tensor.zeros(CLONE_BLOCK, IMU_ERR_DIM, device=_DEV)
_I_CL     = Tensor.eye(CLONE_BLOCK).to(_DEV)
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
  """One IMU(accel) sample. Orientation R_wb is a known input."""
  a_w = (R_wb @ (accel - b_a).unsqueeze(-1)).squeeze(-1) + _G_W
  p_new = p_w + v_w * dt + 0.5 * a_w * dt * dt
  v_new = v_w + a_w * dt
  # Error-state transition F (9x9), order [δp, δv, δb_a]:
  #   δp' = δp + δv·dt - 0.5 R_wb dt² δb_a
  #   δv' =      δv     -     R_wb dt  δb_a
  #   δb_a' =                 δb_a
  rows = [
    [_I3, _I3*dt, -0.5 * R_wb * (dt*dt)],
    [_Z3, _I3,    -R_wb * dt           ],
    [_Z3, _Z3,    _I3                  ],
  ]
  F_imu = _vcat([_hcat(r) for r in rows])
  # noise input G (9x6) for [n_a(3), n_ba(3)]
  g_rows = [
    [-0.5 * R_wb * (dt*dt), _Z3],
    [-R_wb * dt,            _Z3],
    [_Z3,                   _I3],
  ]
  G = _vcat([_hcat(r) for r in g_rows])
  GQGt = (G * (_SIGMA2 * dt).unsqueeze(0)) @ G.transpose(-2, -1)        # (9,9)
  F = F_imu.cat(_Z_IMU_CL, dim=1).cat(_Z_CL_IMU.cat(_I_CL, dim=1), dim=0)
  Q_d = GQGt.cat(_Z_IMU_CL, dim=1).cat(Tensor.zeros(CLONE_BLOCK, ERR_DIM, device=_DEV), dim=0)
  P_new = F @ P @ F.transpose(-2, -1) + Q_d
  return p_new, v_new, P_new

@TinyJit
def _augment_kernel(p_w:Tensor, p_cl:Tensor, P:Tensor, p_offset:Tensor):
  """Clone the camera POSITION = p_w + p_offset (p_offset = R_wb·t_ic, known).
  The clone position error equals the IMU position error δp (deterministic
  offset), so the augment Jacobian just selects the δp block (rows 0:3)."""
  p_cam = p_w + p_offset
  new_p_cl = p_cl[1:].cat(p_cam.unsqueeze(0), dim=0)
  J_rows = P[0:3, :]                                                    # (3, ERR_DIM)
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
  """Euclidean injection — pure additions (no rotation state)."""
  p_new  = p_w + dx[0:3]
  v_new  = v_w + dx[3:6]
  ba_new = b_a + dx[6:9]
  dxc = dx[IMU_ERR_DIM:].reshape(N_CLONES, 3)
  return p_new, v_new, ba_new, p_cl + dxc

@TinyJit
def _feature_update_kernel(p_w:Tensor, v_w:Tensor, b_a:Tensor, p_cl:Tensor, P:Tensor,
                           uvs:Tensor, pw:Tensor, slot_oh:Tensor, mask:Tensor, R_cw:Tensor):
  """MSCKF feature update with KNOWN per-observation camera orientation.
  R_cw: (MAX_K, 3, 3) camera<-world rotations for each observation's clone."""
  p_obs = slot_oh @ p_cl                                              # (MAX_K, 3)
  diff  = (pw.reshape(1, 3) - p_obs).unsqueeze(-1)                    # (MAX_K, 3, 1)
  p_c   = (R_cw @ diff).squeeze(-1)                                   # (MAX_K, 3)
  x = p_c[:, 0:1]; y = p_c[:, 1:2]; z = p_c[:, 2:3]
  z_safe = z + (z.abs() < 1e-3).where(1e-3, 0.0)
  iz = 1.0 / z_safe
  z_pred = (_FX * x * iz + _CX).cat(_FY * y * iz + _CY, dim=1)
  mask2 = mask.repeat_interleave(2).unsqueeze(-1)
  r = ((uvs - z_pred) * mask.unsqueeze(-1)).reshape(MAX_K * 2)
  zero = Tensor.zeros_like(x)
  Jpc_r0 = (_FX * iz).cat(zero, -_FX * x * iz * iz, dim=1)
  Jpc_r1 = zero.cat(_FY * iz, -_FY * y * iz * iz, dim=1)
  Jpc = Jpc_r0.unsqueeze(1).cat(Jpc_r1.unsqueeze(1), dim=1)            # (MAX_K, 2, 3)
  # ∂r/∂p_c = Jpc @ R_cw ; ∂r/∂pw = -Jpc @ R_cw
  H_clone = Jpc @ R_cw                                                 # (MAX_K, 2, 3)
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
  p_new, v_new, ba_new, pcl_new = _inject(p_w, v_w, b_a, p_cl, dx)
  return p_new, v_new, ba_new, pcl_new, P_new

# Absolute POSITION measurement (e.g. AprilTag): H selects δp (rows 0:3).
_H_POS = Tensor(np.concatenate(
  [np.eye(3, dtype=np.float32), np.zeros((3, ERR_DIM-3), np.float32)], axis=1),
  device=_DEV).contiguous()

@TinyJit
def _pos_update_kernel(p_w:Tensor, v_w:Tensor, b_a:Tensor, p_cl:Tensor, P:Tensor,
                       p_meas:Tensor):
  """Absolute world-position measurement of the IMU body. 3-dim → S is 3x3
  (qr is replay-safe at 3x3, unlike the old 6x6 path)."""
  r = p_meas - p_w
  H = _H_POS
  S = H @ P @ H.transpose(-2, -1) + _R_POS_MAT                         # (3,3)
  PHt = P @ H.transpose(-2, -1)
  Q_s, R_s = S.qr()
  Xt = _back_substitute(R_s, Q_s.transpose(-2, -1) @ PHt.transpose(-2, -1))
  K  = Xt.transpose(-2, -1)
  dx = (K @ r.unsqueeze(-1)).squeeze(-1)
  IKH = _I_ERR - K @ H
  P_new = IKH @ P @ IKH.transpose(-2, -1) + (K * _R_POS_ROW) @ K.transpose(-2, -1)
  p_new, v_new, ba_new, pcl_new = _inject(p_w, v_w, b_a, p_cl, dx)
  return p_new, v_new, ba_new, pcl_new, P_new

# ============================================================================
# State container
# ============================================================================

@dataclass
class MsckfState:
  p_w: Tensor; v_w: Tensor; b_a: Tensor
  p_cl: Tensor                       # (N_CLONES, 3) camera positions
  R_cl: list                         # host-side camera orientations (3x3) per clone
  t_cl: list; fid_cl: list
  P: Tensor

  @staticmethod
  def init(p_var:float=1e-4, v_var:float=1e-3, ba_var:float=1e-3) -> "MsckfState":
    p_w = Tensor.zeros(3, device=_DEV)
    v_w = Tensor.zeros(3, device=_DEV)
    b_a = Tensor.zeros(3, device=_DEV)
    p_cl = Tensor.zeros(N_CLONES, 3, device=_DEV)
    diag = [p_var]*3 + [v_var]*3 + [ba_var]*3 + [p_var]*3*N_CLONES
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
    """R_wc: camera<-? world orientation of this frame's camera (known, stored).
    p_offset: R_wb·t_ic, the camera-position offset from the IMU position."""
    off = _np_to_tensor(np.ascontiguousarray(p_offset, np.float32))
    self.p_cl, self.P = _augment_kernel(self.p_w, self.p_cl, self.P, off)
    self.R_cl = self.R_cl[1:] + [np.ascontiguousarray(R_wc, np.float32)]
    self.t_cl = self.t_cl[1:] + [t]
    self.fid_cl = self.fid_cl[1:] + [frame_id]

  def update_with_features(self, points_world:list, observations:list) -> int:
    if not points_world: return 0
    used = 0
    for pw, obs in zip(points_world, observations):
      if len(obs) < K_MIN: continue
      obs_use = obs[-MAX_K:] if len(obs) > MAX_K else obs
      uvs_np  = np.zeros((MAX_K, 2),        np.float32)
      slot_oh = np.zeros((MAX_K, N_CLONES), np.float32)
      mask_np = np.zeros(MAX_K,             np.float32)
      Rcw_np  = np.tile(np.eye(3, dtype=np.float32), (MAX_K, 1, 1))
      for k, (slot, uv) in enumerate(obs_use):
        uvs_np[k] = uv
        slot_oh[k, slot] = 1.0
        mask_np[k] = 1.0
        Rcw_np[k] = self.R_cl[slot].T          # camera<-world
      pw32 = np.ascontiguousarray(pw, np.float32)
      (self.p_w, self.v_w, self.b_a, self.p_cl, self.P) = _feature_update_kernel(
        self.p_w, self.v_w, self.b_a, self.p_cl, self.P,
        _np_to_tensor(uvs_np), _np_to_tensor(pw32),
        _np_to_tensor(slot_oh), _np_to_tensor(mask_np), _np_to_tensor(Rcw_np))
      used += 1
    return used

  def update_with_position(self, p_meas:np.ndarray) -> None:
    """Absolute world-position fix of the IMU body (e.g. from an AprilTag)."""
    p32 = np.ascontiguousarray(p_meas, np.float32)
    (self.p_w, self.v_w, self.b_a, self.p_cl, self.P) = _pos_update_kernel(
      self.p_w, self.v_w, self.b_a, self.p_cl, self.P, _np_to_tensor(p32))
