"""Multi-State Constraint Kalman Filter (Mourikis & Roumeliotis 2007), tinygrad.

Sliding-window MSCKF with a fixed `N_CLONES` clone buffer in the state. All
hot paths are pure tinygrad with static shapes so `TinyJit` caches a single
compiled kernel per code path. Numpy is reserved for cold-path setup —
we no longer round-trip per camera frame.

Nominal state layout (16 + 7·N_CLONES):
  IMU:    [q_wb(4), p_w(3), v_w(3), b_g(3), b_a(3)]      = 16
  Clones: [(q_wbᵢ(4), p_wbᵢ(3))] × N_CLONES               = 7·N

Error state layout (15 + 6·N_CLONES):
  [δθ_b(3), δp_w(3), δv_w(3), δb_g(3), δb_a(3),
   (δθ_b_i(3), δp_w_i(3)) × N_CLONES]

Clone buffer is *always full*. Slot 0 is the oldest, slot N-1 is the
newest. Augment is "drop slot 0, shift left, append new at N-1" — a single
slice-and-cat. At startup all slots hold copies of the initial IMU pose
with the initial IMU pose covariance replicated into the clone diagonal;
they get washed out by real measurements over the first ~N camera frames.
"""
from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.dtype import dtypes

# MSCKF is pinned to CPU regardless of Device.DEFAULT. The filter math is on
# tiny matrices that don't benefit from GPU; pinning to CPU lets us share
# numpy memory via from_blob (true zero-copy) and avoids accidental device
# transfers when the rest of the project (autoaim) runs on a GPU device.
_DEV = "CPU"

def _np_to_tensor(arr:np.ndarray) -> Tensor:
  """Zero-copy view of a contiguous float32 numpy array via from_blob.
  Caller is responsible for keeping `arr` alive past the JIT call."""
  arr = np.ascontiguousarray(arr, dtype=np.float32)
  return Tensor.from_blob(arr.ctypes.data, arr.shape, dtype=dtypes.float32, device=_DEV)

from .linalg import (skew, quat_to_R, quat_mul, quat_exp_so3, quat_normalize, so3_log)
from .imu_preint import (DIM_ERR as IMU_ERR_DIM, DIM_NOISE as IMU_NOISE_DIM)
from . import calib

CLONE_ERR_DIM = 6
N_CLONES   = calib.N_CLONES
CLONE_BLOCK = CLONE_ERR_DIM * N_CLONES                    # 6N
ERR_DIM    = IMU_ERR_DIM + CLONE_BLOCK                    # 15 + 6N
MAX_K      = 10                                            # max observations per feature
K_MIN      = 3                                             # min for null-space projection

# ============================================================================
# Module-level statics. All pinned to CPU device.
# ============================================================================
_G_W      = Tensor([0.0, 0.0, -9.81], device=_DEV)
_SIGMA2   = Tensor([calib.GYRO_NOISE**2]*3 + [calib.ACCEL_NOISE**2]*3 +
                   [calib.GYRO_BIAS_RW**2]*3 + [calib.ACCEL_BIAS_RW**2]*3, device=_DEV)
_I3       = Tensor.eye(3).to(_DEV)
_Z3       = Tensor.zeros(3, 3, device=_DEV)
_I_ERR    = Tensor.eye(ERR_DIM).to(_DEV)
_Z_IMU_CL = Tensor.zeros(IMU_ERR_DIM, CLONE_BLOCK, device=_DEV)
_Z_CL_IMU = Tensor.zeros(CLONE_BLOCK, IMU_ERR_DIM, device=_DEV)
_I_CL     = Tensor.eye(CLONE_BLOCK).to(_DEV)
# Plain floats — folded into the kernel's compiled body as immediates.
_R_PIXEL = calib.PIXEL_NOISE ** 2
_FX, _FY = calib.FX, calib.FY
_CX, _CY = calib.CX, calib.CY

def _hcat(ts:list[Tensor]) -> Tensor: return ts[0].cat(*ts[1:], dim=-1)
def _vcat(ts:list[Tensor]) -> Tensor: return ts[0].cat(*ts[1:], dim=-2)

# ============================================================================
# Pure-tinygrad kernels
# ============================================================================

def _imu_phi_g(q_wb:Tensor, accel:Tensor, gyro:Tensor, b_g:Tensor, b_a:Tensor, dt:Tensor):
  w_b = gyro - b_g
  a_b = accel - b_a
  R_wb = quat_to_R(q_wb)
  Wx = skew(w_b)
  Ra = (R_wb @ a_b.unsqueeze(-1)).squeeze(-1)
  Rax = skew(Ra)
  rows = [
    [_I3 - Wx*dt,        _Z3,        _Z3,    -_I3*dt, _Z3      ],
    [_Z3,                _I3,        _I3*dt,  _Z3,    _Z3      ],
    [-Rax*dt,            _Z3,        _I3,     _Z3,    -R_wb*dt],
    [_Z3,                _Z3,        _Z3,     _I3,    _Z3      ],
    [_Z3,                _Z3,        _Z3,     _Z3,    _I3      ],
  ]
  Phi = _vcat([_hcat(r) for r in rows])
  g_rows = [
    [-_I3*dt,  _Z3,       _Z3,    _Z3   ],
    [_Z3,      _Z3,       _Z3,    _Z3   ],
    [_Z3,      -R_wb*dt,  _Z3,    _Z3   ],
    [_Z3,      _Z3,       _I3,    _Z3   ],
    [_Z3,      _Z3,       _Z3,    _I3   ],
  ]
  G = _vcat([_hcat(r) for r in g_rows])
  return Phi, G

def _imu_nominal(q_wb:Tensor, p_w:Tensor, v_w:Tensor,
                 accel:Tensor, gyro:Tensor, b_g:Tensor, b_a:Tensor, dt:Tensor):
  w_b = gyro - b_g
  a_b = accel - b_a
  R_wb = quat_to_R(q_wb)
  a_w = (R_wb @ a_b.unsqueeze(-1)).squeeze(-1) + _G_W
  p_new = p_w + v_w * dt + 0.5 * a_w * dt * dt
  v_new = v_w + a_w * dt
  dq = quat_exp_so3(w_b * dt)
  q_new = quat_normalize(quat_mul(q_wb, dq))
  return q_new, p_new, v_new

# ----- JIT'd kernels -------------------------------------------------------

@TinyJit
def _predict_kernel(q_wb:Tensor, p_w:Tensor, v_w:Tensor, b_g:Tensor, b_a:Tensor,
                    P:Tensor, accel:Tensor, gyro:Tensor, dt:Tensor):
  Phi_imu, G_imu = _imu_phi_g(q_wb, accel, gyro, b_g, b_a, dt)
  q_new, p_new, v_new = _imu_nominal(q_wb, p_w, v_w, accel, gyro, b_g, b_a, dt)
  Phi = Phi_imu.cat(_Z_IMU_CL, dim=1).cat(_Z_CL_IMU.cat(_I_CL, dim=1), dim=0)
  GQGt = (G_imu * (_SIGMA2 * dt).unsqueeze(0)) @ G_imu.transpose(-2, -1)
  Q_top = GQGt.cat(_Z_IMU_CL, dim=1)
  Q_bot = Tensor.zeros(CLONE_BLOCK, ERR_DIM, device=_DEV)
  Q_d   = Q_top.cat(Q_bot, dim=0)
  P_new = Phi @ P @ Phi.transpose(-2, -1) + Q_d
  return q_new, p_new, v_new, P_new

@TinyJit
def _augment_kernel(q_wb:Tensor, p_w:Tensor, q_cl:Tensor, p_cl:Tensor, P:Tensor,
                    q_ic:Tensor, t_ic:Tensor):
  # Clones store the SLAM-CAMERA pose (standard MSCKF). Compose the camera
  # pose from the IMU pose and the IMU<-camera extrinsic (q_ic, t_ic) at the
  # current gimbal pitch:  R_wc = R_wb @ R_ic,  p_wc = p_w + R_wb @ t_ic.
  q_cam = quat_normalize(quat_mul(q_wb, q_ic))
  p_cam = p_w + (quat_to_R(q_wb) @ t_ic.unsqueeze(-1)).squeeze(-1)
  new_q_cl = q_cl[1:].cat(q_cam.unsqueeze(0), dim=0)
  new_p_cl = p_cl[1:].cat(p_cam.unsqueeze(0), dim=0)
  # NOTE: the covariance augment below uses the identity Jacobian (clone error
  # ≈ IMU pose error). The exact form is the extrinsic adjoint; for a small
  # mount + deterministic pitch this is a minor covariance approximation, the
  # cloned camera *state* above is exact.
  # J_rows = [P[0:3,:]; P[3:6,:]]  with slot-0 cols (15:21) deleted
  J_rows = P[0:3, :].cat(P[3:6, :], dim=0)
  J_kept = J_rows[:, :IMU_ERR_DIM].cat(J_rows[:, IMU_ERR_DIM+CLONE_ERR_DIM:], dim=1)
  sc_top = P[0:3, 0:3].cat(P[0:3, 3:6], dim=1)
  sc_bot = P[3:6, 0:3].cat(P[3:6, 3:6], dim=1)
  self_cov = sc_top.cat(sc_bot, dim=0)
  # Slot-0 row/col block deleted
  tl = P[:IMU_ERR_DIM, :IMU_ERR_DIM]
  tr = P[:IMU_ERR_DIM, IMU_ERR_DIM+CLONE_ERR_DIM:]
  bl = P[IMU_ERR_DIM+CLONE_ERR_DIM:, :IMU_ERR_DIM]
  br = P[IMU_ERR_DIM+CLONE_ERR_DIM:, IMU_ERR_DIM+CLONE_ERR_DIM:]
  P_shift = tl.cat(tr, dim=1).cat(bl.cat(br, dim=1), dim=0)
  new_P_top = P_shift.cat(J_kept.transpose(-2, -1), dim=1)
  new_P_bot = J_kept.cat(self_cov, dim=1)
  new_P = new_P_top.cat(new_P_bot, dim=0)
  return new_q_cl, new_p_cl, new_P

def _back_substitute(R:Tensor, b:Tensor) -> Tensor:
  n = int(R.shape[-1])
  rows = [None]*n
  for i in range(n-1, -1, -1):
    acc = b[i:i+1, :]
    for j in range(i+1, n): acc = acc - R[i:i+1, j:j+1] * rows[j]
    rows[i] = acc / R[i:i+1, i:i+1]
  return rows[0].cat(*rows[1:], dim=0)

def _inject_imu(q_wb:Tensor, p_w:Tensor, v_w:Tensor, b_g:Tensor, b_a:Tensor, dx:Tensor):
  dth = dx[0:3]; dp = dx[3:6]; dv = dx[6:9]
  dbg = dx[9:12]; dba = dx[12:15]
  dq = quat_exp_so3(dth)
  return (quat_normalize(quat_mul(q_wb, dq)),
          p_w + dp, v_w + dv, b_g + dbg, b_a + dba)

def _inject_clones(q_cl:Tensor, p_cl:Tensor, dx:Tensor):
  dxc = dx[IMU_ERR_DIM:].reshape(N_CLONES, 6)
  dq  = quat_exp_so3(dxc[:, 0:3])
  q_new = quat_normalize(quat_mul(q_cl, dq))
  return q_new, p_cl + dxc[:, 3:6]

@TinyJit
def _feature_update_kernel(
  q_wb:Tensor, p_w:Tensor, v_w:Tensor, b_g:Tensor, b_a:Tensor,
  q_cl:Tensor, p_cl:Tensor, P:Tensor,
  uvs:Tensor, pw:Tensor, slot_oh:Tensor, mask:Tensor,
):
  q_obs = slot_oh @ q_cl                                              # (MAX_K, 4)
  p_obs = slot_oh @ p_cl                                              # (MAX_K, 3)
  R_wc  = quat_to_R(q_obs)
  R_cw  = R_wc.transpose(-2, -1)
  diff  = (pw.reshape(1, 3) - p_obs).unsqueeze(-1)
  p_c   = (R_cw @ diff).squeeze(-1)
  x = p_c[:, 0:1]; y = p_c[:, 1:2]; z = p_c[:, 2:3]
  z_safe = z + (z.abs() < 1e-3).where(1e-3, 0.0)
  iz = 1.0 / z_safe
  z_pred = (_FX * x * iz + _CX).cat(_FY * y * iz + _CY, dim=1)
  mask_v = mask.unsqueeze(-1)
  r_per  = (uvs - z_pred) * mask_v
  r = r_per.reshape(MAX_K * 2)
  zero = Tensor.zeros_like(x)
  Jpc_row0 = (_FX * iz).cat(zero, -_FX * x * iz * iz, dim=1)
  Jpc_row1 = zero.cat(_FY * iz, -_FY * y * iz * iz, dim=1)
  Jpc = Jpc_row0.unsqueeze(1).cat(Jpc_row1.unsqueeze(1), dim=1)        # (MAX_K, 2, 3)
  Jth = skew(p_c)
  Jp  = -R_cw
  J_clone = Jpc @ Jth.cat(Jp, dim=2)                                    # (MAX_K, 2, 6)
  H_f_per = -(Jpc @ R_cw)
  H_f     = H_f_per.reshape(MAX_K * 2, 3) * mask.repeat_interleave(2).unsqueeze(-1)
  jc_flat = J_clone.reshape(MAX_K, 12)
  ko = slot_oh.unsqueeze(-1) * jc_flat.unsqueeze(1)                    # (MAX_K, N_CLONES, 12)
  ko = ko.reshape(MAX_K, N_CLONES, 2, 6).permute(0, 2, 1, 3)
  H_x_clone = ko.reshape(MAX_K * 2, CLONE_BLOCK)
  H_x_imu   = Tensor.zeros(MAX_K * 2, IMU_ERR_DIM, device=_DEV)
  H_x       = H_x_imu.cat(H_x_clone, dim=1) * mask.repeat_interleave(2).unsqueeze(-1)
  Q, _ = H_f.qr()
  Nproj = Q[:, 3:]
  H_o   = Nproj.transpose(-2, -1) @ H_x
  r_o   = Nproj.transpose(-2, -1) @ r
  HP = H_o @ P
  S  = HP @ H_o.transpose(-2, -1) + _R_PIXEL * Tensor.eye(MAX_K*2 - 3).to(_DEV)
  PHt  = P @ H_o.transpose(-2, -1)
  Q_s, R_s = S.qr()
  rhs  = Q_s.transpose(-2, -1) @ PHt.transpose(-2, -1)
  Xt   = _back_substitute(R_s, rhs)
  K    = Xt.transpose(-2, -1)
  dx   = (K @ r_o.unsqueeze(-1)).squeeze(-1)
  IKH  = _I_ERR - K @ H_o
  P_new = IKH @ P @ IKH.transpose(-2, -1) + (K * _R_PIXEL) @ K.transpose(-2, -1)
  q_wb_new, p_w_new, v_w_new, b_g_new, b_a_new = _inject_imu(q_wb, p_w, v_w, b_g, b_a, dx)
  q_cl_new, p_cl_new = _inject_clones(q_cl, p_cl, dx)
  return (q_wb_new, p_w_new, v_w_new, b_g_new, b_a_new,
          q_cl_new, p_cl_new, P_new)

# H for the absolute 6DoF pose measurement is constant: identity on the IMU
# δθ (rows 0:3 → cols 0:3) and δp (rows 3:6 → cols 3:6) error blocks, zero
# everywhere else. Build it once.
_H_POSE = Tensor(np.concatenate([
  np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, ERR_DIM-3), np.float32)], axis=1),
  np.concatenate([np.zeros((3, 3), np.float32), np.eye(3, dtype=np.float32),
                  np.zeros((3, ERR_DIM-6), np.float32)], axis=1),
], axis=0), device=_DEV).contiguous()
# Measurement noise variance for an AprilTag pose fix [rot(3), pos(3)] — a
# fixed calibration value, baked in as a constant (matching how the feature
# kernel uses _R_PIXEL) so it's not a per-call JIT input.
_R_POSE_DIAG = [calib.TAG_ROT_NOISE**2]*3 + [calib.TAG_POS_NOISE**2]*3
_R_POSE_MAT  = Tensor(_R_POSE_DIAG, device=_DEV).diag().contiguous()
_R_POSE_ROW  = Tensor(_R_POSE_DIAG, device=_DEV).reshape(1, 6).contiguous()

# Static pads to lift the 6x6 innovation solve to 7x7: tinygrad's qr NaNs on
# JIT replay at *exactly* 6x6 (5,7,8,...,12 are all fine — shape-specific
# codegen bug). We block-diag a 1 onto S and a zero column onto PHt, solve the
# decoupled 7x7 (qr-safe), and slice the first 6 rows back out — so the whole
# pose update can be @TinyJit'd like every other path.
_PAD_COL6 = Tensor.zeros(6, 1, device=_DEV)
_PAD_ROW7 = Tensor.zeros(1, 6, device=_DEV).cat(Tensor.ones(1, 1, device=_DEV), dim=1)  # (1,7)
_PAD_PHT  = Tensor.zeros(ERR_DIM, 1, device=_DEV)

@TinyJit
def _pose_update_kernel(
  q_wb:Tensor, p_w:Tensor, v_w:Tensor, b_g:Tensor, b_a:Tensor,
  q_cl:Tensor, p_cl:Tensor, P:Tensor,
  q_meas:Tensor, p_meas:Tensor,
):
  """Absolute 6DoF pose measurement (e.g. AprilTag). q_meas/p_meas are the
  world<-body orientation/position."""
  # Residual: orientation in body-frame tangent, position in world.
  R_nom = quat_to_R(q_wb); R_meas = quat_to_R(q_meas)
  r_th  = so3_log((R_nom.transpose(-2, -1) @ R_meas))                  # (3,)
  r_p   = p_meas - p_w                                                  # (3,)
  r = r_th.cat(r_p, dim=0)                                              # (6,)
  H = _H_POSE
  HP = H @ P
  S  = HP @ H.transpose(-2, -1) + _R_POSE_MAT                          # (6, 6)
  PHt = P @ H.transpose(-2, -1)                                         # (ERR_DIM, 6)
  # Pad S→7x7 (decoupled corner) and PHt→7 cols; qr is replay-safe at 7x7.
  S7  = S.cat(_PAD_COL6, dim=1).cat(_PAD_ROW7, dim=0)                   # (7, 7)
  PHt7 = PHt.cat(_PAD_PHT, dim=1)                                       # (ERR_DIM, 7)
  Q_s, R_s = S7.qr()
  rhs = Q_s.transpose(-2, -1) @ PHt7.transpose(-2, -1)                 # (7, ERR_DIM)
  Xt  = _back_substitute(R_s, rhs)                                      # (7, ERR_DIM)
  K   = Xt[:6].transpose(-2, -1)                                        # (ERR_DIM, 6)
  dx  = (K @ r.unsqueeze(-1)).squeeze(-1)
  IKH = _I_ERR - K @ H
  P_new = IKH @ P @ IKH.transpose(-2, -1) + (K * _R_POSE_ROW) @ K.transpose(-2, -1)
  q_wb_new, p_w_new, v_w_new, b_g_new, b_a_new = _inject_imu(q_wb, p_w, v_w, b_g, b_a, dx)
  q_cl_new, p_cl_new = _inject_clones(q_cl, p_cl, dx)
  return (q_wb_new, p_w_new, v_w_new, b_g_new, b_a_new,
          q_cl_new, p_cl_new, P_new)

# ============================================================================
# State container (only the actually-mutable buffers)
# ============================================================================

@dataclass
class MsckfState:
  q_wb: Tensor; p_w: Tensor; v_w: Tensor; b_g: Tensor; b_a: Tensor
  q_cl: Tensor; p_cl: Tensor
  t_cl: list[float]; fid_cl: list[int]
  P: Tensor

  @staticmethod
  def init(p_var:float=1e-4, v_var:float=1e-3, th_var:float=1e-4,
           bg_var:float=1e-4, ba_var:float=1e-3) -> "MsckfState":
    q_wb = Tensor([1.0, 0.0, 0.0, 0.0], device=_DEV)
    p_w  = Tensor.zeros(3, device=_DEV)
    v_w  = Tensor.zeros(3, device=_DEV)
    b_g  = Tensor.zeros(3, device=_DEV)
    b_a  = Tensor.zeros(3, device=_DEV)
    q_cl = q_wb.reshape(1, 4).expand(N_CLONES, 4).contiguous()
    p_cl = p_w.reshape(1, 3).expand(N_CLONES, 3).contiguous()
    diag = ([th_var]*3 + [p_var]*3 + [v_var]*3 + [bg_var]*3 + [ba_var]*3
            + ([th_var]*3 + [p_var]*3) * N_CLONES)
    P = Tensor(diag, device=_DEV).diag()
    return MsckfState(
      q_wb=q_wb.contiguous(), p_w=p_w.contiguous(), v_w=v_w.contiguous(),
      b_g=b_g.contiguous(), b_a=b_a.contiguous(),
      q_cl=q_cl.contiguous(), p_cl=p_cl.contiguous(),
      t_cl=[0.0]*N_CLONES, fid_cl=[-1]*N_CLONES,
      P=P.contiguous(),
    )

  def predict(self, accel:np.ndarray, gyro:np.ndarray, dt:float) -> None:
    # Hold numpy refs locally so the from_blob pointers stay valid through
    # the JIT call.
    accel32 = np.ascontiguousarray(accel, dtype=np.float32)
    gyro32  = np.ascontiguousarray(gyro,  dtype=np.float32)
    dt_np   = np.array([dt], dtype=np.float32)
    a    = _np_to_tensor(accel32)
    g    = _np_to_tensor(gyro32)
    dt_t = _np_to_tensor(dt_np)
    self.q_wb, self.p_w, self.v_w, self.P = _predict_kernel(
      self.q_wb, self.p_w, self.v_w, self.b_g, self.b_a, self.P, a, g, dt_t)

  def augment(self, t:float, frame_id:int,
              q_ic:np.ndarray|None=None, t_ic:np.ndarray|None=None) -> None:
    """Clone the current SLAM-camera pose. (q_ic, t_ic) is the IMU<-camera
    extrinsic at this frame's gimbal pitch (calib.cam_from_imu); defaults to
    identity (camera == IMU frame)."""
    if q_ic is None: q_ic = np.array([1.0, 0, 0, 0], np.float32)
    if t_ic is None: t_ic = np.zeros(3, np.float32)
    self.q_cl, self.p_cl, self.P = _augment_kernel(
      self.q_wb, self.p_w, self.q_cl, self.p_cl, self.P,
      _np_to_tensor(np.ascontiguousarray(q_ic, np.float32)),
      _np_to_tensor(np.ascontiguousarray(t_ic, np.float32)))
    self.t_cl = self.t_cl[1:] + [t]
    self.fid_cl = self.fid_cl[1:] + [frame_id]

  def update_with_features(self, points_world:list[np.ndarray],
                           observations:list[list[tuple[int, np.ndarray]]]) -> int:
    if not points_world: return 0
    used = 0
    for pw, obs in zip(points_world, observations):
      if len(obs) < K_MIN: continue
      obs_use = obs[-MAX_K:] if len(obs) > MAX_K else obs
      uvs_np   = np.zeros((MAX_K, 2),         np.float32)
      slot_oh  = np.zeros((MAX_K, N_CLONES),  np.float32)
      mask_np  = np.zeros(MAX_K,              np.float32)
      for k, (slot, uv) in enumerate(obs_use):
        uvs_np[k] = uv
        slot_oh[k, slot] = 1.0
        mask_np[k] = 1.0
      pw32 = np.ascontiguousarray(pw, dtype=np.float32)  # kept alive in scope
      (self.q_wb, self.p_w, self.v_w, self.b_g, self.b_a,
       self.q_cl, self.p_cl, self.P) = _feature_update_kernel(
        self.q_wb, self.p_w, self.v_w, self.b_g, self.b_a,
        self.q_cl, self.p_cl, self.P,
        _np_to_tensor(uvs_np),
        _np_to_tensor(pw32),
        _np_to_tensor(slot_oh),
        _np_to_tensor(mask_np),
      )
      used += 1
    return used

  def update_with_pose(self, q_meas:np.ndarray, p_meas:np.ndarray) -> None:
    """Apply an absolute world<-body 6DoF pose measurement (e.g. AprilTag).

    q_meas: (4,) world<-body quaternion (w,x,y,z); p_meas: (3,) world position.
    Measurement noise is the fixed calibration value baked into the kernel
    (calib.TAG_ROT_NOISE / TAG_POS_NOISE).
    """
    q32 = np.ascontiguousarray(q_meas, dtype=np.float32)
    p32 = np.ascontiguousarray(p_meas, dtype=np.float32)
    (self.q_wb, self.p_w, self.v_w, self.b_g, self.b_a,
     self.q_cl, self.p_cl, self.P) = _pose_update_kernel(
      self.q_wb, self.p_w, self.v_w, self.b_g, self.b_a,
      self.q_cl, self.p_cl, self.P,
      _np_to_tensor(q32), _np_to_tensor(p32))
