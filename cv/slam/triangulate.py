"""Per-feature MSCKF triangulation by Gauss-Newton on inverse-depth.

Numpy only — this is a 3-DOF nonlinear least squares per feature, run
sequentially over a small number of features per camera frame. Tinygrad is
the wrong tool here: shapes are tiny, the loop is data-dependent (it stops
when converged or when it reaches max iters), and we already have a clean
numpy/scipy interface.

The output point is in the world frame. The MSCKF measurement update in
`msckf.py` then re-projects the world point into every observing clone and
computes residuals on those re-projections (Mourikis & Roumeliotis 2007).
"""
import numpy as np
from numpy.typing import NDArray

from . import calib

def _se3_inv(R:NDArray, t:NDArray) -> tuple[NDArray, NDArray]:
  Rt = R.T
  return Rt, -Rt @ t

def _se3_compose(R1:NDArray, t1:NDArray, R2:NDArray, t2:NDArray) -> tuple[NDArray, NDArray]:
  return R1 @ R2, R1 @ t2 + t1

def triangulate_feature(uv_obs:NDArray, R_wc:list[NDArray], t_wc:list[NDArray],
                        max_iters:int=10, tol:float=1e-5, huber_px:float=2.0
                        ) -> tuple[NDArray, bool]:
  """Triangulate one feature from N observations.

  Args:
    uv_obs:  (N, 2) float pixel coordinates in image space (canonical pinhole).
    R_wc:    list of N (3, 3) world<-camera rotations (one per observation).
    t_wc:    list of N (3,) world camera positions.
    max_iters: Gauss-Newton iteration cap.
    tol:     stop when ‖step‖∞ < tol.
    huber_px: per-pixel Huber threshold; residuals beyond this get a sub-linear
             weight so a single bad track doesn't dominate the fit.

  Returns:
    (point_world (3,), ok) — ok is False if no convergence, behind-camera, or
    Hessian was singular.
  """
  N = uv_obs.shape[0]
  assert N >= 2 and len(R_wc) == N and len(t_wc) == N

  # Normalize observations to ideal pinhole coords (drop intrinsics)
  fx, fy, cx, cy = calib.FX, calib.FY, calib.CX, calib.CY
  uvn = np.empty_like(uv_obs)
  uvn[:, 0] = (uv_obs[:, 0] - cx) / fx
  uvn[:, 1] = (uv_obs[:, 1] - cy) / fy

  # Anchor = first observation. Express every other camera relative to it.
  R_wa, t_wa = R_wc[0], t_wc[0]
  R_aw, t_aw = _se3_inv(R_wa, t_wa)
  R_ia = [None]*N
  t_ia = [None]*N
  for i in range(N):
    R_ia[i], t_ia[i] = _se3_compose(*_se3_inv(R_wc[i], t_wc[i]), R_wa, t_wa)

  # Initialize: α, β from anchor uv (perfect zero residual there); ρ from
  # mid-range arena depth (1/3 m^-1 ≈ 3 m).
  alpha, beta, rho = float(uvn[0, 0]), float(uvn[0, 1]), 1.0/3.0

  inv_sigma_sq = 1.0 / (calib.PIXEL_NOISE / fx) ** 2

  for _ in range(max_iters):
    H = np.zeros((3, 3), dtype=np.float64)
    g = np.zeros(3, dtype=np.float64)
    bad = False
    for i in range(N):
      Ri, ti = R_ia[i], t_ia[i]
      # Bearing ray in cam i: h = Ri @ [α, β, 1] + ρ ti  (depth-scaled). Project.
      r3 = Ri @ np.array([alpha, beta, 1.0]) + rho * ti
      z = r3[2]
      if z <= 1e-3:
        bad = True
        break
      pred = r3[:2] / z
      # Jacobian wrt (α, β, ρ):
      # d r3 / d (α, β) = Ri[:, 0:2]
      # d r3 / d ρ     = ti
      # d (x/z, y/z) / d r3 = [[1/z, 0, -x/z²], [0, 1/z, -y/z²]]
      inv_z = 1.0 / z
      drdv = np.empty((2, 3), dtype=np.float64)
      drdv[0] = [inv_z, 0.0, -r3[0] * inv_z * inv_z]
      drdv[1] = [0.0, inv_z, -r3[1] * inv_z * inv_z]
      dh_dx = np.column_stack([Ri[:, 0:2], ti])     # (3, 3)
      J = drdv @ dh_dx                              # (2, 3)
      res = pred - uvn[i]                           # (2,)
      # Huber weight in pixel units
      px_norm = float(np.hypot(res[0]*fx, res[1]*fy))
      w = 1.0 if px_norm <= huber_px else huber_px / px_norm
      H += w * inv_sigma_sq * (J.T @ J)
      g += w * inv_sigma_sq * (J.T @ res)
    if bad: return np.zeros(3, dtype=np.float32), False
    try: step = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError: return np.zeros(3, dtype=np.float32), False
    alpha += step[0]
    beta  += step[1]
    rho   += step[2]
    if np.max(np.abs(step)) < tol: break

  if rho <= 1e-4: return np.zeros(3, dtype=np.float32), False
  depth = 1.0 / rho
  p_anchor = np.array([alpha*depth, beta*depth, depth], dtype=np.float64)
  p_world = (R_wa @ p_anchor + t_wa).astype(np.float32)
  return p_world, True
