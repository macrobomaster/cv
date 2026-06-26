"""End-to-end + unit tests for the SLAM stack.

Run with: ``python -m cv.slam.test`` — each test prints PASS/FAIL and exits
nonzero if any fails.
"""
import sys

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation as R
from tinygrad import Tensor

from .linalg import (det3, inv3, skew, vee, so3_exp, so3_log,
                     so3_right_jacobian, so3_right_jacobian_inv,
                     quat_mul, quat_to_R, quat_exp_so3,
                     solve_upper_triangular, solve_lower_triangular)
from .triangulate import triangulate_feature
from .msckf import MsckfState, ERR_DIM, N_CLONES, IMU_ERR_DIM, CLONE_ERR_DIM
from . import calib

_failures = []
def _check(name:str, ok:bool, detail:str="") -> None:
  print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
  if not ok: _failures.append(name)

# ---------------------------------------------------------------------------
def test_linalg() -> None:
  rng = np.random.default_rng(0)

  # det/inv
  M = rng.standard_normal((3,3)).astype(np.float32)
  _check("det3 matches numpy", abs(det3(Tensor(M)).item() - np.linalg.det(M)) < 1e-5)
  _check("inv3 matches numpy", np.allclose(inv3(Tensor(M)).numpy(), np.linalg.inv(M), atol=1e-4))

  # skew round-trip
  v = rng.standard_normal(3).astype(np.float32)
  _check("vee(skew(v)) == v", np.allclose(vee(skew(Tensor(v))).numpy(), v))

  # so3 exp/log
  for ang in (0.05, 0.5, 2.0):
    phi = rng.standard_normal(3).astype(np.float32)
    phi *= ang / max(np.linalg.norm(phi), 1e-8)
    R_ref = R.from_rotvec(phi).as_matrix().astype(np.float32)
    err_exp = np.abs(so3_exp(Tensor(phi)).numpy() - R_ref).max()
    err_log = np.abs(so3_log(Tensor(R_ref)).numpy() - phi).max()
    _check(f"so3_exp |phi|={ang}", err_exp < 1e-5, f"err={err_exp:.2e}")
    _check(f"so3_log |phi|={ang}", err_log < 1e-5, f"err={err_log:.2e}")

  # Right Jacobian sanity: exp(phi + d) ≈ exp(phi) · exp(Jr · d)
  phi = np.array([0.3, -0.4, 0.5], dtype=np.float32)
  d = np.array([1e-4, -1e-4, 1e-4], dtype=np.float32)
  R0 = R.from_rotvec(phi).as_matrix()
  R1 = R.from_rotvec(phi + d).as_matrix()
  Jr = so3_right_jacobian(Tensor(phi)).numpy()
  R1_pred = R0 @ R.from_rotvec(Jr @ d).as_matrix()
  _check("right Jacobian", np.abs(R1 - R1_pred).max() < 1e-5)
  Jri = so3_right_jacobian_inv(Tensor(phi)).numpy()
  _check("Jr·Jr⁻¹ = I", np.abs(Jr @ Jri - np.eye(3)).max() < 1e-5)

  # Quaternion ops
  q1 = rng.standard_normal(4); q1 /= np.linalg.norm(q1); q1 = q1.astype(np.float32)
  q2 = rng.standard_normal(4); q2 /= np.linalg.norm(q2); q2 = q2.astype(np.float32)
  R_q = quat_to_R(Tensor(q1)).numpy()
  R_sc = R.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix().astype(np.float32)
  _check("quat_to_R matches scipy", np.abs(R_q - R_sc).max() < 1e-5)
  q_prod = quat_mul(Tensor(q1), Tensor(q2)).numpy()
  q_sc = R.from_quat([q1[1], q1[2], q1[3], q1[0]]) * R.from_quat([q2[1], q2[2], q2[3], q2[0]])
  q_sc = q_sc.as_quat(); q_sc = np.array([q_sc[3], q_sc[0], q_sc[1], q_sc[2]])
  _check("quat_mul matches scipy", min(np.abs(q_prod - q_sc).max(), np.abs(q_prod + q_sc).max()) < 1e-5)

  # Triangular solves
  A = rng.standard_normal((6,6)).astype(np.float32)
  b = rng.standard_normal(6).astype(np.float32)
  U = np.triu(A); L = np.tril(A)
  for i in range(6):
    if abs(U[i,i]) < 0.5: U[i,i] += 1
    if abs(L[i,i]) < 0.5: L[i,i] += 1
  _check("upper-triangular solve",
         np.allclose(solve_upper_triangular(Tensor(U), Tensor(b)).numpy(),
                     solve_triangular(U, b, lower=False), atol=1e-4))
  _check("lower-triangular solve",
         np.allclose(solve_lower_triangular(Tensor(L), Tensor(b)).numpy(),
                     solve_triangular(L, b, lower=True), atol=1e-4))

_I3 = np.eye(3, dtype=np.float32)
_Z3v = np.zeros(3, dtype=np.float32)

# ---------------------------------------------------------------------------
def test_predict() -> None:
  # Orientation is an input now (R_wb). At rest with R_wb=I the accelerometer
  # reads -g = (0,0,9.81), so a_w=0 → state stays put.
  st = MsckfState.init()
  accel_rest = np.array([0, 0, 9.81], np.float32)
  for _ in range(200): st.predict(accel_rest, 0.005, _I3)
  p, v = st.p_w.numpy(), st.v_w.numpy()
  _check("stationary stays still", float(np.abs(p).max()) < 1e-4 and float(np.abs(v).max()) < 1e-4,
         f"p={p}, v={v}")

  # 1 m/s² along +x for 1 s → p≈0.5, v≈1.0
  st = MsckfState.init()
  accel_x = np.array([1.0, 0, 9.81], np.float32)
  for _ in range(100): st.predict(accel_x, 0.01, _I3)
  p, v = st.p_w.numpy(), st.v_w.numpy()
  _check("1 m/s² for 1s", abs(p[0]-0.5) < 1e-3 and abs(v[0]-1.0) < 1e-3,
         f"p_x={p[0]:.5f}, v_x={v[0]:.5f}")

# ---------------------------------------------------------------------------
def test_triangulate() -> None:
  pt_w = np.array([2.0, 0.5, 3.0], dtype=np.float64)
  Rs = [np.eye(3), np.eye(3),
        R.from_euler('y', 0.1).as_matrix()]
  ts = [np.zeros(3), np.array([0.5, 0, 0]), np.array([1.0, 0, 0])]
  uvs = []
  for Rm, t in zip(Rs, ts):
    p_c = Rm.T @ (pt_w - t)
    uvs.append([calib.FX * p_c[0]/p_c[2] + calib.CX, calib.FY * p_c[1]/p_c[2] + calib.CY])
  uvs = np.array(uvs, dtype=np.float32)
  est, ok = triangulate_feature(uvs, Rs, ts)
  _check("triangulation noiseless", ok and np.linalg.norm(est - pt_w) < 1e-3,
         f"err={np.linalg.norm(est - pt_w):.4f}")

  # With pixel noise
  rng = np.random.default_rng(7)
  uvs_n = uvs + rng.standard_normal(uvs.shape).astype(np.float32) * 0.5
  est, ok = triangulate_feature(uvs_n, Rs, ts)
  _check("triangulation noisy", ok and np.linalg.norm(est - pt_w) < 0.05,
         f"err={np.linalg.norm(est - pt_w):.4f}")

# ---------------------------------------------------------------------------
def test_msckf_endtoend() -> None:
  """Constant-velocity trajectory + feature observations. Orientation is the
  known input (identity here). Verify stability + reasonable position."""
  st = MsckfState.init()
  st.v_w = Tensor(np.array([0.5, 0.0, 0.0], dtype=np.float32), device="CPU").contiguous()
  accel = np.array([0.0, 0.0, 9.81], dtype=np.float32)   # a_w=0 with R_wb=I → const vel

  pw_true = np.array([3.0, 0.0, 0.5], dtype=np.float32)
  K = 5
  for k in range(K):
    for _ in range(25): st.predict(accel, 0.01, _I3)
    st.augment(t=0.25*(k+1), frame_id=k, R_wc=_I3, p_offset=_Z3v)

  rng = np.random.default_rng(1)
  p_cl_np = st.p_cl.numpy()
  slots = list(range(N_CLONES - K, N_CLONES))     # the K augments land in the last K slots
  uvs = []
  for s in slots:
    p_c = pw_true - p_cl_np[s]
    if p_c[2] <= 0: continue
    uv = np.array([calib.FX * p_c[0]/p_c[2] + calib.CX, calib.FY * p_c[1]/p_c[2] + calib.CY])
    uv += rng.standard_normal(2) * 0.5
    uvs.append((s, uv.astype(np.float32)))
  Rs = [_I3 for _ in slots]
  ts = [p_cl_np[s] for s in slots]
  est_pw, ok = triangulate_feature(np.array([uv for _, uv in uvs], np.float32), Rs, ts)
  _check("VIO triangulation ok", ok)
  pos_var_before = float(np.trace(st.P.numpy()[0:3, 0:3]))
  st.update_with_features([est_pw], [uvs])
  pos_var_after = float(np.trace(st.P.numpy()[0:3, 0:3]))
  _check("MSCKF feature update reduces position variance", pos_var_after < pos_var_before,
         f"trace(P_pos) {pos_var_before:.2e} -> {pos_var_after:.2e}")

  P_eig = np.linalg.eigvalsh(st.P.numpy() + 1e-9*np.eye(ERR_DIM))
  _check("P remains positive definite", float(P_eig.min()) > -1e-5, f"min eig={P_eig.min():.2e}")

  exp_x = 0.5 * 1.25  # 0.5 m/s × 1.25 s
  err_x = abs(float(st.p_w.numpy()[0]) - exp_x)
  _check("filter p_x near 0.625m", err_x < 0.05, f"p_x={float(st.p_w.numpy()[0]):.4f}")

# ---------------------------------------------------------------------------
def test_position_update() -> None:
  """Absolute position fix (AprilTag) pulls drift back and survives repeated
  calls (3x3 solve — no qr-replay issues)."""
  accel = np.array([1.0, 0.0, 9.81], dtype=np.float32)
  st = MsckfState.init()
  finite_all = True
  for k in range(6):
    for _ in range(10): st.predict(accel, 0.01, _I3)
    st.update_with_position(np.array([0.02*k, 0.0, 0.0], np.float32))
    if not np.isfinite(st.p_w.numpy()).all(): finite_all = False; break
  _check("position update survives repeated calls", finite_all)

  st = MsckfState.init()
  for _ in range(50): st.predict(accel, 0.01, _I3)        # drifts +x
  x_before = float(st.p_w.numpy()[0])
  st.update_with_position(np.zeros(3, np.float32))
  x_after = float(st.p_w.numpy()[0])
  _check("position update reduces error", abs(x_after) < abs(x_before),
         f"x {x_before:.4f} -> {x_after:.4f}")

# ---------------------------------------------------------------------------
def test_yaw_update() -> None:
  """δψ yaw-bias state: covariance grows under random walk, a yaw residual
  pulls the correction toward it, and repeated updates stay finite."""
  from .msckf import PSI
  accel = np.array([0, 0, 9.81], np.float32)
  st = MsckfState.init()
  psi0 = float(st.P.numpy()[PSI, PSI])
  for _ in range(50): st.predict(accel, 0.01, _I3)        # random-walk grows δψ var
  psi1 = float(st.P.numpy()[PSI, PSI])
  _check("yaw variance grows under random walk", psi1 > psi0, f"{psi0:.2e} -> {psi1:.2e}")

  # A yaw residual of +0.1 rad should be partially absorbed (covariance-weighted)
  # and shrink the yaw variance; the returned δψ increment has the right sign.
  dpsi = st.update_with_yaw(0.1)
  psi2 = float(st.P.numpy()[PSI, PSI])
  _check("yaw update reduces yaw variance", psi2 < psi1, f"{psi1:.2e} -> {psi2:.2e}")
  _check("yaw correction has correct sign", 0.0 < dpsi <= 0.1, f"dpsi={dpsi:.4f}")

  # survives repeated interleaved predict + yaw updates
  finite = True
  for k in range(6):
    for _ in range(10): st.predict(accel, 0.01, _I3)
    st.update_with_yaw(0.05)
    if not np.isfinite(st.P.numpy()).all(): finite = False; break
  _check("yaw update survives repeated calls", finite)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
  test_linalg()
  test_predict()
  test_triangulate()
  test_msckf_endtoend()
  test_position_update()
  test_yaw_update()
  print()
  if _failures:
    print(f"{len(_failures)} FAILURE(S): {_failures}")
    sys.exit(1)
  print("all good")
