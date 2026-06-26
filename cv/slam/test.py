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
from .imu_preint import imu_step_nominal, imu_step_phi_g, DIM_ERR, DIM_NOISE
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

# ---------------------------------------------------------------------------
def test_imu_preint() -> None:
  q = Tensor(np.array([1.0, 0, 0, 0], np.float32))
  p = Tensor.zeros(3); v = Tensor.zeros(3)
  b_g = Tensor.zeros(3); b_a = Tensor.zeros(3)
  g_w = Tensor(calib.GRAVITY)

  # Stationary IMU: must stay at rest. accel reads −R_bw·g = (0,0,9.81) for identity attitude.
  a = Tensor(np.array([0,0,9.81], np.float32)); g = Tensor.zeros(3)
  for _ in range(200): q, p, v = imu_step_nominal(q, p, v, a, g, b_g, b_a, g_w, 0.005)
  _check("stationary IMU stays still", float(p.numpy().max()) < 1e-5 and float(v.numpy().max()) < 1e-5,
         f"p={p.numpy()}, v={v.numpy()}")

  # 1 m/s² along +x for 1 s → p≈0.5, v≈1.0
  a = Tensor(np.array([1.0,0,9.81], np.float32))
  q, p, v = Tensor(np.array([1.0,0,0,0], np.float32)), Tensor.zeros(3), Tensor.zeros(3)
  for _ in range(100): q, p, v = imu_step_nominal(q, p, v, a, g, b_g, b_a, g_w, 0.01)
  _check("1 m/s² for 1s", abs(p.numpy()[0] - 0.5) < 1e-3 and abs(v.numpy()[0] - 1.0) < 1e-3,
         f"p={p.numpy()[0]:.5f}, v={v.numpy()[0]:.5f}")

  # Phi/G shapes
  Phi, G = imu_step_phi_g(Tensor(np.array([1.0,0,0,0], np.float32)),
                          Tensor.zeros(3), Tensor.zeros(3), b_g, b_a, 0.01)
  _check("Phi shape", Phi.shape == (DIM_ERR, DIM_ERR))
  _check("G shape", G.shape == (DIM_ERR, DIM_NOISE))

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
  """Constant-velocity trajectory + a few feature observations. Verify the
  filter remains stable and outputs reasonable pose estimates."""
  st = MsckfState.init()
  st.v_w = Tensor(np.array([0.5, 0.0, 0.0], dtype=np.float32), device="CPU")
  accel = np.array([0.0, 0.0, 9.81], dtype=np.float32)
  gyro = np.zeros(3, dtype=np.float32)

  pw_true = np.array([3.0, 0.0, 0.5], dtype=np.float32)
  K = 5
  for k in range(K):
    for _ in range(25): st.predict(accel, gyro, 0.01)
    st.augment(t=0.25*(k+1), frame_id=k)

  # In the always-full sliding window, the K augments land in the last K
  # slots (N_CLONES-K..N_CLONES-1, oldest→newest).
  rng = np.random.default_rng(1)
  p_cl_np = st.p_cl.numpy()
  slots = list(range(N_CLONES - K, N_CLONES))
  uvs = []
  for s in slots:
    p_c = pw_true - p_cl_np[s]
    if p_c[2] <= 0: continue
    uv = np.array([calib.FX * p_c[0]/p_c[2] + calib.CX, calib.FY * p_c[1]/p_c[2] + calib.CY])
    uv += rng.standard_normal(2) * 0.5
    uvs.append((s, uv.astype(np.float32)))
  Rs = [np.eye(3) for _ in slots]
  ts = [p_cl_np[s] for s in slots]
  est_pw, ok = triangulate_feature(np.array([uv for _, uv in uvs], np.float32), Rs, ts)
  _check("VIO triangulation ok", ok)
  n_used = st.update_with_features([est_pw], [uvs])
  _check("MSCKF update accepted feature", n_used == 1)

  # Verify P stays ~PSD. Threshold is float32-roundoff tolerant: CPU reduction
  # ordering is nondeterministic so the smallest eigenvalue jitters around 0 by
  # ~1e-6 on a covariance whose entries are ~1e-3. A real non-PD bug shows much
  # larger negatives.
  P_eig = np.linalg.eigvalsh(st.P.numpy() + 1e-9*np.eye(ERR_DIM))
  _check("P remains positive definite", float(P_eig.min()) > -1e-5,
         f"min eig={P_eig.min():.2e}")

  # Position should be approximately the constant-velocity prediction
  exp_x = 0.5 * 1.25  # 0.5 m/s × 1.25 s
  err_x = abs(float(st.p_w.numpy()[0]) - exp_x)
  _check("filter p_x near 0.625m", err_x < 0.05, f"p_x={float(st.p_w.numpy()[0]):.4f}")

# ---------------------------------------------------------------------------
def test_apriltag_pose_update() -> None:
  """An absolute pose measurement should pull a drifted state back, and must
  survive many calls (guards the tinygrad 6x6 qr JIT-replay bug)."""
  def q_yaw(a):
    q = R.from_euler('z', a).as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

  st = MsckfState.init()
  accel = np.array([1.0, 0.0, 9.81], dtype=np.float32)
  gyro  = np.zeros(3, dtype=np.float32)

  finite_all = True
  for k in range(6):
    for _ in range(10): st.predict(accel, gyro, 0.01)
    # truth: at origin, yaw = 0.1*k
    st.update_with_pose(q_yaw(0.1*k), np.array([0.02*k, 0.0, 0.0], np.float32))
    if not (np.isfinite(st.q_wb.numpy()).all() and np.isfinite(st.p_w.numpy()).all()):
      finite_all = False; break
  _check("pose update survives repeated calls (no qr-JIT NaN)", finite_all)

  # A tight pose fix should pull the drifted x back toward the measurement.
  st = MsckfState.init()
  for _ in range(50): st.predict(accel, gyro, 0.01)   # drifts +x
  x_before = float(st.p_w.numpy()[0])
  st.update_with_pose(np.array([1.0,0,0,0], np.float32), np.zeros(3, np.float32))
  x_after = float(st.p_w.numpy()[0])
  _check("pose update reduces position error", abs(x_after) < abs(x_before),
         f"x {x_before:.4f} -> {x_after:.4f}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
  test_linalg()
  test_imu_preint()
  test_triangulate()
  test_msckf_endtoend()
  test_apriltag_pose_update()
  print()
  if _failures:
    print(f"{len(_failures)} FAILURE(S): {_failures}")
    sys.exit(1)
  print("all good")
