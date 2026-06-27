"""End-to-end + unit tests for the SLAM stack.

Run with: ``python -m cv.slam.test`` — each test prints PASS/FAIL and exits
nonzero if any fails.
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from .triangulate import triangulate_feature
from .msckf import MsckfState, ERR_DIM, N_CLONES, IMU_ERR_DIM, CLONE_ERR_DIM, PSI
from . import common

_failures = []
def _check(name:str, ok:bool, detail:str="") -> None:
  print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
  if not ok: _failures.append(name)

_I3 = np.eye(3, dtype=np.float64)
_Z3v = np.zeros(3, dtype=np.float64)
# Accelerometer reading at rest (so a_w = 0), consistent with the gravity flag:
# specific force (-g) if the IMU includes gravity, else gravity-removed (0).
_G = common.GRAVITY.astype(np.float64) if common.ACCEL_INCLUDES_GRAVITY else np.zeros(3)
_ACC_REST = (-_G).astype(np.float32)

# ---------------------------------------------------------------------------
def test_predict() -> None:
  # Orientation is an input now (R_wb). At rest the IMU reads _ACC_REST so a_w=0
  # → state stays put (flag-agnostic: works whether or not accel includes g).
  st = MsckfState.init()
  for _ in range(200): st.predict(_ACC_REST, 0.005, _I3)
  p, v = st.p_w, st.v_w
  _check("stationary stays still", float(np.abs(p).max()) < 1e-4 and float(np.abs(v).max()) < 1e-4,
         f"p={p}, v={v}")

  # 1 m/s² along +x for 1 s → p≈0.5, v≈1.0
  st = MsckfState.init()
  accel_x = _ACC_REST + np.array([1.0, 0, 0], np.float32)
  for _ in range(100): st.predict(accel_x, 0.01, _I3)
  p, v = st.p_w, st.v_w
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
    uvs.append([common.FX * p_c[0]/p_c[2] + common.CX, common.FY * p_c[1]/p_c[2] + common.CY])
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

  # Zero baseline (stationary cameras, identical views) → no parallax → reject.
  uv0 = np.array([common.FX*0.1 + common.CX, common.CY], np.float32)
  _, ok0 = triangulate_feature(np.array([uv0, uv0, uv0], np.float32),
                               [_I3, _I3, _I3], [_Z3v, _Z3v, _Z3v])
  _check("triangulation rejects zero parallax", not ok0)

# ---------------------------------------------------------------------------
def test_msckf_endtoend() -> None:
  """Constant-velocity trajectory + feature observations. Orientation is the
  known input (identity here). Verify stability + reasonable position."""
  st = MsckfState.init()
  st.v_w = np.array([0.5, 0.0, 0.0])
  accel = _ACC_REST                                       # a_w=0 with R_wb=I → const vel

  pw_true = np.array([3.0, 0.0, 0.5], dtype=np.float32)
  K = 5
  for k in range(K):
    for _ in range(25): st.predict(accel, 0.01, _I3)
    st.augment(t=0.25*(k+1), frame_id=k, R_wc=_I3, p_offset=_Z3v)

  rng = np.random.default_rng(1)
  p_cl_np = st.p_cl
  slots = list(range(N_CLONES - K, N_CLONES))     # the K augments land in the last K slots
  uvs = []
  for s in slots:
    p_c = pw_true - p_cl_np[s]
    if p_c[2] <= 0: continue
    uv = np.array([common.FX * p_c[0]/p_c[2] + common.CX, common.FY * p_c[1]/p_c[2] + common.CY])
    uv += rng.standard_normal(2) * 0.5
    uvs.append((s, uv.astype(np.float32)))
  Rs = [_I3 for _ in slots]
  ts = [p_cl_np[s] for s in slots]
  est_pw, ok = triangulate_feature(np.array([uv for _, uv in uvs], np.float32), Rs, ts)
  _check("VIO triangulation ok", ok)
  pos_var_before = float(np.trace(st.P[0:3, 0:3]))
  st.update_with_features([est_pw], [uvs])
  pos_var_after = float(np.trace(st.P[0:3, 0:3]))
  _check("MSCKF feature update reduces position variance", pos_var_after < pos_var_before,
         f"trace(P_pos) {pos_var_before:.2e} -> {pos_var_after:.2e}")

  P_eig = np.linalg.eigvalsh(st.P + 1e-9*np.eye(ERR_DIM))
  _check("P remains positive definite", float(P_eig.min()) > -1e-5, f"min eig={P_eig.min():.2e}")

  exp_x = 0.5 * 1.25  # 0.5 m/s × 1.25 s
  err_x = abs(float(st.p_w[0]) - exp_x)
  _check("filter p_x near 0.625m", err_x < 0.05, f"p_x={float(st.p_w[0]):.4f}")

# ---------------------------------------------------------------------------
def test_position_update() -> None:
  """Absolute position fix (AprilTag) pulls drift back and survives repeated
  calls (3x3 solve — no qr-replay issues)."""
  accel = _ACC_REST + np.array([1.0, 0.0, 0.0], np.float32)
  st = MsckfState.init()
  finite_all = True
  for k in range(6):
    for _ in range(10): st.predict(accel, 0.01, _I3)
    st.update_with_position(np.array([0.02*k, 0.0, 0.0], np.float32))
    if not np.isfinite(st.p_w).all(): finite_all = False; break
  _check("position update survives repeated calls", finite_all)

  st = MsckfState.init()
  for _ in range(50): st.predict(accel, 0.01, _I3)        # drifts +x
  x_before = float(st.p_w[0])
  st.update_with_position(np.zeros(3, np.float32))
  x_after = float(st.p_w[0])
  _check("position update reduces error", abs(x_after) < abs(x_before),
         f"x {x_before:.4f} -> {x_after:.4f}")

# ---------------------------------------------------------------------------
def test_yaw_update() -> None:
  """δψ yaw-bias state: covariance grows under random walk, a yaw residual
  pulls the correction toward it, and repeated updates stay finite."""
  accel = _ACC_REST
  st = MsckfState.init()
  psi0 = float(st.P[PSI, PSI])
  for _ in range(50): st.predict(accel, 0.01, _I3)        # random-walk grows δψ var
  psi1 = float(st.P[PSI, PSI])
  _check("yaw variance grows under random walk", psi1 > psi0, f"{psi0:.2e} -> {psi1:.2e}")

  # A yaw residual of +0.1 rad should be partially absorbed (covariance-weighted)
  # and shrink the yaw variance; the returned δψ increment has the right sign.
  dpsi = st.update_with_yaw(0.1)
  psi2 = float(st.P[PSI, PSI])
  _check("yaw update reduces yaw variance", psi2 < psi1, f"{psi1:.2e} -> {psi2:.2e}")
  _check("yaw correction has correct sign", 0.0 < dpsi <= 0.1, f"dpsi={dpsi:.4f}")

  # survives repeated interleaved predict + yaw updates
  finite = True
  for k in range(6):
    for _ in range(10): st.predict(accel, 0.01, _I3)
    st.update_with_yaw(0.05)
    if not np.isfinite(st.P).all(): finite = False; break
  _check("yaw update survives repeated calls", finite)

# ---------------------------------------------------------------------------
def test_velocity_update() -> None:
  """Wheel-odometry velocity update pins v_w (kills the coast); the gimbal
  heading psi rotates the measurement; impossible readings are rejected."""
  # Drifting velocity pulled back toward a zero (stopped) wheel reading.
  st = MsckfState.init()
  accel = _ACC_REST + np.array([1.0, 0, 0], np.float32)
  for _ in range(50): st.predict(accel, 0.01, _I3)         # v drifts +x
  v_before = float(np.linalg.norm(st.v_w))
  st.update_with_velocity(0.0, 0.0, 0.0)                    # wheels: stopped (psi=0)
  v_after = float(np.linalg.norm(st.v_w))
  _check("zero-velocity update reduces speed", v_after < v_before,
         f"|v| {v_before:.3f} -> {v_after:.3f}")

  # Impossible wheel speed (sensor fault) is rejected; state untouched.
  st = MsckfState.init(); st.v_w = np.array([0.2, 0.0, 0.0]); v_pre = st.v_w.copy()
  dpsi = st.update_with_velocity(1e3, 0.0, 0.0)
  _check("velocity rejects impossible reading", np.allclose(st.v_w, v_pre) and dpsi == 0.0,
         f"v={st.v_w}")

  # Heading rotates the measurement: forward speed at psi=90° → world +y.
  st = MsckfState.init()
  for _ in range(5): st.predict(_ACC_REST, 0.01, _I3)
  st.update_with_velocity(1.0, 0.0, np.pi/2)
  _check("velocity update respects gimbal heading", st.v_w[1] > abs(st.v_w[0]),
         f"v={st.v_w.round(3).tolist()}")

  # Planar constraint: a vertical-velocity drift (which wheels don't observe) is
  # pulled back toward 0 by the v_w.z=0 row — this is the "drifts upward" fix.
  st = MsckfState.init()
  accel_up = _ACC_REST + np.array([0, 0, 1.0], np.float32)  # spurious +z accel
  for _ in range(50): st.predict(accel_up, 0.01, _I3)        # v_z drifts up
  vz_before = float(st.v_w[2])
  st.update_with_velocity(0.0, 0.0, 0.0)
  _check("planar constraint pulls down vertical drift", abs(st.v_w[2]) < abs(vz_before),
         f"v_z {vz_before:.3f} -> {float(st.v_w[2]):.3f}")

  # Stopped robot under a persistent accel bias stays put when wheels report 0
  # each frame — velocity process noise keeps the pin authoritative so P[v]
  # doesn't collapse (the "stopped but keeps drifting incl. vertical" fix).
  st = MsckfState.init()
  bias = _ACC_REST + np.array([0.05, 0.0, 0.05], np.float32)   # x + z accel bias
  for k in range(300):
    st.predict(bias, 1/150, _I3)                                # ~2 s of IMU at 150 Hz
    if k % 10 == 0: st.update_with_velocity(0.0, 0.0, 0.0)      # wheels: stopped @ ~15 Hz
  _check("stopped stays bounded under accel bias", float(np.linalg.norm(st.p_w)) < 0.3,
         f"p={st.p_w.round(3).tolist()}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
  test_predict()
  test_triangulate()
  test_msckf_endtoend()
  test_position_update()
  test_yaw_update()
  test_velocity_update()
  print()
  if _failures:
    print(f"{len(_failures)} FAILURE(S): {_failures}")
    sys.exit(1)
  print("all good")
