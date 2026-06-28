"""Unit tests for the localization pose filter.

Run with: ``python -m cv.slam.test`` — each test prints PASS/FAIL and exits
nonzero if any fails.
"""
import sys

import numpy as np

from .filter import PoseEKF, ERR_DIM, PSI
from . import common

_failures = []
def _check(name:str, ok:bool, detail:str="") -> None:
  print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
  if not ok: _failures.append(name)

_I3 = np.eye(3, dtype=np.float64)
# Accelerometer reading at rest (so a_w = 0), consistent with the gravity flag:
# specific force (-g) if the IMU includes gravity, else gravity-removed (0).
_G = common.GRAVITY.astype(np.float64) if common.ACCEL_INCLUDES_GRAVITY else np.zeros(3)
_ACC_REST = (-_G).astype(np.float32)

# ---------------------------------------------------------------------------
def test_predict() -> None:
  # Orientation is an input (R_wb). At rest the IMU reads _ACC_REST so a_w=0
  # → state stays put (flag-agnostic: works whether or not accel includes g).
  st = PoseEKF.init()
  for _ in range(200): st.predict(_ACC_REST, 0.005, _I3)
  p, v = st.p_w, st.v_w
  _check("stationary stays still", float(np.abs(p).max()) < 1e-4 and float(np.abs(v).max()) < 1e-4,
         f"p={p}, v={v}")

  # 1 m/s² along +x for 1 s → p≈0.5, v≈1.0
  st = PoseEKF.init()
  accel_x = _ACC_REST + np.array([1.0, 0, 0], np.float32)
  for _ in range(100): st.predict(accel_x, 0.01, _I3)
  p, v = st.p_w, st.v_w
  _check("1 m/s² for 1s", abs(p[0]-0.5) < 1e-3 and abs(v[0]-1.0) < 1e-3,
         f"p_x={p[0]:.5f}, v_x={v[0]:.5f}")

# ---------------------------------------------------------------------------
def test_position_update() -> None:
  """Absolute position fix (AprilTag) pulls drift back and survives repeated calls."""
  accel = _ACC_REST + np.array([1.0, 0.0, 0.0], np.float32)
  st = PoseEKF.init()
  finite_all = True
  for k in range(6):
    for _ in range(10): st.predict(accel, 0.01, _I3)
    st.update_with_position(np.array([0.02*k, 0.0, 0.0], np.float32))
    if not np.isfinite(st.p_w).all(): finite_all = False; break
  _check("position update survives repeated calls", finite_all)

  st = PoseEKF.init()
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
  st = PoseEKF.init()
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
  st = PoseEKF.init()
  accel = _ACC_REST + np.array([1.0, 0, 0], np.float32)
  for _ in range(50): st.predict(accel, 0.01, _I3)         # v drifts +x
  v_before = float(np.linalg.norm(st.v_w))
  st.update_with_velocity(0.0, 0.0, 0.0)                    # wheels: stopped (psi=0)
  v_after = float(np.linalg.norm(st.v_w))
  _check("zero-velocity update reduces speed", v_after < v_before,
         f"|v| {v_before:.3f} -> {v_after:.3f}")

  # Impossible wheel speed (sensor fault) is rejected; state untouched.
  st = PoseEKF.init(); st.v_w = np.array([0.2, 0.0, 0.0]); v_pre = st.v_w.copy()
  dpsi = st.update_with_velocity(1e3, 0.0, 0.0)
  _check("velocity rejects impossible reading", np.allclose(st.v_w, v_pre) and dpsi == 0.0,
         f"v={st.v_w}")

  # Heading rotates the measurement: forward speed at psi=90° → world +y.
  st = PoseEKF.init()
  for _ in range(5): st.predict(_ACC_REST, 0.01, _I3)
  st.update_with_velocity(1.0, 0.0, np.pi/2)
  _check("velocity update respects gimbal heading", st.v_w[1] > abs(st.v_w[0]),
         f"v={st.v_w.round(3).tolist()}")

  # Planar constraint: a vertical-velocity drift (which wheels don't observe) is
  # pulled back toward 0 by the v_w.z=0 row — this is the "drifts upward" fix.
  st = PoseEKF.init()
  accel_up = _ACC_REST + np.array([0, 0, 1.0], np.float32)  # spurious +z accel
  for _ in range(50): st.predict(accel_up, 0.01, _I3)        # v_z drifts up
  vz_before = float(st.v_w[2])
  st.update_with_velocity(0.0, 0.0, 0.0)
  _check("planar constraint pulls down vertical drift", abs(st.v_w[2]) < abs(vz_before),
         f"v_z {vz_before:.3f} -> {float(st.v_w[2]):.3f}")

  # Stopped robot under a persistent accel bias stays put when wheels report 0
  # each frame — velocity process noise keeps the pin authoritative so P[v]
  # doesn't collapse (the "stopped but keeps drifting incl. vertical" fix).
  st = PoseEKF.init()
  bias = _ACC_REST + np.array([0.05, 0.0, 0.05], np.float32)   # x + z accel bias
  for k in range(300):
    st.predict(bias, 1/150, _I3)                                # ~2 s of IMU at 150 Hz
    if k % 10 == 0: st.update_with_velocity(0.0, 0.0, 0.0)      # wheels: stopped @ ~15 Hz
  _check("stopped stays bounded under accel bias", float(np.linalg.norm(st.p_w)) < 0.3,
         f"p={st.p_w.round(3).tolist()}")
  P_eig = float(np.linalg.eigvalsh(st.P + 1e-9*np.eye(ERR_DIM)).min())
  _check("P remains positive definite", P_eig > -1e-5, f"min eig={P_eig:.2e}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
  test_predict()
  test_position_update()
  test_yaw_update()
  test_velocity_update()
  print()
  if _failures:
    print(f"{len(_failures)} FAILURE(S): {_failures}")
    sys.exit(1)
  print("all good")
