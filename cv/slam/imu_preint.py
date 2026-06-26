"""Per-sample IMU propagation primitives for the MSCKF predict step.

`msckf.State.predict` accumulates IMU samples between camera frames by
calling these functions sample-by-sample. We use sample-by-sample (not
Forster-style pre-integration) because (a) IMU is ~100 Hz not 1 kHz so the
extra QR calls are cheap and (b) the sqrt-form is cleaner without an
intermediate bias-Jacobian dance.

Conventions:
- World frame: +z up. Gravity in calib.GRAVITY.
- Body frame == IMU frame.
- Nominal state layout (16): `[q_wb(4), p_w(3), v_w(3), b_g(3), b_a(3)]`
  with q in scalar-first Hamilton convention.
- Error state layout (15):   `[δθ_b(3), δp_w(3), δv_w(3), δb_g(3), δb_a(3)]`
  with δθ_b a body-frame rotation tangent: R_true = R_nom · exp(δθ_b).
- IMU measurement model (continuous):
    gyro_meas = ω_b + b_g + n_g
    accel_meas = R_bw (a_w − g_w) + b_a + n_a
  So bias-corrected ω_b = gyro_meas − b_g, and the true world-frame linear
  acceleration is a_w = R_wb (accel_meas − b_a) + g_w.
"""
from tinygrad import Tensor

from .linalg import skew, so3_exp, quat_to_R, quat_mul, quat_exp_so3, quat_normalize

# Error-state slot offsets
TH, P, V, BG, BA = 0, 3, 6, 9, 12
DIM_ERR = 15
DIM_NOISE = 12   # [n_g(3), n_a(3), n_bg(3), n_ba(3)]

def imu_step_nominal(q_wb:Tensor, p_w:Tensor, v_w:Tensor,
                     accel:Tensor, gyro:Tensor,
                     b_g:Tensor, b_a:Tensor, g_w:Tensor, dt:float
                     ) -> tuple[Tensor, Tensor, Tensor]:
  """Midpoint-style nominal state update for one IMU sample.

  Returns (q_wb_new, p_w_new, v_w_new). Uses the rotation at the beginning
  of the interval for the linear-acceleration rotation — Forster Eq 9 (Euler
  forward). For dt ~10 ms the midpoint correction is negligible.
  """
  w_b = gyro - b_g
  a_b = accel - b_a
  R_wb = quat_to_R(q_wb)
  a_w = (R_wb @ a_b.unsqueeze(-1)).squeeze(-1) + g_w

  p_new = p_w + v_w * dt + 0.5 * a_w * (dt * dt)
  v_new = v_w + a_w * dt
  dq = quat_exp_so3(w_b * dt)
  q_new = quat_normalize(quat_mul(q_wb, dq))
  return q_new, p_new, v_new

def imu_step_phi_g(q_wb:Tensor, accel:Tensor, gyro:Tensor,
                   b_g:Tensor, b_a:Tensor, dt:float
                   ) -> tuple[Tensor, Tensor]:
  """Discrete-time state-transition Φ (15x15) and noise input G (15x12).

  Derivation (continuous F then Φ ≈ I + F dt + ½ F² dt²; we keep first-order):

      F = [[ -[ω]_×, 0, 0, -I, 0 ],
           [ 0,      0, I,  0, 0 ],
           [ -R [a]_×, 0, 0, 0, -R ],
           [ 0,      0, 0,  0, 0 ],
           [ 0,      0, 0,  0, 0 ]]

  with ω = gyro − b_g, a = accel − b_a, R = R_wb. Noise input (continuous):

      G_c = [[ -I, 0, 0, 0 ],
             [ 0,  0, 0, 0 ],
             [ 0, -R, 0, 0 ],
             [ 0,  0, I, 0 ],
             [ 0,  0, 0, I ]]   (15x12)

  Discrete G returned here multiplies the per-axis stddev vector scaled by
  √dt (caller composes √Q_d as `G @ diag([σ_g·√dt, σ_a·√dt, σ_bg·√dt, σ_ba·√dt])`).
  """
  w_b = gyro - b_g
  a_b = accel - b_a
  R_wb = quat_to_R(q_wb)
  I3 = Tensor.eye(3, dtype=q_wb.dtype).to(q_wb.device)
  Z3 = Tensor.zeros(3, 3, dtype=q_wb.dtype, device=q_wb.device)

  Wx = skew(w_b)                                   # (3, 3)
  Ra = (R_wb @ a_b.unsqueeze(-1)).squeeze(-1)      # (3,)
  Rax = skew(Ra)                                   # (3, 3)

  # Φ blocks (first-order, I + F dt)
  # Row-major: rows in order [θ, p, v, b_g, b_a].
  phi_rows = [
    [I3 - Wx*dt,         Z3,        Z3,    -I3*dt, Z3      ],
    [Z3,                  I3,       I3*dt,  Z3,    Z3      ],
    [-Rax*dt,            Z3,        I3,     Z3,    -R_wb*dt],
    [Z3,                  Z3,       Z3,     I3,    Z3      ],
    [Z3,                  Z3,       Z3,     Z3,    I3      ],
  ]
  Phi = _block(phi_rows)

  # G blocks (continuous-time noise input; caller scales by σ·√dt per axis)
  # Cols in order [n_g, n_a, n_bg, n_ba].
  g_rows = [
    [-I3*dt,  Z3,       Z3,    Z3   ],
    [Z3,      Z3,       Z3,    Z3   ],
    [Z3,      -R_wb*dt, Z3,    Z3   ],
    [Z3,      Z3,       I3,    Z3   ],
    [Z3,      Z3,       Z3,    I3   ],
  ]
  G = _block(g_rows)
  return Phi, G

def _block(rows:list[list[Tensor]]) -> Tensor:
  """Vertically/horizontally concatenate a 2D list of Tensors into a single matrix."""
  cat_rows = [r[0].cat(*r[1:], dim=-1) for r in rows]
  return cat_rows[0].cat(*cat_rows[1:], dim=-2)
