"""Linear algebra and Lie-group helpers for the SLAM stack, in tinygrad.

Tinygrad already provides `Tensor.qr` (batched Householder) and `Tensor.svd`
(Jacobi). What's missing for square-root MSCKF + IMU pre-integration:
triangular solves, SO(3) exp/log + right Jacobians, quaternion ops, small
3x3 helpers. Everything here is batched on leading dims and pure tinygrad.
"""
from tinygrad import Tensor
from tinygrad.dtype import dtypes

EPS = 1e-8

# ---------------------------------------------------------------------------
# small dense helpers (3x3)
# ---------------------------------------------------------------------------

def det3(M:Tensor) -> Tensor:
  """Determinant of a (..., 3, 3) tensor."""
  a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
  d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
  g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
  return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

def inv3(M:Tensor) -> Tensor:
  """Inverse of a (..., 3, 3) tensor via cofactor / determinant."""
  a, b, c = M[..., 0, 0:1], M[..., 0, 1:2], M[..., 0, 2:3]
  d, e, f = M[..., 1, 0:1], M[..., 1, 1:2], M[..., 1, 2:3]
  g, h, i = M[..., 2, 0:1], M[..., 2, 1:2], M[..., 2, 2:3]
  c00, c01, c02 =  (e*i - f*h), -(d*i - f*g),  (d*h - e*g)
  c10, c11, c12 = -(b*i - c*h),  (a*i - c*g), -(a*h - b*g)
  c20, c21, c22 =  (b*f - c*e), -(a*f - c*d),  (a*e - b*d)
  row0 = c00.cat(c10, c20, dim=-1)
  row1 = c01.cat(c11, c21, dim=-1)
  row2 = c02.cat(c12, c22, dim=-1)
  adj = row0.stack(row1, row2, dim=-2)
  d_ = det3(M).unsqueeze(-1).unsqueeze(-1)
  return adj / d_

# ---------------------------------------------------------------------------
# triangular solves (back/forward substitution)
# ---------------------------------------------------------------------------

def solve_upper_triangular(R:Tensor, b:Tensor) -> Tensor:
  """Solve R @ x = b for upper-triangular R. R: (..., n, n), b: (..., n) or (..., n, k)."""
  n = int(R.shape[-1])
  squeeze = b.ndim < R.ndim
  if squeeze: b = b.unsqueeze(-1)
  rows = [None]*n
  for i in range(n-1, -1, -1):
    acc = b[..., i:i+1, :]
    for j in range(i+1, n): acc = acc - R[..., i:i+1, j:j+1] * rows[j]
    rows[i] = acc / R[..., i:i+1, i:i+1]
  x = rows[0].cat(*rows[1:], dim=-2)
  return x.squeeze(-1) if squeeze else x

def solve_lower_triangular(L:Tensor, b:Tensor) -> Tensor:
  """Solve L @ x = b for lower-triangular L. L: (..., n, n), b: (..., n) or (..., n, k)."""
  n = int(L.shape[-1])
  squeeze = b.ndim < L.ndim
  if squeeze: b = b.unsqueeze(-1)
  rows = [None]*n
  for i in range(n):
    acc = b[..., i:i+1, :]
    for j in range(i): acc = acc - L[..., i:i+1, j:j+1] * rows[j]
    rows[i] = acc / L[..., i:i+1, i:i+1]
  x = rows[0].cat(*rows[1:], dim=-2)
  return x.squeeze(-1) if squeeze else x

# ---------------------------------------------------------------------------
# so(3) / SO(3)
# ---------------------------------------------------------------------------

def skew(v:Tensor) -> Tensor:
  """3-vector -> 3x3 skew-symmetric. v: (..., 3) -> (..., 3, 3)."""
  zero = Tensor.zeros_like(v[..., :1])
  x, y, z = v[..., 0:1], v[..., 1:2], v[..., 2:3]
  row0 = zero.cat(-z, y, dim=-1)
  row1 = z.cat(zero, -x, dim=-1)
  row2 = (-y).cat(x, zero, dim=-1)
  return row0.stack(row1, row2, dim=-2)

def vee(W:Tensor) -> Tensor:
  """Inverse of skew: 3x3 skew -> 3-vector. Assumes input is skew-symmetric."""
  return W[..., 2, 1:2].cat(W[..., 0, 2:3], W[..., 1, 0:1], dim=-1)

def _eye3_like(v:Tensor) -> Tensor:
  return Tensor.eye(3, dtype=v.dtype).to(v.device).expand(*v.shape[:-1], 3, 3)

def so3_exp(phi:Tensor) -> Tensor:
  """Rodrigues: tangent vector -> rotation matrix. phi: (..., 3) -> (..., 3, 3).

  Numerically stable near phi=0: the standard formula's 1/theta divisions are
  cancelled by the leading sin(theta) / (1-cos(theta)) factors. We use the
  small-angle Taylor expansion of those factors to avoid the indeterminacy.
  """
  theta_sq = phi.square().sum(-1, keepdim=True)            # (..., 1)
  theta = theta_sq.add(EPS).sqrt()                          # (..., 1)
  # coefficients a = sin(t)/t, b = (1-cos(t))/t^2 — both smooth at 0
  small = theta_sq < 1e-8
  a = small.where(1 - theta_sq/6,           theta.sin() / theta)
  b = small.where(0.5 - theta_sq/24,        (1 - theta.cos()) / theta_sq)
  K = skew(phi)                                             # (..., 3, 3)
  I = _eye3_like(phi)
  return I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * (K @ K)

def so3_log(R:Tensor) -> Tensor:
  """Inverse of Rodrigues: rotation matrix -> tangent. R: (..., 3, 3) -> (..., 3)."""
  trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
  cos_t = ((trace - 1) * 0.5).clamp(-1.0 + EPS, 1.0 - EPS)
  theta = cos_t.acos().unsqueeze(-1)                        # (..., 1)
  # vee(R - R^T) gives 2 sin(theta) * axis
  diff = R - R.transpose(-2, -1)
  v = diff[..., 2, 1:2].cat(diff[..., 0, 2:3], diff[..., 1, 0:1], dim=-1)
  # phi = theta * axis. For small theta, theta/(2 sin theta) -> 0.5 + theta^2/12 + ...
  small = theta.square() < 1e-8
  coeff = small.where(0.5 + theta.square()/12, theta / (2 * theta.sin() + EPS))
  return coeff * v

def so3_right_jacobian(phi:Tensor) -> Tensor:
  """Right Jacobian J_r of SO(3). phi: (..., 3) -> (..., 3, 3).

  J_r(phi) = I - ((1 - cos t)/t^2) K + ((t - sin t)/t^3) K^2, K = skew(phi).
  Series-expanded near 0 for stability.
  """
  theta_sq = phi.square().sum(-1, keepdim=True)
  theta = theta_sq.add(EPS).sqrt()
  small = theta_sq < 1e-8
  c1 = small.where(0.5 - theta_sq/24,           (1 - theta.cos()) / theta_sq)
  c2 = small.where((1/6.0) - theta_sq/120,      (theta - theta.sin()) / (theta_sq * theta))
  K = skew(phi)
  I = _eye3_like(phi)
  return I - c1.unsqueeze(-1) * K + c2.unsqueeze(-1) * (K @ K)

def so3_right_jacobian_inv(phi:Tensor) -> Tensor:
  """Inverse right Jacobian J_r^-1. phi: (..., 3) -> (..., 3, 3).

  J_r^-1 = I + 0.5 K + (1/t^2 - (1+cos t)/(2 t sin t)) K^2.
  Series near 0: (1/t^2 - (1+cos t)/(2 t sin t)) -> 1/12 + t^2/720 + ...
  """
  theta_sq = phi.square().sum(-1, keepdim=True)
  theta = theta_sq.add(EPS).sqrt()
  small = theta_sq < 1e-8
  long = 1.0/theta_sq - (1 + theta.cos()) / (2 * theta * theta.sin() + EPS)
  c = small.where((1/12.0) + theta_sq/720, long)
  K = skew(phi)
  I = _eye3_like(phi)
  return I + 0.5 * K + c.unsqueeze(-1) * (K @ K)

# ---------------------------------------------------------------------------
# quaternions [w, x, y, z] (Hamilton, scalar-first)
# ---------------------------------------------------------------------------

def quat_normalize(q:Tensor) -> Tensor:
  return q / q.square().sum(-1, keepdim=True).add(EPS).sqrt()

def quat_mul(p:Tensor, q:Tensor) -> Tensor:
  """Hamilton product of scalar-first quaternions. p, q: (..., 4) -> (..., 4)."""
  pw, px, py, pz = p[..., 0:1], p[..., 1:2], p[..., 2:3], p[..., 3:4]
  qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
  w = pw*qw - px*qx - py*qy - pz*qz
  x = pw*qx + px*qw + py*qz - pz*qy
  y = pw*qy - px*qz + py*qw + pz*qx
  z = pw*qz + px*qy - py*qx + pz*qw
  return w.cat(x, y, z, dim=-1)

def quat_conj(q:Tensor) -> Tensor:
  return q[..., 0:1].cat(-q[..., 1:2], -q[..., 2:3], -q[..., 3:4], dim=-1)

def quat_to_R(q:Tensor) -> Tensor:
  """Scalar-first quaternion -> rotation matrix. q: (..., 4) -> (..., 3, 3)."""
  w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
  ww, xx, yy, zz = w*w, x*x, y*y, z*z
  wx, wy, wz = w*x, w*y, w*z
  xy, xz, yz = x*y, x*z, y*z
  r00, r01, r02 = ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)
  r10, r11, r12 = 2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)
  r20, r21, r22 = 2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz
  row0 = r00.cat(r01, r02, dim=-1)
  row1 = r10.cat(r11, r12, dim=-1)
  row2 = r20.cat(r21, r22, dim=-1)
  return row0.stack(row1, row2, dim=-2)

def quat_exp_so3(phi:Tensor) -> Tensor:
  """tangent -> unit quaternion (scalar-first). phi: (..., 3) -> (..., 4)."""
  theta_sq = phi.square().sum(-1, keepdim=True)
  theta = theta_sq.add(EPS).sqrt()
  half = theta * 0.5
  small = theta_sq < 1e-8
  # sin(half)/theta -> 0.5 - theta^2/48 near 0
  s_over_theta = small.where(0.5 - theta_sq/48, half.sin() / theta)
  w = half.cos()                                              # (..., 1)
  xyz = s_over_theta * phi                                    # (..., 3)
  return w.cat(xyz, dim=-1)
