import math

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

def channel_shuffle(x: Tensor, r:int=2) -> list[Tensor]:
  b, c, h, w = x.shape
  assert c % 4 == 0
  x = x.reshape(b * c // r, r, h * w).permute(1, 0, 2)
  x = x.reshape(r, -1, c // r, h, w)
  return list(x[i] for i in range(r))

def pixel_unshuffle(x:Tensor, factor:int) -> Tensor:
  b, c, h, w = x.shape
  oc, oh, ow = c*(factor*factor), h//factor, w//factor
  x = x.reshape(b, c, oh, factor, ow, factor)
  x = x.permute(0, 1, 3, 5, 2, 4)
  return x.reshape(b, oc, oh, ow)

def pixel_shuffle(x:Tensor, factor: int) -> Tensor:
  b, c, h, w = x.shape
  oc, oh, ow = c//(factor*factor), h*factor, w*factor
  x = x.reshape(b, oc, factor, factor, h, w)
  x = x.permute(0, 1, 4, 2, 5, 3)
  return x.reshape(b, oc, oh, ow)

def upsample(x:Tensor, scale:int) -> Tensor:
  bs, c, py, px = x.shape
  return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale, px, scale).reshape(bs, c, py * scale, px * scale)

def symlog(x:Tensor) -> Tensor:
  return x.sign() * x.abs().add(1).log()
def symexp(x:Tensor) -> Tensor:
  return x.sign() * x.abs().exp().sub(1)

twohot_bins = {}
def twohot(x:Tensor, bins:int, low:float, high:float) -> Tensor:
  global twohot_bins
  if bins not in twohot_bins:
    twohot_bins[bins, low, high] = Tensor.linspace(low, high, bins)
  buckets = twohot_bins[bins, low, high]

  below = (buckets <= x[..., None]).cast(dtypes.int32).sum(-1) - 1
  above = bins - (buckets > x[..., None]).cast(dtypes.int32).sum(-1)

  below = below.clamp(0, bins - 1)
  above = above.clamp(0, bins - 1)

  equal = below == above
  to_below = equal.where(1, (buckets[below] - x).abs())
  to_above = equal.where(1, (buckets[above] - x).abs())

  total = to_below + to_above
  w_below = to_above / total
  w_above = to_below / total

  return below.one_hot(bins) * w_below[..., None] + above.one_hot(bins) * w_above[..., None]

def twohot_loss(logits:Tensor, y:Tensor, bins:int, low:float, high:float) -> Tensor:
  target = twohot(y, bins, low, high)
  loss = -logits.log_softmax(-1).mul(target).sum(-1)
  return loss.mean()

def masked_twohot_uncertainty_loss(logits:Tensor, log_var:Tensor, y:Tensor, mask:Tensor, bins:int, low:float, high:float) -> Tensor:
  target = twohot(y, bins, low, high)
  # cross entropy
  loss = -logits.log_softmax(-1).mul(target).sum(-1)
  # aleatoric uncertainty as a gaussian
  loss = log_var.neg().exp() * loss + log_var
  # mean across outputs
  loss = loss.mean(-1)
  # masking
  return mask.where(loss, 0).sum() / mask.cast(dtypes.int32).sum().add(1e-6)

def focal_loss(pred:Tensor, y:Tensor, alpha:float=0.25, gamma:float=2) -> Tensor:
  p, ce = pred.sigmoid(), pred.binary_crossentropy_logits(y, reduction="none")
  pt = p * y + (1 - p) * (1 - y)
  alpha_ = y * alpha + (1 - y) * (1 - alpha)
  loss = ce * ((1 - pt) ** gamma) * alpha_
  return loss.mean()

def mal_loss(pred:Tensor, y:Tensor, quality:Tensor, gamma:float=2) -> Tensor:
  target = y.where(quality.detach() ** gamma, 0)
  ce = pred.binary_crossentropy_logits(target, reduction="none")
  loss = ce * y.where(1, pred.sigmoid().detach() ** gamma)
  return loss.mean()

def masked_cross_entropy(pred:Tensor, y:Tensor, mask:Tensor, reduction:str="mean") -> Tensor:
  assert reduction == "mean", "only mean reduction is supported"
  ce = pred.cross_entropy(y, reduction="none")
  return mask.where(ce, 0).sum() / mask.cast(dtypes.int32).sum().add(1e-6)

def norm(x:Tensor, axis:int|None=None, keepdim:bool=False) -> Tensor:
  return x.square().sum(axis, keepdim=keepdim).sqrt()

def rms_norm(x:Tensor, axis:int|None=-1, eps:float=1e-6) -> Tensor:
  return x * x.square().mean(axis, keepdim=True).add(eps).rsqrt()

def telu(x:Tensor) -> Tensor:
  return x * x.exp().tanh()

def log_gaussian_pdf(y:Tensor, mu:Tensor, log_var:Tensor) -> Tensor:
  return -0.5 * log_var - 0.5 * math.log(2 * math.pi) - 0.5 * y.unsqueeze(1).sub(mu).square().div(log_var.exp())

def masked_mdn_loss(y:Tensor, mu:Tensor, log_var:Tensor, pi:Tensor, temp:Tensor, mask:Tensor, entropy_reg:float=0.02) -> Tensor:
  log_prob = log_gaussian_pdf(y, mu, log_var)

  # apply temperature
  pi = pi / temp
  log_pi = pi.log_softmax().unsqueeze(-1)

  loss = Tensor.logsumexp(log_prob + log_pi, axis=1).sum(-1).neg()

  # entropy regularization
  if entropy_reg > 0:
    entropy = pi.softmax().mul(log_pi.squeeze(-1)).sum(-1).neg()
    loss = loss - entropy_reg * entropy

  return mask.where(loss, 0).sum() / mask.cast(dtypes.int32).sum().add(1e-6)

def hinge_discriminator_loss(logits_real:Tensor, logits_fake:Tensor) -> Tensor:
  real_loss = (1.0 - logits_real).relu().mean()
  fake_loss = (1.0 + logits_fake).relu().mean()
  return 0.5 * (real_loss + fake_loss)

def _dft_matrix(n:int) -> tuple[Tensor, Tensor]:
  x = Tensor.arange(n, dtype=dtypes.float32).reshape(n, 1)
  y = Tensor.arange(n, dtype=dtypes.float32).reshape(1, n)
  angle = (x @ y) * (-2 * math.pi / n)
  return angle.cos(), angle.sin()
def _complex_matmul(a_re:Tensor, a_im:Tensor, b_re:Tensor, b_im:Tensor) -> tuple[Tensor, Tensor]:
  c_re = a_re@b_re - a_im@b_im
  c_im = a_re@b_im + a_im@b_re
  return c_re, c_im
def dft_image(x:Tensor) -> Tensor:
  b, c, h, w = x.shape
  w_re, w_im = _dft_matrix(w)
  x_re, x_im = _complex_matmul(x, Tensor.zeros_like(x), w_re, w_im)
  w_re, w_im = _dft_matrix(h)
  x_re, x_im = _complex_matmul(w_re, w_im, x_re, x_im)

  # combine real and imaginary parts
  x = Tensor.stack(x_re, x_im, dim=1)

  # normalize
  x = x / math.sqrt(h * w)
  return x
