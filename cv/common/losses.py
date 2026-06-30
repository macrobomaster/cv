import math

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

from .tensor import twohot, log_gaussian_pdf

# *** base losses
def twohot_loss(logits:Tensor, y:Tensor, bins:int, low:float, high:float) -> Tensor:
  target = twohot(y, bins, low, high)
  loss = -logits.log_softmax(-1).mul(target).sum(-1)
  return loss

def focal_loss(pred:Tensor, y:Tensor, alpha:float=0.25, gamma:float=2) -> Tensor:
  p, ce = pred.sigmoid(), pred.binary_crossentropy_logits(y, reduction="none")
  pt = p * y + (1 - p) * (1 - y)
  alpha_ = y * alpha + (1 - y) * (1 - alpha)
  loss = ce * ((1 - pt) ** gamma) * alpha_
  return loss

def quality_focal_loss(pred:Tensor, y:Tensor, gamma:float=2) -> Tensor:
  # QFL (Generalized Focal Loss): soft-target sigmoid loss. The modulating factor |y - sigmoid(pred)|^gamma
  # drives the loss to 0 when the score matches the soft target exactly (classic focal_loss can't, on soft y).
  p = pred.sigmoid()
  ce = pred.binary_crossentropy_logits(y, reduction="none")
  return ce * (y - p).abs() ** gamma

def mal_loss(pred:Tensor, y:Tensor, quality:Tensor, gamma:float=2) -> Tensor:
  target = y.where(quality.detach() ** gamma, 0)
  ce = pred.binary_crossentropy_logits(target, reduction="none").contiguous().contiguous_backward()
  loss = ce * y.where(1, pred.sigmoid().detach() ** gamma)
  loss = loss.sum(-1)
  return loss

def cross_entropy(pred:Tensor, y:Tensor) -> Tensor:
  loss = pred.cross_entropy(y, reduction="none")
  return loss

def mdn_loss(y:Tensor, mu:Tensor, log_var:Tensor, pi:Tensor, temp:Tensor, entropy_reg:float=0.02) -> Tensor:
  log_prob = log_gaussian_pdf(y, mu, log_var)

  # apply temperature
  pi = pi / temp
  log_pi = pi.log_softmax().unsqueeze(-1)

  loss = Tensor.logsumexp(log_prob + log_pi, axis=1).sum(-1).neg()

  # entropy regularization
  if entropy_reg > 0:
    entropy = pi.softmax().mul(log_pi.squeeze(-1)).sum(-1).neg()
    loss = loss - entropy_reg * entropy

  return loss

def hinge_discriminator_loss(logits_real:Tensor, logits_fake:Tensor) -> Tensor:
  real_loss = (1.0 - logits_real).relu().mean()
  fake_loss = (1.0 + logits_fake).relu().mean()
  return 0.5 * (real_loss + fake_loss)

# *** loss addons
def gaussian_uncertainty(loss:Tensor, log_var:Tensor) -> Tensor:
  loss = log_var.neg().exp() * loss + log_var
  return loss

# *** loss reducers
def masked_mean(loss:Tensor, mask:Tensor) -> Tensor:
  return mask.where(loss, 0).sum() / mask.cast(dtypes.int32).sum().add(1e-6)
