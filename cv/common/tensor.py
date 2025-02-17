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
def twohot(x:Tensor, bins:int) -> Tensor:
  global twohot_bins

  k0 = x.floor().clamp(0, bins-1)
  k1 = x.ceil().clamp(0, bins-1)

  equal = k0 == k1
  to_below = equal.where(1, (k0 - x).abs())
  to_above = equal.where(0, (k1 - x).abs())

  total = to_below + to_above
  w_below = to_above / total
  w_above = to_below / total

  if bins not in twohot_bins:
    twohot_bins[bins] = Tensor.arange(bins).reshape(1, bins)
  ar = twohot_bins[bins].expand(x.shape[0], bins)
  th = (ar == k0.reshape(x.shape[0], 1)).where(w_below.reshape(x.shape[0], 1), 0)
  th = th + (ar == k1.reshape(x.shape[0], 1)).where(w_above.reshape(x.shape[0], 1), 0)
  return th

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
