from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import make_tuple, prod

class BatchNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Tensor | None = Tensor.ones(sz, dtype=dtypes.float32) if affine else None
    self.bias: Tensor | None = Tensor.zeros(sz, dtype=dtypes.float32) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype=dtypes.long).is_param_(False)
    if track_running_stats:
      self.running_mean = Tensor.zeros(sz, dtype=dtypes.float32).is_param_(False)
      self.running_var = Tensor.ones(sz, dtype=dtypes.float32).is_param_(False)

  def calc_stats(self, x:Tensor) -> tuple[Tensor, Tensor]:
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    if self.track_running_stats and not Tensor.training: return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
    batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x != 1)))
    y = (x - batch_mean.detach().reshape(shape=shape_mask))  # d(var)/d(mean) = 0
    batch_var = (y*y).mean(axis=reduce_axes)
    return batch_mean, batch_var

  def __call__(self, x:Tensor) -> Tensor:
    xd = x.cast(dtypes.float32)
    batch_mean, batch_var = self.calc_stats(xd)
    # NOTE: wow, this is done all throughout training in most PyTorch models
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(self.running_mean.dtype))
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * xd.numel()/(xd.numel()-xd.shape[1]) * batch_var.detach().cast(self.running_var.dtype))
      self.num_batches_tracked += 1
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt()).cast(x.dtype)

class AllNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Tensor | None = Tensor.ones(sz) if affine else None
    self.bias: Tensor | None = Tensor.zeros(sz) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype=dtypes.long).is_param_(False)
    if track_running_stats:
      self.running_mean = Tensor.zeros(1, dtype=dtypes.float32).is_param_(False)
      self.running_var = Tensor.ones(1, dtype=dtypes.float32).is_param_(False)

  def calc_stats(self, x:Tensor) -> tuple[Tensor, Tensor]:
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    if self.track_running_stats and not Tensor.training: return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
    batch_mean = x.mean()
    y = (x - batch_mean.detach().reshape(shape=shape_mask))
    batch_var = (y*y).mean()
    return batch_mean, batch_var

  def __call__(self, x:Tensor) -> Tensor:
    xd = x.cast(dtypes.float32)
    batch_mean, batch_var = self.calc_stats(xd)
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(xd.shape)/(prod(xd.shape)-xd.shape[1]) * batch_var.detach())
      self.num_batches_tracked += 1
    # reshape and expand batch_mean and batch_var
    shape_mask: list[int] = [1, -1, *([1]*(xd.ndim-2))]
    batch_mean = batch_mean.reshape(shape=shape_mask).expand(list(xd.shape[i] if shape_mask[i] == -1 else 1 for i in range(xd.ndim)))
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt()).cast(x.dtype)

class LayerNorm:
  def __init__(self, normalized_shape:int|tuple[int, ...], eps=1e-5, elementwise_affine=True):
    self.normalized_shape: tuple[int, ...] = make_tuple(normalized_shape, 1)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight: Tensor|None = Tensor.ones(*self.normalized_shape) if elementwise_affine else None
    self.bias: Tensor|None = Tensor.zeros(*self.normalized_shape) if elementwise_affine else None

  def __call__(self, x:Tensor) -> Tensor:
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    x = x.cast(dtypes.float32).layernorm(eps=self.eps, axis=self.axis).cast(x.dtype)
    if not self.elementwise_affine: return x
    assert self.weight is not None and self.bias is not None
    return x * self.weight + self.bias

class RMSNorm:
  def __init__(self, dim:int, eps:float=1e-6, elementwise_affine:bool=True):
    self.dim, self.eps = dim, eps
    self.weight: Tensor|None = Tensor.ones(dim) if elementwise_affine else None

  def __call__(self, x:Tensor) -> Tensor:
    sq_sum = x.square().sum(-1, keepdim=True, dtype=dtypes.float32)
    rms = (sq_sum.div(self.dim) + self.eps).rsqrt().cast(x.dtype)
    x = x * rms
    return x if self.weight is None else x * self.weight

class RMSNorm2d(RMSNorm):
  def __call__(self, x: Tensor) -> Tensor: return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class GRN:
  """
  Global Response Normalization.
  """
  def __init__(self, dim:int, eps:float=1e-6):
    self.eps = eps
    self.gamma = Tensor.zeros(dim, dtype=dtypes.float32)
    self.beta = Tensor.zeros(dim, dtype=dtypes.float32)

  def __call__(self, x:Tensor) -> Tensor:
    gx = x.square().sum((2, 3), keepdim=True, dtype=dtypes.float32).add(self.eps).sqrt()
    nx = (gx / (gx.mean(axis=1, keepdim=True) + self.eps)).cast(x.dtype)
    gamma = self.gamma.cast(x.dtype).reshape(1, -1, 1, 1)
    beta = self.beta.cast(x.dtype).reshape(1, -1, 1, 1)
    return gamma * (x * nx) + beta + x
