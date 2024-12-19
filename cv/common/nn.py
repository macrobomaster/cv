from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import prod
from tinygrad import nn

class BatchNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Tensor | None = Tensor.ones(sz, dtype=dtypes.float32) if affine else None
    self.bias: Tensor | None = Tensor.zeros(sz, dtype=dtypes.float32) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype='long' if is_dtype_supported(dtypes.long) else 'int', requires_grad=False)
    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(sz, dtype=dtypes.float32, requires_grad=False), Tensor.ones(sz, dtype=dtypes.float32, requires_grad=False)

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
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(dtypes.float32))
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * xd.numel()/(xd.numel()-xd.shape[1]) * batch_var.detach().cast(dtypes.float32))
      self.num_batches_tracked += 1
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt()).cast(x.dtype)

class AllNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Tensor | None = Tensor.ones(sz) if affine else None
    self.bias: Tensor | None = Tensor.zeros(sz) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype='long' if is_dtype_supported(dtypes.long) else 'int', requires_grad=False)
    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(1, requires_grad=False), Tensor.ones(1, requires_grad=False)

  def calc_stats(self, x:Tensor) -> tuple[Tensor, Tensor]:
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    if self.track_running_stats and not Tensor.training: return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
    batch_mean = x.mean()
    y = (x - batch_mean.detach().reshape(shape=shape_mask))
    batch_var = (y*y).mean()
    return batch_mean, batch_var

  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = self.calc_stats(x)
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(x.shape)/(prod(x.shape)-x.shape[1]) * batch_var.detach())
      self.num_batches_tracked += 1
    # reshape and expand batch_mean and batch_var
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    batch_mean = batch_mean.reshape(shape=shape_mask).expand(list(x.shape[i] if shape_mask[i] == -1 else 1 for i in range(x.ndim)))
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt())

class ConvNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, groups:int=1, dilation:int=1, bias:bool=False):
    self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=bias)
    self.n = BatchNorm(out_channels)
  def __call__(self, x:Tensor) -> Tensor: return self.n(self.c(x))

class SE:
  def __init__(self, dim:int, cmid:int):
    self.cv1 = nn.Conv2d(dim, cmid, kernel_size=1, bias=False)
    self.cv2 = nn.Conv2d(cmid, dim, kernel_size=1, bias=False)
  def __call__(self, x: Tensor):
    xx = x.mean((2, 3), keepdim=True)
    xx = self.cv1(xx).relu()
    xx = self.cv2(xx).sigmoid()
    return x * xx

class Attention:
  def __init__(self, dim:int, qk_dim:int, heads:int=8):
    self.dim, self.qk_dim, self.heads = dim, qk_dim, heads
    self.qkv = nn.Linear(dim, qk_dim * 2 + dim)
    self.out = nn.Linear(dim, dim)

  def __call__(self, x:Tensor) -> Tensor:
    b, t, c = x.shape
    q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
    q = q.reshape(b, t, self.heads, self.qk_dim // self.heads).transpose(1, 2)
    k = k.reshape(b, t, self.heads, self.qk_dim // self.heads).transpose(1, 2)
    v = v.reshape(b, t, self.heads, c // self.heads).transpose(1, 2)
    attn = Tensor.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, c)
    return self.out(attn)
