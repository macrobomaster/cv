from typing import Literal

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import make_tuple, prod, round_up
from tinygrad import nn

from .tensor import rms_norm

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
    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(1, dtype=dtypes.float32, requires_grad=False), Tensor.ones(1, dtype=dtypes.float32, requires_grad=False)

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
    return x * self.weight + self.bias

class ConvNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride:int, padding:int, groups:int=1, dilation:int=1, bias:bool=False):
    self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=bias)
    self.n = BatchNorm(out_channels)
  def __call__(self, x:Tensor) -> Tensor: return self.n(self.c(x))

class ConvTransposeNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, output_padding:int, groups:int=1, bias:bool=False):
    self.c = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups=groups, bias=bias)
    self.n = BatchNorm(out_channels)
  def __call__(self, x:Tensor) -> Tensor: return self.n(self.c(x))

class UpsampleConvNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, groups:int=1, dilation:int=1, bias:bool=False):
    self.c = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, groups=groups, dilation=dilation, bias=bias)
    self.n = BatchNorm(out_channels)
    self.scale_factor = stride
  def __call__(self, x:Tensor) -> Tensor:
    bs, c, py, px = x.shape
    x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, self.scale_factor, px, self.scale_factor).reshape(bs, c, py*self.scale_factor, px*self.scale_factor)
    return self.n(self.c(x))

class SE:
  def __init__(self, dim:int, cmid:int):
    self.cv1 = nn.Conv2d(dim, cmid, kernel_size=1, bias=False)
    self.cv2 = nn.Conv2d(cmid, dim, kernel_size=1, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    xx = x.mean((2, 3), keepdim=True)
    xx = self.cv1(xx).relu()
    xx = self.cv2(xx).sigmoid()
    return x * xx

class SRM:
  def __init__(self, dim:int):
    self.cv1 = nn.Conv2d(dim, dim, kernel_size=(1, 2), bias=False)
    self.cv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    b, c, _, _ = x.shape
    mean = x.flatten(2).mean(2).reshape(b, c, 1, 1)
    std = x.flatten(2).std(2).reshape(b, c, 1, 1)
    u = mean.cat(std, dim=3)
    xx = self.cv1(u).relu()
    xx = self.cv2(xx).sigmoid()
    return x * xx

class Attention:
  """
  Cross and Self Attention with qk-norm and configurable output modulation
  """
  def __init__(self, dim:int, qk_dim:int, heads:int, out:Literal["proj", "mod"]|None="proj"):
    assert qk_dim % heads == 0, "qk_dim must be divisible by heads"
    assert out in ["proj", "mod", None], "out must be one of 'proj', 'mod', or None"

    self.dim, self.qk_dim, self.heads = dim, qk_dim, heads
    self.q = nn.Linear(dim, qk_dim, bias=False)
    self.kv = nn.Linear(dim, qk_dim + dim, bias=False)

    self.out = out
    match out:
      case "proj":
        self.proj = nn.Linear(dim, dim, bias=False)
      case "mod":
        self.gate = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

  def __call__(self, x:Tensor, kv:Tensor|None=None) -> Tensor:
    b, t, c = x.shape
    if kv is not None: kvt = kv.shape[1]
    else: kvt = t

    q = self.q(x)
    k, v = self.kv(x if kv is None else kv).split([self.qk_dim, self.dim], dim=-1)
    q = rms_norm(q.reshape(b, t, self.heads, self.qk_dim // self.heads)).transpose(1, 2)
    k = rms_norm(k.reshape(b, kvt, self.heads, self.qk_dim // self.heads)).transpose(1, 2)
    v = v.reshape(b, kvt, self.heads, c // self.heads).transpose(1, 2)

    attn = q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(b, t, c)

    match self.out:
      case "proj":
        return self.proj(attn)
      case "mod":
        return self.proj(attn.sigmoid() * self.gate(x).sigmoid())
      case _: return attn

class RecConv:
  """
  Recursive Convolution Module

  See: https://arxiv.org/pdf/2412.19628v1
  """
  def __init__(self, dim:int, kernel_size:int, levels:int=2):
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    self.levels = levels
    self.down = nn.Conv2d(dim, dim, kernel_size, 2, kernel_size//2, groups=dim, bias=False)
    self.convs = [nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim, bias=False) for _ in range(levels + 1)]
    self.up = nn.ConvTranspose2d(dim, dim, round_up(kernel_size, 2), 2, kernel_size//2, groups=dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    features = [x]
    for _ in range(self.levels):
      features.append(self.down(features[-1]))

    x = self.convs[-1](features[-1])
    for f, conv in zip(reversed(features[:-1]), reversed(self.convs[:-1])):
      x = conv(self.up(x) + f)

    return x
