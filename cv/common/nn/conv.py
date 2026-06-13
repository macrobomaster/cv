from tinygrad.tensor import Tensor
from tinygrad import nn

from .fuse import FusedBlock
from .norm import RMSNorm2d
from ..tensor import upsample

class ConvNorm(FusedBlock):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride:int, padding:int, dilation:int=1, groups:int=1, bias:bool=False):
    self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
    self.n = RMSNorm2d(out_channels)

  def __call__(self, x:Tensor) -> Tensor:
    return self.n(self.c(x))

  def fuse(self) -> bool:
    # RMSNorm can't be fused into conv weights the way BN can — RMS depends on per-sample input
    # statistics, not learned running stats. No-op so callers that walk FusedBlock subclasses don't break.
    return super().fuse()

class ConvTransposeNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, output_padding:int=0, groups:int=1, bias:bool=False):
    self.c = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups=groups, bias=bias)
    self.n = RMSNorm2d(out_channels)
  def __call__(self, x:Tensor) -> Tensor: return self.n(self.c(x))

class UpsampleConv:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, groups:int=1, dilation:int=1, bias:bool=False):
    self.c = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, groups=groups, dilation=dilation, bias=bias)
    self.scale_factor = stride
  def __call__(self, x:Tensor) -> Tensor:
    if self.scale_factor != 1:
      x = upsample(x, self.scale_factor)
    return self.c(x)

class UpsampleConvNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, groups:int=1, dilation:int=1, bias:bool=False):
    self.c = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, groups=groups, dilation=dilation, bias=bias)
    self.n = RMSNorm2d(out_channels)
    self.scale_factor = stride
  def __call__(self, x:Tensor) -> Tensor:
    if self.scale_factor != 1:
      x = upsample(x, self.scale_factor)
    return self.n(self.c(x))

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

  def __call__(self, x:Tensor) -> Tensor:
    features = [x]
    for _ in range(self.levels):
      features.append(self.down(features[-1]))

    x = self.convs[-1](features[-1])
    for f, conv in zip(reversed(features[:-1]), reversed(self.convs[:-1])):
      x = conv(upsample(x, 2) + f)

    return x
