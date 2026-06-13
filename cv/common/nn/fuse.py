from abc import ABC, abstractmethod

from tinygrad import nn

class FusedBlock(ABC):
  fused = False

  @abstractmethod
  def fuse(self) -> bool:
    was_fused = self.fused
    self.fused = True
    return was_fused

  @staticmethod
  def fuse_conv2d_bn(conv, bn):
    w = bn.weight / (bn.running_var + bn.eps).sqrt()
    b = bn.bias - w * bn.running_mean

    w = conv.weight * w.reshape(-1, 1, 1, 1)

    if conv.bias is not None:
      b += w * conv.bias

    c = nn.Conv2d(
      conv.weight.shape[1] * conv.groups,
      conv.weight.shape[0],
      conv.kernel_size,
      conv.stride,
      conv.padding,
      conv.dilation,
      conv.groups,
      bias=True
    )
    c.weight.replace(w.cast(c.weight.dtype))
    assert c.bias is not None
    c.bias.replace(b.cast(c.bias.dtype))
    return c

  @staticmethod
  def fuse_bn_conv2d_pw(bn, conv):
    assert conv.bias is None, "conv must not have bias"

    w = bn.weight / (bn.running_var + bn.eps).sqrt()
    b = bn.bias - w * bn.running_mean

    w = conv.weight * w.reshape(1, -1, 1, 1)

    b = b @ conv.weight.squeeze(-1).squeeze(-1).T

    c = nn.Conv2d(
      conv.weight.shape[1] * conv.groups,
      conv.weight.shape[0],
      conv.kernel_size,
      conv.stride,
      conv.padding,
      conv.dilation,
      conv.groups,
      bias=True
    )
    c.weight.replace(w.cast(c.weight.dtype))
    assert c.bias is not None
    c.bias.replace(b.cast(c.bias.dtype))
    return c
