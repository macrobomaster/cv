from tinygrad.tensor import Tensor
from tinygrad import nn

from ..common.nn import ConvNorm

class Discriminator:
  def __init__(self, cin:int=3):
    self.stem = [
      nn.Conv2d(cin, 64, 4, 2, 1),
      lambda x: x.leaky_relu(0.2),
    ]
    self.layers = [
      ConvNorm(64, 128, 4, 2, 1, bias=False),
      lambda x: x.leaky_relu(0.2),
      ConvNorm(128, 256, 4, 2, 1, bias=False),
      lambda x: x.leaky_relu(0.2),
    ]
    self.output = [
      ConvNorm(256, 512, 4, 1, 1, bias=False),
      lambda x: x.leaky_relu(0.2),
      nn.Conv2d(512, 1, 4, 1, 1),
    ]

  def __call__(self, x:Tensor) -> Tensor:
    x = x.sequential(self.stem)
    x = x.sequential(self.layers)
    return x.sequential(self.output)
