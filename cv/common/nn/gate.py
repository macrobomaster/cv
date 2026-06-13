from tinygrad.tensor import Tensor
from tinygrad import nn

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

class RefineGate:
  """
  Refine Gate Module

  See: https://arxiv.org/pdf/1910.09890
  """
  def __init__(self, dim:int):
    self.refine = nn.Linear(dim, dim, bias=False)
    self.gate = nn.Linear(dim, dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    r = self.refine(x).sigmoid().mul(2).sub(1)
    g = self.gate(x).sigmoid()
    phi = r * g * (1 - g)
    gate = g + phi
    return x * gate

class RefineGate2d:
  """
  Refine Gate Module for 2D Tensors

  See: https://arxiv.org/pdf/1910.09890
  """
  def __init__(self, dim:int):
    self.refine = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
    self.gate = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    r = self.refine(x).sigmoid().mul(2).sub(1)
    g = self.gate(x).sigmoid()
    phi = r * g * (1 - g)
    gate = g + phi
    return x * gate
