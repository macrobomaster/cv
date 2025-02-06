from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import channel_shuffle, pixel_unshuffle, telu
from ..common.nn import BatchNorm, ConvNorm, Attention, SE, SRM, RecConv

def nonlinear(x:Tensor) -> Tensor: return x.gelu()

class TokenMixer:
  def __init__(self, dim:int, stride:int=1, attn:bool=False):
    assert not (attn and stride != 1), "attn only works with stride=1"
    assert stride in [1, 2], "stride must be 1 or 2"
    self.stride, self.has_attn = stride, attn

    # complex_dim = dim // 4
    # simple_dim = (dim * 3) // 4

    complex_dim = dim // 2
    simple_dim = dim // 2

    if self.has_attn:
      self.attn = Attention(complex_dim, min(16, complex_dim), heads=1)
    else:
      if self.stride == 1:
        self.rec = RecConv(complex_dim, 5, levels=2)
      else:
        self.conv7x7 = nn.Conv2d(complex_dim, complex_dim, 7, stride, 3, groups=complex_dim, bias=False)

    self.conv3x3 = nn.Conv2d(simple_dim, simple_dim, 3, stride, 1, groups=simple_dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    # # 1/4, 3/4 channel shuffle split
    # x0, x1, x2, x3 = channel_shuffle(x, 4)
    # x1 = x1.cat(x2, x3, dim=1)

    # 1/2, 1/2 channel shuffle split
    x0, x1 = channel_shuffle(x, 2)

    # complex mixer
    if self.has_attn:
      b, c, h, w = x0.shape
      x0 = x0.flatten(2).transpose(1, 2)
      x0 = self.attn(x0).transpose(1, 2).reshape(b, c, h, w)
    else:
      if self.stride == 1:
        x0 = self.rec(x0)
      else:
        x0 = self.conv7x7(x0)

    # simple mixer
    x1 = self.conv3x3(x1)

    return x0.cat(x1, dim=1)

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0):
    self.proj = nn.Conv2d(cin, cin * 2, 1, 1, 0, bias=False)
    self.mix = nn.Conv2d(cin * 2, cin * 2, 3, 1, 1, groups=cin * 2, bias=False)
    self.out = nn.Conv2d(cin, cin if cout == 0 else cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.proj(x)

    x, gate = self.mix(x).chunk(2, dim=1)
    x = x * nonlinear(gate)

    return self.out(x)

class Block:
  def __init__(self, dim:int, attn:bool=False):
    self.tnorm = BatchNorm(dim)
    self.token_mixer = TokenMixer(dim, attn=attn)
    self.cnorm = BatchNorm(dim)
    self.channel_mixer = ChannelMixer(dim)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.token_mixer(self.tnorm(x))
    x = x + xx
    xx = self.channel_mixer(self.cnorm(xx))
    return x + xx

class Downsample:
  def __init__(self, cin:int, cout:int):
    self.cnorm = BatchNorm(cin)
    self.channel_mixer = ChannelMixer(cin, cout)
    self.tnorm = BatchNorm(cout)
    self.token_mixer = TokenMixer(cout, stride=2)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.channel_mixer(self.cnorm(x))
    xx = self.token_mixer(self.tnorm(xx))

    # shortcut
    x = pixel_unshuffle(x, 2)
    b, c, h, w = x.shape
    x = x.reshape(b, xx.shape[1], c // xx.shape[1], h, w)
    x = x.mean(2)

    return x + xx

class Stem:
  def __init__(self, cin:int, cout:int):
    self.conv1 = ConvNorm(cin, cout // 2, 3, 2, 1, bias=False)
    self.conv2 = ConvNorm(cout // 2, cout, 3, 2, 1, bias=False)
    # self.se = SE(cout, max(4, cout // 16))
    self.srm = SRM(cout)

  def __call__(self, x: Tensor) -> Tensor:
    x = nonlinear(self.conv1(x))
    x = self.conv2(x)
    return self.srm(x)

class Stage:
  def __init__(self, cin:int, cout:int, num_blocks:int, attn:bool=False):
    self.downsample = Downsample(cin, cout) if cin != cout else lambda x: x
    self.blocks = [Block(cout, attn=attn) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    x = self.downsample(x)
    return x.sequential(self.blocks)

class Backbone:
  def __init__(self, cin:int, cstage:list[int], stages:list[int], attn:bool):
    self.stem = Stem(cin, cstage[0])

    self.stage0 = Stage(cstage[0], cstage[0], stages[0])
    self.stage1 = Stage(cstage[0], cstage[1], stages[1])
    self.stage2 = Stage(cstage[1], cstage[2], stages[2], attn=attn)
    self.stage3 = Stage(cstage[2], cstage[3], stages[3], attn=attn)

  def __call__(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    x = self.stem(x)

    x0 = self.stage0(x)
    x1 = self.stage1(x0)
    x2 = self.stage2(x1)
    x3 = self.stage3(x2)

    return x0, x1, x2, x3

class FFNBlock:
  def __init__(self, dim:int, exp_dim:int):
    self.up = nn.Linear(dim, exp_dim)
    self.down = nn.Linear(exp_dim, dim * 2)
  def __call__(self, x:Tensor) -> Tensor:
    xx = nonlinear(self.up(x))
    xx, gate = self.down(xx).chunk(2, dim=-1)
    xx = xx * nonlinear(gate)
    return x + xx

class FFN:
  def __init__(self, in_dim:int, out_dim:int, mid_dim:int, exp_dim:int=0, blocks:int=1):
    if exp_dim == 0: exp_dim = mid_dim
    self.proj = nn.Linear(in_dim, mid_dim)
    self.blocks = [FFNBlock(mid_dim, exp_dim) for _ in range(blocks)]
    self.out = nn.Linear(mid_dim, out_dim)

  def __call__(self, x:Tensor) -> Tensor:
    x = nonlinear(self.proj(x))
    x = x.sequential(self.blocks)
    return self.out(x)

class Neck:
  def __init__(self, cins:list[int], cout:int, cmid:int=256):
    # self.x0norm = BatchNorm(cins[0])
    # self.x0 = ChannelMixer(cins[0], cmid)
    # self.x1norm = BatchNorm(cins[1])
    # self.x1 = ChannelMixer(cins[1], cmid)
    # self.x2norm = BatchNorm(cins[2])
    # self.x2 = ChannelMixer(cins[2], cmid)
    # self.x3norm = BatchNorm(cins[3])
    # self.x3 = ChannelMixer(cins[3], cmid)
    self.x2 = ConvNorm(cins[2], cmid, 1, 1, 0, bias=False)
    self.x3 = ConvNorm(cins[3], cmid, 1, 1, 0, bias=False)

    self.out_norm = BatchNorm(cmid*2)
    self.out = FFN(cmid*2, cout, cmid, exp_dim=cmid*2, blocks=2)

  def __call__(self, xs:tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    x0, x1, x2, x3 = xs
    # x0 = self.x0(self.x0norm(x0)).mean((2, 3))
    # x1 = self.x1(self.x1norm(x1)).mean((2, 3))
    # x2 = self.x2(self.x2norm(x2)).mean((2, 3))
    # x3 = self.x3(self.x3norm(x3)).mean((2, 3))
    x2 = self.x2(x2).mean((2, 3))
    x3 = self.x3(x3).mean((2, 3))

    x = x2.cat(x3, dim=1)

    # project out
    return self.out(self.out_norm(x))

class Head:
  def __init__(self, in_dim:int, mid_dim:int, out_dim:int):
    self.ffn = FFN(in_dim, mid_dim, out_dim, blocks=1)

  def __call__(self, x:Tensor) -> Tensor:
    return self.ffn(x)

class Model:
  def __init__(self, mid:int=256, head:int=64, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 6, 2]):
    # feature extractor
    self.backbone = Backbone(cin=6, cstage=cstage, stages=stages, attn=True)
    self.neck = Neck(cstage, mid)

    # heads
    self.cls_head = Head(mid, 2, head)
    self.x_head = Head(mid, 512, head)
    self.y_head = Head(mid, 256, head)
    self.dist_head = Head(mid, 256, head)

  def __call__(self, img:Tensor):
    # image normalization
    img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    xs = self.backbone(img)
    f = self.neck(xs)

    cl = self.cls_head(f),
    x = self.x_head(f)
    y = self.y_head(f)
    dist = self.dist_head(f)

    if not Tensor.training:
      cl = (cl[0].sigmoid().argmax(1), cl[0].sigmoid()[:, cl[0].argmax(1)])
      x = (x.softmax() @ Tensor.arange(512)).float()
      y = (y.softmax() @ Tensor.arange(256)).float()
      dist = ((dist.softmax() @ Tensor.arange(256)) / 16).float()

    return *cl, x, y, dist

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad.helpers import GlobalCounters
  from tinygrad.engine.jit import TinyJit
  from functools import partial

  model = Model()
  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")

  @partial(TinyJit, prune=True)
  def run(x:Tensor):
    return model(x)

  run(Tensor.randn(1, 128, 256, 6))
  GlobalCounters.reset()
  run(Tensor.randn(1, 128, 256, 6))
  GlobalCounters.reset()
  run(Tensor.randn(1, 128, 256, 6))

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
