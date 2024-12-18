from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import channel_shuffle, pixel_unshuffle
from ..common.nn import BatchNorm, ConvNorm, Attention, SE

class TokenMixer:
  def __init__(self, dim:int, stride:int=1, attn:bool=False):
    assert not (attn and stride != 1), "attn only works with stride=1"
    assert stride in [1, 2], "stride must be 1 or 2"
    self.stride, self.has_attn = stride, attn

    if self.has_attn:
      self.attn = Attention(dim // 4, min(16, dim // 4), heads=1)
    else:
      self.conv7x7 = nn.Conv2d(dim // 4, dim // 4, 7, stride, 3, groups=dim // 4, bias=True)
    self.conv3x3 = nn.Conv2d(dim // 4, dim // 4, 3, stride, 1, groups=dim // 4, bias=True)
    self.conv2x2 = nn.Conv2d(dim // 4, dim // 4, 2, stride, 1, dilation=2, groups=dim // 4, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    x0, x1, x2, x3 = channel_shuffle(x, 4)

    if self.has_attn:
      b, c, h, w = x1.shape
      x1 = x1.flatten(2).transpose(1, 2)
      x1 = self.attn(x1).transpose(1, 2).reshape(b, c, h, w)
    else:
      x1 = self.conv7x7(x1)
    x2 = self.conv3x3(x2)
    x3 = self.conv2x2(x3)

    # shortcut
    if self.stride == 2:
      x0 = pixel_unshuffle(x0, 2)
      b, c, h, w = x0.shape
      x0 = x0.reshape(b, x1.shape[1], c // x1.shape[1], h, w)
      x0 = x0.mean(2)

    return x0.cat(x1, x2, x3, dim=1)

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0):
    self.proj = nn.Conv2d(cin, cin * 2, 1, 1, 0, bias=True)
    self.mix = nn.Conv2d(cin * 2, cin * 2, 3, 1, 1, groups=cin * 2, bias=True)
    self.out = nn.Conv2d(cin, cin if cout == 0 else cout, 1, 1, 0, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.proj(x)

    x, gate = self.mix(x).chunk(2, dim=1)
    x = x * gate.gelu()

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
    self.se = SE(cout, max(4, cout // 16))

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x).gelu()
    x = self.conv2(x)
    return self.se(x)

class Stage:
  def __init__(self, cin:int, cout:int, num_blocks:int, attn:bool=False):
    self.downsample = Downsample(cin, cout) if cin != cout else lambda x: x
    self.blocks = [Block(cout, attn=attn) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    x = self.downsample(x)
    return x.sequential(self.blocks)

class Backbone:
  def __init__(self, cin:int=3, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 5, 1]):
    self.stem = Stem(cin, cstage[0])

    self.stage0 = Stage(cstage[0], cstage[0], stages[0])
    self.stage1 = Stage(cstage[0], cstage[1], stages[1])
    self.stage2 = Stage(cstage[1], cstage[2], stages[2], attn=True)
    self.stage3 = Stage(cstage[2], cstage[3], stages[3], attn=True)

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
    xx = self.up(x).gelu()
    xx, gate = self.down(xx).chunk(2, dim=-1)
    xx = xx * gate.gelu()
    return (x + xx).gelu()

class FFN:
  def __init__(self, in_dim:int, mid_dim:int, out_dim:int, exp_dim:int=0, blocks:int=1):
    if exp_dim == 0: exp_dim = mid_dim
    self.proj = nn.Linear(in_dim, mid_dim)
    self.blocks = [FFNBlock(mid_dim, exp_dim) for _ in range(blocks)]
    self.out = nn.Linear(mid_dim, out_dim)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.proj(x).gelu()
    x = x.sequential(self.blocks)
    return self.out(x)

class Neck:
  def __init__(self, cins:list[int], cout:int, cmid:int=256):
    self.x0 = nn.Conv2d(cins[0], cmid, 1, 1, 0, bias=True)
    self.x1 = nn.Conv2d(cins[1], cmid, 1, 1, 0, bias=True)
    self.x2 = nn.Conv2d(cins[2], cmid, 1, 1, 0, bias=True)
    self.x3 = nn.Conv2d(cins[3], cmid, 1, 1, 0, bias=True)

    self.features = cout // cmid
    self.feature = Tensor.kaiming_normal(1, self.features, cmid)

    self.attn_norm = BatchNorm(cmid)
    self.attn = Attention(cmid, cmid, heads=4)
    self.ffn_norm = BatchNorm(cmid)
    self.ffn = FFN(cmid, cmid, cmid, exp_dim=cmid*2)

  def __call__(self, xs:tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    x0, x1, x2, x3 = xs
    x3 = self.x3(x3).mean((2, 3))
    x2 = self.x2(x2).mean((2, 3))
    x1 = self.x1(x1).mean((2, 3))
    x0 = self.x0(x0).mean((2, 3))
    x = Tensor.stack(x3, x2, x1, x0, dim=1)

    # concat with feature
    x = x.cat(self.feature.expand(x.shape[0], self.features, -1), dim=1)

    # attention
    out = x + self.attn(self.attn_norm(x.transpose(1, 2)).transpose(1, 2))

    # ffn
    out = out[:, -self.features:]
    return self.ffn(self.ffn_norm(out.transpose(1, 2)).transpose(1, 2)).flatten(1)

class Head:
  def __init__(self, in_dim:int, mid_dim:int, out_dim:int):
    self.ffn = FFN(in_dim, mid_dim, out_dim)

  def __call__(self, x:Tensor) -> Tensor:
    return self.ffn(x)

class Model:
  def __init__(self, mid:int=512, head:int=64, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 5, 1]):
    # feature extractor
    self.backbone = Backbone(cin=6, cstage=cstage, stages=stages)
    self.neck = Neck(cstage, mid)

    # heads
    self.cls_head = Head(mid, head, 2)
    self.x_head = Head(mid, head, 512)
    self.y_head = Head(mid, head, 256)
    self.dist_head = Head(mid, head, 200)

  def __call__(self, img:Tensor):
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
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
      dist = ((dist.softmax() @ Tensor.arange(200)) / 10).float()

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
