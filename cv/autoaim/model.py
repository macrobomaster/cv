from typing import cast
from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import BatchNorm, ConvNorm, Attention, SE

def nonlinear(x:Tensor) -> Tensor: return x.gelu()

class TokenMixer:
  def __init__(self, dim:int, attn:bool=False, sideband:int=0):
    assert sideband == 0 or attn, "attn required for sideband"
    self.has_attn = attn
    self.sideband = sideband

    if self.has_attn:
      self.attn = Attention(dim, dim // 4, heads=1, out="mod")
    else:
      self.conv7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)

  def __call__(self, x:Tensor, sb:Tensor|None=None) -> Tensor | tuple[Tensor, Tensor]:
    if self.has_attn:
      b, c, h, w = x.shape

      x = x.flatten(2).transpose(1, 2)
      if sb is not None:
        x = x.cat(sb.reshape(b, self.sideband, c), dim=1)

      x = self.attn(x)

      if sb is not None:
        x, sb = x.split([h * w, self.sideband], dim=1)
        sb = sb.reshape(b, -1)
      x = x.transpose(1, 2).reshape(b, c, h, w)

      if sb is not None: return x, sb
      else: return x
    else:
      x = self.conv7x7(x)
      return x

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=2):
    if cout == 0: cout = cin

    self.up = nn.Conv2d(cin, cout * exp, 1, 1, 0, bias=False)
    self.down = nn.Conv2d(cout * exp, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = nonlinear(self.up(x))
    return self.down(x)

class Block:
  def __init__(self, dim:int, attn:bool=False, sideband:int=0, last:bool=False):
    assert sideband == 0 or attn, "attn required for sideband"

    if attn:
      self.cpe_norm = BatchNorm(dim)
      self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    self.tnorm = BatchNorm(dim)
    if sideband > 0: self.sideband_tnorm = nn.RMSNorm(dim * sideband)
    self.token_mixer = TokenMixer(dim, attn=attn, sideband=sideband)

    self.last = last
    if not last:
      self.cnorm = BatchNorm(dim)
      self.channel_mixer = ChannelMixer(dim)

    if sideband > 0:
      self.sideband_cnorm = nn.RMSNorm(dim * sideband)
      self.sideband_channel_mixer = FFN(dim * sideband, dim * sideband, dim * sideband * 2, blocks=0)

  def __call__(self, x:Tensor, sb:Tensor|None=None) -> Tensor | tuple[Tensor, Tensor]:
    if hasattr(self, "cpe"):
      xx = self.cpe(self.cpe_norm(x))
      x = x + xx

    if sb is not None:
      xx, sbsb = self.token_mixer(self.tnorm(x), self.sideband_tnorm(sb))
      sb = sb + sbsb
    else:
      xx = self.token_mixer(self.tnorm(x))
    x = x + xx

    if not self.last:
      xx = self.channel_mixer(self.cnorm(x))
      x = x + xx

    if sb is not None:
      sbsb = self.sideband_channel_mixer(self.sideband_cnorm(sb))
      sb = sb + sbsb

    if sb is not None: return x, sb
    else: return x

class Downsample:
  def __init__(self, cin:int, cout:int):
    self.conv = ConvNorm(cin, cout, 3, 2, 1, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.conv(x)

    # shortcut
    x = pixel_unshuffle(x, 2)
    b, c, h, w = x.shape
    x = x.reshape(b, xx.shape[1], c // xx.shape[1], h, w)
    x = x.mean(2)

    return x + xx

class Stage:
  def __init__(self, cin:int, cout:int, num_blocks:int, attn:bool=False, sideband:int=0, last:bool=False):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [Block(cout, attn=attn, sideband=sideband, last=last and i==num_blocks-1) for i in range(num_blocks)]

  def __call__(self, x:Tensor, sb:Tensor|None=None) -> Tensor | tuple[Tensor, Tensor]:
    if hasattr(self, "downsample"):
      x = self.downsample(x)

    for block in self.blocks:
      if sb is not None:
        x, sb = block(x, sb)
      else:
        x = cast(Tensor, block(x))

    if sb is not None: return x, sb
    return x

class Stem:
  def __init__(self, cin:int, cout:int):
    self.conv1 = ConvNorm(cin, cout // 2, 5, 2, 2, bias=False)
    self.conv2 = ConvNorm(cout // 2, cout * 2, 5, 2, 2, bias=False)
    self.proj = ConvNorm(cout * 2, cout, 1, 1, 0, bias=False)
    self.se = SE(cout, cout // 8)

  def __call__(self, x: Tensor) -> Tensor:
    x = nonlinear(self.conv1(x))
    x = nonlinear(self.conv2(x))
    x = self.proj(x)
    return self.se(x)

class Backbone:
  def __init__(self, cin:int, cstage:list[int], stages:list[int], attn:bool, sideband:int=0):
    self.stem = Stem(cin, cstage[0])

    self.stage0 = Stage(cstage[0], cstage[0], stages[0])
    self.stage1 = Stage(cstage[0], cstage[1], stages[1])
    self.stage2 = Stage(cstage[1], cstage[2], stages[2], attn=attn, sideband=sideband)
    if sideband > 0:
      self.stage2_sideband_norm = nn.RMSNorm(cstage[2] * sideband)
      self.stage2_sideband_proj = nn.Linear(cstage[2] * sideband, cstage[3] * sideband, bias=False)
    self.stage3 = Stage(cstage[2], cstage[3], stages[3], attn=attn, sideband=sideband, last=True)

  def __call__(self, x:Tensor, sb:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = self.stem(x)

    x0 = cast(Tensor, self.stage0(x))
    x1 = cast(Tensor, self.stage1(x0))
    if sb is not None:
      x2, sb = self.stage2(x1, sb)
      sb = self.stage2_sideband_proj(self.stage2_sideband_norm(sb))
      x3, sb = self.stage3(x2, sb)
    else:
      x2 = cast(Tensor, self.stage2(x1))
      x3 = cast(Tensor, self.stage3(x2))

    if sb is not None: return x0, x1, x2, x3, sb
    return x0, x1, x2, x3

class FFNBlock:
  def __init__(self, dim:int, exp:int, norm:bool=True):
    if norm: self.norm = BatchNorm(dim)
    self.up = nn.Linear(dim, dim * exp)
    self.mix = nn.Linear(dim * exp, dim * exp)
    self.down = nn.Linear((dim * exp)//2, dim)

  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "norm"): xx = self.norm(x)
    else: xx = x
    xx = nonlinear(self.up(xx))
    xx, gate = self.mix(xx).chunk(2, dim=-1)
    xx = xx * nonlinear(gate)
    xx = nonlinear(self.down(xx))
    return x + xx

class FFN:
  def __init__(self, in_dim:int, out_dim:int, mid_dim:int, exp:int=1, blocks:int=1, norm:bool=True):
    self.up = nn.Linear(in_dim, mid_dim)
    self.blocks = [FFNBlock(mid_dim, exp, norm) for _ in range(blocks)]
    self.down = nn.Linear(mid_dim, out_dim)

  def __call__(self, x:Tensor) -> Tensor:
    x = nonlinear(self.up(x))
    x = x.sequential(self.blocks)
    return self.down(x)

class Decoder:
  def __init__(self, cin:int, cout:int, blocks:int):
    self.ffn = FFN(cin, cout, cout, exp=2, blocks=blocks, norm=True)

  def __call__(self, sb:Tensor) -> Tensor:
    return self.ffn(sb)

class Head:
  def __init__(self, in_dim:int, mid_dim:int, out_dim:int):
    self.ffn = FFN(in_dim, mid_dim, out_dim, blocks=1, exp=2, norm=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.ffn(x)

class Heads:
  def __init__(self, in_dim:int):
    self.cls_head = Head(in_dim, 2, 64)
    self.x_head = Head(in_dim, 512, 128)
    self.y_head = Head(in_dim, 256, 128)
    # self.dist_head = Head(in_dim, 64, 64)
    self.color_head = Head(in_dim, 4, 64)
    self.number_head = Head(in_dim, 6, 64)

  def __call__(self, f:Tensor):
    cl = self.cls_head(f)
    x = self.x_head(f)
    y = self.y_head(f)
    # dist = self.dist_head(f)
    color = self.color_head(f)
    number = self.number_head(f)

    if not Tensor.training:
      cl = cl.softmax(1)
      clm, clp = cl.argmax(1, keepdim=True), cl.max(1, keepdim=True).float()

      if not hasattr(self, "x_arange"): self.x_arange = Tensor.arange(512).unsqueeze(1)
      if not hasattr(self, "y_arange"): self.y_arange = Tensor.arange(256).unsqueeze(1)
      x = (x.softmax() @ self.x_arange).float()
      y = (y.softmax() @ self.y_arange).float()

      # dist = (dist.softmax() @ Tensor.arange(64).unsqueeze(1)).float() / 4

      color = color.softmax(1)
      colorm, colorp = color.argmax(1, keepdim=True), color.max(1, keepdim=True).float()

      number = number.softmax(1)
      numberm, numberp = number.argmax(1, keepdim=True) + 1, number.max(1, keepdim=True).float()

      return Tensor.cat(clm, clp, x, y, colorm, colorp, numberm, numberp, dim=1)

    return cl, x, y, color, number

class Model:
  def __init__(self, dim:int=128, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 6, 2], sideband:int=2):
    self.sideband = Tensor.zeros(1, cstage[-2] * sideband)
    self.backbone = Backbone(cin=6, cstage=cstage, stages=stages, attn=True, sideband=sideband)
    self.decoder = Decoder(cstage[-1] * sideband, dim, blocks=1)
    self.heads = Heads(dim)

  def __call__(self, img:Tensor):
    # image normalization
    img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    xs = self.backbone(img, self.sideband.expand(img.shape[0], -1))
    f = self.decoder(xs[-1])
    return self.heads(f)


if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad.helpers import GlobalCounters, getenv
  from tinygrad.engine.jit import TinyJit
  from functools import partial

  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  model = Model()

  @partial(TinyJit, prune=True)
  def run(x:Tensor):
    return model(x)
  x = Tensor.randn(1, 128, 256, 6).realize()
  GlobalCounters.reset()
  run(x)
  x = Tensor.randn(1, 128, 256, 6).realize()
  GlobalCounters.reset()
  run(x)

  # full run
  x = Tensor.randn(1, 128, 256, 6).realize()
  GlobalCounters.reset()
  run(x)

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
  print(f"backbone parameters: {sum(p.numel() for p in get_parameters(model.backbone))}")
  print(f"decoder parameters: {sum(p.numel() for p in get_parameters(model.decoder))}")
  print(f"head parameters: {sum(p.numel() for p in get_parameters(model.heads))}")
