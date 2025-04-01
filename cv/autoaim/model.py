from typing import cast
from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import BatchNorm, ConvNorm, Attention, SE, FFN

def nonlinear(x:Tensor) -> Tensor: return x.gelu()

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=2):
    if cout == 0: cout = cin

    self.up = nn.Conv2d(cin, cout * exp, 1, 1, 0, bias=False)
    self.down = nn.Conv2d(cout * exp, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = nonlinear(self.up(x))
    return self.down(x)

class ConvBlock:
  def __init__(self, dim:int):
    self.tnorm = BatchNorm(dim)
    self.token_mixer = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)

    self.cnorm = BatchNorm(dim)
    self.channel_mixer = ChannelMixer(dim)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.token_mixer(self.tnorm(x))
    x = x + xx

    xx = self.channel_mixer(self.cnorm(x))
    x = x + xx

    return x

class AttnBlock:
  def __init__(self, dim:int, sideband:int, sideband_only:bool=False):
    self.sideband, self.sideband_only = sideband, sideband_only

    self.cpe_norm = BatchNorm(dim)
    self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    self.tnorm = nn.RMSNorm(dim)
    self.token_mixer = Attention(dim, dim // 4, heads=1, out="mod")

    if not sideband_only:
      self.cnorm = BatchNorm(dim)
      self.channel_mixer = ChannelMixer(dim)

    self.sideband_cnorm = nn.RMSNorm(dim * sideband)
    self.sideband_channel_mixer = FFN(dim * sideband, dim * sideband, dim * sideband * 2, blocks=0, norm=False)

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    b, c, h, w = x.shape

    # conditional positional encoding
    xx = self.cpe(self.cpe_norm(x))
    x = x + xx

    # concat sideband to tokens
    xx = x.flatten(2).transpose(1, 2)
    xx = xx.cat(sb.reshape(b, self.sideband, c), dim=1)

    # run token mixer
    xx = self.token_mixer(self.tnorm(xx))

    # split tokens and sideband
    xx, sbsb = xx.split([h * w, self.sideband], dim=1)
    xx = xx.transpose(1, 2).reshape(b, c, h, w)

    # residual
    sb = sb + sbsb.reshape(b, -1)
    x = x + xx

    # don't run last channel mixer if sideband_only
    if not self.sideband_only:
      xx = self.channel_mixer(self.cnorm(x))
      x = x + xx

    # run sideband channel mixer
    sbsb = self.sideband_channel_mixer(self.sideband_cnorm(sb))
    sb = sb + sbsb

    return x, sb

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

class ConvStage:
  def __init__(self, cin:int, cout:int, num_blocks:int):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [ConvBlock(cout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    return x.sequential(self.blocks)

class AttnStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, sideband:int, sideband_proj:bool=False, output_sideband_only:bool=False):
    if cin != cout: self.downsample = Downsample(cin, cout)
    if sideband_proj:
      self.sideband_norm = nn.RMSNorm(cin * sideband)
      self.sideband_proj = nn.Linear(cin * sideband, cout * sideband, bias=False)
    self.blocks = [AttnBlock(cout, sideband, output_sideband_only and i == num_blocks - 1) for i in range(num_blocks)]

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    if hasattr(self, "downsample"):
      x = self.downsample(x)

    if hasattr(self, "sideband_proj"):
      sb = self.sideband_proj(self.sideband_norm(sb))

    for block in self.blocks:
      x, sb = block(x, sb)

    return x, sb

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
  def __init__(self, cin:int, cstage:list[int], stages:list[int], sideband:int):
    self.stem = Stem(cin, cstage[0])

    self.stage0 = ConvStage(cstage[0], cstage[0], stages[0])
    self.stage1 = ConvStage(cstage[0], cstage[1], stages[1])
    self.stage2 = AttnStage(cstage[1], cstage[2], stages[2], sideband=sideband)
    self.stage3 = AttnStage(cstage[2], cstage[3], stages[3], sideband=sideband, sideband_proj=True, output_sideband_only=True)

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = self.stem(x)

    x0 = self.stage0(x)
    x1 = self.stage1(x0)
    x2, sb = self.stage2(x1, sb)
    x3, sb = self.stage3(x2, sb)

    return x0, x1, x2, x3, sb

class Decoder:
  def __init__(self, cin:int, cout:int, blocks:int):
    self.ffn = FFN(cin, cout, cout, exp=2, blocks=blocks, norm=True)

  def __call__(self, sb:Tensor) -> Tensor:
    return self.ffn(sb)

class Head:
  def __init__(self, in_dim:int, out_dim:int, mid_dim:int, outputs:int=1):
    self.ffns = [FFN(in_dim, out_dim, mid_dim, blocks=1, exp=2, norm=False) for _ in range(outputs)]

  def __call__(self, x:Tensor) -> list[Tensor]:
    return [ffn(x) for ffn in self.ffns]

class Heads:
  def __init__(self, in_dim:int):
    self.x_head = Head(in_dim, 512, 128, outputs=5)
    self.y_head = Head(in_dim, 256, 128, outputs=5)
    self.color_head = Head(in_dim, 4, 64)
    self.number_head = Head(in_dim, 6, 64)

  def __call__(self, f:Tensor):
    xc, xtl, xtr, xbl, xbr = self.x_head(f)
    yc, ytl, ytr, ybl, ybr = self.y_head(f)
    (color,) = self.color_head(f)
    (number,) = self.number_head(f)

    if not Tensor.training:
      color = color.softmax(1)
      colorm, colorp = color.argmax(1, keepdim=True), color.max(1, keepdim=True).float()

      if not hasattr(self, "x_arange"): self.x_arange = Tensor.arange(512).unsqueeze(1)
      if not hasattr(self, "y_arange"): self.y_arange = Tensor.arange(256).unsqueeze(1)
      xc = (xc.softmax() @ self.x_arange).float()
      yc = (yc.softmax() @ self.y_arange).float()
      xtl = (xtl.softmax() @ self.x_arange).float()
      ytl = (ytl.softmax() @ self.y_arange).float()
      xtr = (xtr.softmax() @ self.x_arange).float()
      ytr = (ytr.softmax() @ self.y_arange).float()
      xbl = (xbl.softmax() @ self.x_arange).float()
      ybl = (ybl.softmax() @ self.y_arange).float()
      xbr = (xbr.softmax() @ self.x_arange).float()
      ybr = (ybr.softmax() @ self.y_arange).float()

      number = number.softmax(1)
      numberm, numberp = number.argmax(1, keepdim=True) + 1, number.max(1, keepdim=True).float()

      return Tensor.cat(colorm, colorp, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, numberm, numberp, dim=1)

    return color, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, number

class Model:
  def __init__(self, dim:int=128, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 6, 2], sideband:int=2):
    self.sideband = Tensor.zeros(1, cstage[-2] * sideband)
    self.backbone = Backbone(cin=6, cstage=cstage, stages=stages, sideband=sideband)
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
