from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import BatchNorm, ConvNorm, Attention, FFN, FusedBlock

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=3):
    if cout == 0: cout = cin

    self.up = nn.Conv2d(cin, cout * exp, 1, 1, 0, bias=False)
    self.down = nn.Conv2d(cout * exp, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.up(x).gelu()
    return self.down(x)

class TokenMixer(FusedBlock):
  def __init__(self, dim:int):
    self.dim = dim
    self.conv7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
    self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    if not self.fused:
      return self.conv7x7(x) + self.conv3x3(x)
    else:
      return self.conv(x)

  def fuse(self):
    super().fuse()

    conv7x7_w = self.conv7x7.weight
    conv3x3_w = self.conv3x3.weight
    conv3x3_w = conv3x3_w.pad((2, 2, 2, 2))
    w = conv7x7_w + conv3x3_w

    self.conv = nn.Conv2d(self.dim, self.dim, 7, 1, 3, groups=self.conv7x7.groups, bias=False)
    self.conv.weight.assign(w)

    del self.conv7x7
    del self.conv3x3

class ConvBlock(FusedBlock):
  def __init__(self, dim:int):
    self.tnorm = BatchNorm(dim)
    self.token_mixer = TokenMixer(dim)

    self.cnorm = BatchNorm(dim)
    self.channel_mixer = ChannelMixer(dim)

  def __call__(self, x:Tensor) -> Tensor:
    if not self.fused:
      xx = self.token_mixer(self.tnorm(x))
      x = x + xx
    else:
      xx = self.token_mixer(self.tnorm(x))
      x = x + xx

    if not self.fused:
      xx = self.channel_mixer(self.cnorm(x))
    else:
      xx = self.channel_mixer(x)
    x = x + xx

    return x

  def fuse(self):
    super().fuse()

    self.token_mixer.fuse()
    # self.token_mixer = FusedBlock.fuse_bn_conv2d_dw(self.tnorm, self.token_mixer.conv)
    # self.token_mixer = FusedBlock.fuse_conv2d_residual(self.token_mixer)
    # del self.tnorm

    self.channel_mixer.up = FusedBlock.fuse_bn_conv2d_pw(self.cnorm, self.channel_mixer.up)
    del self.cnorm

class AttnBlock(FusedBlock):
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
    if not self.fused:
      xx = self.cpe(self.cpe_norm(x))
      x = x + xx
    else:
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

    # residuals
    sb = sb + sbsb.reshape(b, -1)
    x = x + xx

    # run channel mixer if not sideband only
    if not self.sideband_only:
      if not self.fused:
        xx = self.channel_mixer(self.cnorm(x))
      else:
        xx = self.channel_mixer(x)
      x = x + xx

    # run sideband channel mixer
    sbsb = self.sideband_channel_mixer(self.sideband_cnorm(sb))
    sb = sb + sbsb

    return x, sb

  def fuse(self):
    super().fuse()

    # self.cpe = FusedBlock.fuse_bn_conv2d_dw(self.cpe_norm, self.cpe)
    # self.cpe = FusedBlock.fuse_conv2d_residual(self.cpe)
    # del self.cpe_norm

    if not self.sideband_only:
      self.channel_mixer.up = FusedBlock.fuse_bn_conv2d_pw(self.cnorm, self.channel_mixer.up)
      del self.cnorm

class Downsample(FusedBlock):
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

  def fuse(self):
    super().fuse()

    self.conv.fuse()

class ConvStage(FusedBlock):
  def __init__(self, cin:int, cout:int, num_blocks:int):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [ConvBlock(cout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    return x.sequential(self.blocks)

  def fuse(self):
    super().fuse()

    if hasattr(self, "downsample"):
      self.downsample.fuse()

    for block in self.blocks:
      block.fuse()

class AttnStage(FusedBlock):
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

  def fuse(self):
    super().fuse()

    if hasattr(self, "downsample"):
      self.downsample.fuse()

    for block in self.blocks:
      block.fuse()

class Stem(FusedBlock):
  def __init__(self, cin:int, cout:int):
    cmid = max(cin * 2, cout // 2)
    self.conv1 = ConvNorm(cin, cmid, 5, 2, 2, bias=False)
    self.conv2 = ConvNorm(cmid, cout * 4, 5, 2, 2, bias=False)
    self.proj = ConvNorm(cout * 4, cout, 1, 1, 0, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x).gelu()
    x = self.conv2(x).gelu()
    return self.proj(x)

  def fuse(self):
    super().fuse()

    self.conv1.fuse()
    self.conv2.fuse()
    self.proj.fuse()

class Backbone(FusedBlock):
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

  def fuse(self):
    super().fuse()

    self.stem.fuse()
    self.stage0.fuse()
    self.stage1.fuse()
    self.stage2.fuse()
    self.stage3.fuse()

class Decoder:
  def __init__(self, cin:int, cout:int, blocks:int):
    self.ffn = FFN(cin, cout, cout, exp=3, blocks=blocks, norm=True)

  def __call__(self, sb:Tensor) -> Tensor:
    return self.ffn(sb)

class Head:
  def __init__(self, in_dim:int, out_dim:int, mid_dim:int, outputs:int=1):
    self.ffns = [FFN(in_dim, out_dim, mid_dim, blocks=1, exp=2, norm=False) for _ in range(outputs)]

  def __call__(self, x:Tensor) -> tuple[Tensor, ...]:
    return tuple(ffn(x) for ffn in self.ffns)

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
      xc = (xc.softmax() @ self.x_arange).float().mul(2).sub(256)
      yc = (yc.softmax() @ self.y_arange).float().mul(2).sub(128)
      xtl = (xtl.softmax() @ self.x_arange).float().mul(2).sub(256)
      ytl = (ytl.softmax() @ self.y_arange).float().mul(2).sub(128)
      xtr = (xtr.softmax() @ self.x_arange).float().mul(2).sub(256)
      ytr = (ytr.softmax() @ self.y_arange).float().mul(2).sub(128)
      xbl = (xbl.softmax() @ self.x_arange).float().mul(2).sub(256)
      ybl = (ybl.softmax() @ self.y_arange).float().mul(2).sub(128)
      xbr = (xbr.softmax() @ self.x_arange).float().mul(2).sub(256)
      ybr = (ybr.softmax() @ self.y_arange).float().mul(2).sub(128)

      number = number.softmax(1)
      numberm, numberp = number.argmax(1, keepdim=True) + 1, number.max(1, keepdim=True).float()

      return Tensor.cat(colorm, colorp, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, numberm, numberp, dim=1)

    return color, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, number

class Model(FusedBlock):
  def __init__(self, dim:int=256, cstage:list[int]=[24, 48, 96, 192], stages:list[int]=[2, 2, 6, 2], sideband:int=2):
    self.sideband = Tensor.zeros(1, cstage[-2] * sideband)
    self.backbone = Backbone(cin=6, cstage=cstage, stages=stages, sideband=sideband)
    self.decoder = Decoder(cstage[-1] * sideband, dim, blocks=1)
    self.heads = Heads(dim)

  def __call__(self, img:Tensor):
    img = img.cast(dtypes.default_float).permute(0, 3, 1, 2).div(255)
    xs = self.backbone(img, self.sideband.expand(img.shape[0], -1))
    f = self.decoder(xs[-1])
    return self.heads(f)

  def fuse(self):
    super().fuse()

    self.backbone.fuse()

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad.helpers import GlobalCounters, getenv
  from tinygrad.engine.jit import TinyJit
  from functools import partial

  if getenv("HALF", 0):
    dtypes.default_float = dtypes.float16

  model = Model()
  if getenv("FUSE", 0): model.fuse()

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
