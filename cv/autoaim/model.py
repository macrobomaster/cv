from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import BatchNorm, ConvNorm, Attention, FFN, FusedBlock

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=2):
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
  def __init__(self, dim:int, dropout:float=0.0):
    self.dropout = dropout

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
    x = x + xx.dropout(self.dropout)

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
  def __init__(self, dim:int, sideband:int, sideband_only:bool=False, dropout:float=0.0):
    self.dropout = dropout
    self.sideband, self.sideband_only = sideband, sideband_only

    self.cpe_norm = BatchNorm(dim)
    self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    self.tnorm = nn.RMSNorm(dim)
    self.token_mixer = Attention(dim, dim // 4, heads=1, out="mod", dropout=dropout)

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
      x = x + xx.dropout(self.dropout)

    # run sideband channel mixer
    sbsb = self.sideband_channel_mixer(self.sideband_cnorm(sb))
    sb = sb + sbsb.dropout(self.dropout)

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
    self.pw = ConvNorm(cin, cout, 1, 1, 0, bias=False)
    self.dw = ConvNorm(cout, cout, 3, 2, 1, groups=cout, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.dw(self.pw(x))

    # shortcut
    x = pixel_unshuffle(x, 2)
    b, c, h, w = x.shape
    x = x.reshape(b, xx.shape[1], c // xx.shape[1], h, w)
    x = x.mean(2)

    return x + xx

  def fuse(self):
    super().fuse()

    self.pw.fuse()
    self.dw.fuse()

class ConvStage(FusedBlock):
  def __init__(self, cin:int, cout:int, num_blocks:int, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [ConvBlock(cout, dropout=dropout) for _ in range(num_blocks)]

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
  def __init__(self, cin:int, cout:int, num_blocks:int, sideband:int, sideband_proj:bool=False, sideband_only:bool=False, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    if sideband_proj:
      self.sideband_norm = nn.RMSNorm(cin * sideband)
      self.sideband_proj = nn.Linear(cin * sideband, cout * sideband, bias=False)
    self.blocks = [AttnBlock(cout, sideband, sideband_only and i == num_blocks - 1, dropout=dropout) for i in range(num_blocks)]

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
  def __init__(self, cin:int, cstage:list[int], stages:list[int], sideband:int, sideband_only:bool=False, dropout:float=0.0):
    self.stem = Stem(cin, cstage[0])

    self.stage0 = ConvStage(cstage[0], cstage[0], stages[0], dropout=dropout)
    self.stage1 = ConvStage(cstage[0], cstage[1], stages[1], dropout=dropout)
    self.stage2 = AttnStage(cstage[1], cstage[2], stages[2], sideband=sideband, dropout=dropout)
    self.stage3 = AttnStage(cstage[2], cstage[3], stages[3], sideband=sideband, sideband_proj=True, sideband_only=sideband_only, dropout=dropout)

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
  def __init__(self, cstage:list[int], sideband:int, cout:int, blocks:int, dropout:float=0.0):
    self.cstage, self.sideband = cstage, sideband
    self.x3_proj = nn.Linear(cstage[-1], cstage[-1] * sideband, bias=True)
    self.ffn = FFN(cstage[-1] * sideband, cout, cout, exp=2, blocks=blocks, norm=True, dropout=dropout)

  def __call__(self, x0:Tensor, x1:Tensor, x2:Tensor, x3:Tensor, sb:Tensor) -> Tensor:
    x3 = self.x3_proj(x3.mean((2, 3)))
    x = x3 + sb

    return self.ffn(x)

class CLSHead:
  def __init__(self, in_dim:int, classes:int, mid_dim:int, dropout:float=0.0):
    self.ffn = FFN(in_dim, classes, mid_dim, blocks=1, exp=2, norm=True, dropout=dropout)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.ffn(x)

    if not Tensor.training:
      x = x.softmax(1)
      xm, xp = x.argmax(1, keepdim=True), x.max(1, keepdim=True).float()
      return Tensor.cat(xm, xp, dim=1)
    else:
      return x

class THRegHead:
  def __init__(self, in_dim:int, outputs:int, mid_dim:int, bins:int, low:float, high:float, dropout:float=0.0):
    self.outputs, self.bins, self.low, self.high = outputs, bins, low, high
    self.ffn = FFN(in_dim, outputs * bins + outputs, mid_dim, blocks=1, exp=2, norm=True, dropout=dropout)

  def __call__(self, x:Tensor) -> tuple[Tensor, Tensor]:
    x = self.ffn(x)

    logits, log_var = x.split([self.outputs * self.bins, self.outputs], dim=1)
    logits = logits.reshape(-1, self.outputs, self.bins)
    log_var = log_var.reshape(-1, self.outputs).tanh().mul(14)

    if not Tensor.training:
      mu = logits.softmax().mul(Tensor.linspace(self.low, self.high, self.bins).reshape(1, 1, -1)).sum(-1)
      var = log_var.exp()

      mu = mu.flatten(1)
      var = var.flatten(1)

      return mu, var
    else:
      return logits, log_var

class Heads:
  def __init__(self, in_dim:int, dropout:float=0.0):
    self.color_head = CLSHead(in_dim, 4, 32, dropout=dropout)
    self.number_head = CLSHead(in_dim, 6, 32, dropout=dropout)
    self.plate_head = THRegHead(in_dim, 10, 128, 64, -2, 2, dropout=dropout)

  def __call__(self, f:Tensor):
    color = self.color_head(f)
    number = self.number_head(f)
    plate_logits_mu, plate_log_var = self.plate_head(f)

    if not Tensor.training:
      return Tensor.cat(color, number, plate_logits_mu, plate_log_var, dim=1)
    else:
      return color, number, plate_logits_mu, plate_log_var

class Model(FusedBlock):
  def __init__(self, dim:int=256, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 6, 2], sideband:int=4, dropout:float=0.1):
    self.sideband = Tensor.zeros(1, cstage[-2] * sideband)
    self.backbone = Backbone(cin=6, cstage=cstage, stages=stages, sideband=sideband, dropout=dropout)
    self.decoder = Decoder(cstage, sideband, dim, blocks=2, dropout=dropout)
    self.heads = Heads(dim, dropout=dropout)

  def __call__(self, img:Tensor):
    img = img.cast(dtypes.default_float).permute(0, 3, 1, 2).div(255)
    img_mean, img_std = img.mean([2, 3], keepdim=True), img.std([2, 3], keepdim=True)
    img = img.sub(img_mean).div(img_std.add(1e-6))

    xs = self.backbone(img, self.sideband.expand(img.shape[0], -1))
    f = self.decoder(*xs)
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
