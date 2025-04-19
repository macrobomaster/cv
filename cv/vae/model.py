import math

from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

from ..common.tensor import pixel_shuffle
from ..common.nn import BatchNorm, UpsampleConvNorm, Attention, FFN

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=2):
    if cout == 0: cout = cin

    self.up = nn.Conv2d(cin, cout * exp, 1, 1, 0, bias=False)
    self.down = nn.Conv2d(cout * exp, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.up(x).gelu()
    return self.down(x)

class TokenMixer:
  def __init__(self, dim:int):
    self.conv7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
    self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.conv7x7(x) + self.conv3x3(x)

class ConvBlock:
  def __init__(self, dim:int, dropout:float=0.0):
    self.dropout = dropout

    self.tnorm = BatchNorm(dim)
    self.token_mixer = TokenMixer(dim)

    self.cnorm = BatchNorm(dim)
    self.channel_mixer = ChannelMixer(dim)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.token_mixer(self.tnorm(x))
    x = x + xx

    xx = self.channel_mixer(self.cnorm(x))
    x = x + xx.dropout(self.dropout)

    return x

class AttnBlock:
  def __init__(self, dim:int, sideband:int, sideband_mode:str="both", dropout:float=0.0):
    self.dropout = dropout
    self.sideband, self.sideband_mode = sideband, sideband_mode

    self.cpe_norm = BatchNorm(dim)
    self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    self.tnorm = nn.RMSNorm(dim)
    self.token_mixer = Attention(dim, dim // 4, heads=1, out="mod", dropout=dropout)

    if sideband_mode != "only":
      self.cnorm = BatchNorm(dim)
      self.channel_mixer = ChannelMixer(dim)

    if sideband_mode != "none":
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

    # residuals
    sb = sb + sbsb.reshape(b, -1)
    x = x + xx

    # channel mixer
    if self.sideband_mode != "only":
      xx = self.channel_mixer(self.cnorm(x))
      x = x + xx.dropout(self.dropout)

    # sideband channel mixer
    if self.sideband_mode != "none":
      sbsb = self.sideband_channel_mixer(self.sideband_cnorm(sb))
      sb = sb + sbsb.dropout(self.dropout)

    return x, sb

class Upsample:
  def __init__(self, cin:int, cout:int):
    # self.dw = UpsampleConvNorm(cin, cin, 3, 2, 1, groups=cin, bias=False)
    # self.pw = UpsampleConvNorm(cin, cout, 1, 1, 0, bias=False)
    self.conv = UpsampleConvNorm(cin, cout, 3, 2, 1, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    # return self.pw(self.dw(x))
    xx = self.conv(x)

    x = x.repeat_interleave(xx.shape[1] * 4 // x.shape[1], dim=1)
    x = pixel_shuffle(x, 2)

    return x + xx

class ConvStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, x_proj:bool=True, dropout:float=0.0):
    self.blocks = [ConvBlock(cin, dropout=dropout) for _ in range(num_blocks)]
    if x_proj: self.upsample = Upsample(cin, cout)

  def __call__(self, x:Tensor) -> Tensor:
    x = x.sequential(self.blocks)

    if hasattr(self, "upsample"):
      x = self.upsample(x)

    return x

class AttnStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, sideband:int, sideband_proj:bool=False, sideband_mode:str="both", x_proj:bool=True, dropout:float=0.0):
    self.blocks = [AttnBlock(cin, sideband, sideband_mode if i == num_blocks - 1 else "both", dropout=dropout) for i in range(num_blocks)]
    if x_proj: self.upsample = Upsample(cin, cout)
    if sideband_proj:
      self.sideband_norm = nn.RMSNorm(cin * sideband)
      self.sideband_proj = nn.Linear(cin * sideband, cout * sideband, bias=False)

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    for block in self.blocks:
      x, sb = block(x, sb)

    if hasattr(self, "upsample"):
      x = self.upsample(x)

    if hasattr(self, "sideband_proj"):
      sb = self.sideband_proj(self.sideband_norm(sb))

    return x, sb

class Unpatcher:
  def __init__(self, patch_size:int):
    self.patch_size = patch_size

  def __call__(self, x:Tensor) -> Tensor:
    for _ in range(int(math.log2(self.patch_size))):
      x = self._idwt(x)
    return x

  def _idwt(self, x:Tensor) -> Tensor:
    if not hasattr(self, "wavelets"):
      self.wavelets = Tensor([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=dtypes.float32, device=x.device)
      self.flips = Tensor([1, -1], dtype=dtypes.float32, device=x.device)
    h = self.wavelets
    n = h.shape[0]
    g = x.shape[1] // 4
    hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).cast(x.dtype)
    hh = (h * self.flips).reshape(1, 1, -1).repeat(g, 1, 1).cast(x.dtype)

    xll, xlh, xhl, xhh = x.chunk(4, dim=1)

    yl = xll.conv_transpose2d(hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
    yl = yl + xlh.conv_transpose2d(hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
    yh = xhl.conv_transpose2d(hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
    yh = yh + xhh.conv_transpose2d(hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
    y = yl.conv_transpose2d(hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))
    y = y + yh.conv_transpose2d(hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))

    return y * 2

class Model:
  def __init__(self, in_dim:int=128, cstage:list[int]=[128, 64, 32, 16], stages:list[int]=[2, 2, 2, 2], sideband:int=4, patch_size:int=2, dropout:float=0.0):
    self.input_proj = nn.Linear(in_dim * sideband, cstage[0] * sideband)
    self.image_tokens = Tensor.zeros(1, cstage[0], 2, 4)

    self.stage0 = AttnStage(cstage[0], cstage[1], stages[0], sideband=sideband, sideband_proj=True, dropout=dropout)
    self.stage1 = AttnStage(cstage[1], cstage[2], stages[1], sideband=sideband, sideband_mode="none", dropout=dropout)
    self.stage2 = ConvStage(cstage[2], cstage[3], stages[2], dropout=dropout)
    self.stage3 = ConvStage(cstage[3], cstage[3], stages[3], x_proj=False, dropout=dropout)

    self.out_norm = nn.RMSNorm(cstage[3])
    self.out = nn.Conv2d(cstage[3], 3 * patch_size * patch_size, 3, 1, 1)

    self.unpatcher = Unpatcher(patch_size)

  def __call__(self, sb:Tensor) -> Tensor:
    sb = self.input_proj(sb)
    x = self.image_tokens.expand(sb.shape[0], -1, -1, -1)

    x, sb = self.stage0(x, sb)
    x, sb = self.stage1(x, sb)
    x = self.stage2(x)
    x = self.stage3(x)

    x = self.out(self.out_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    x = self.unpatcher(x)
    return x

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad.helpers import GlobalCounters
  from tinygrad.engine.jit import TinyJit
  from functools import partial

  model = Model()

  @partial(TinyJit, prune=True)
  def run(x:Tensor):
    return model(x)
  x = Tensor.randn(1, 512).realize()
  GlobalCounters.reset()
  run(x)
  x = Tensor.randn(1, 512).realize()
  GlobalCounters.reset()
  run(x)

  # full run
  x = Tensor.randn(1, 512).realize()
  GlobalCounters.reset()
  run(x)

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
