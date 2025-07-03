import math

from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import channel_shuffle, pixel_unshuffle
from ..common.nn import BatchNorm, ConvNorm, Attention, FFN, FFNBlock, RecConv

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=3):
    if cout == 0: cout = cin

    self.up = nn.Conv2d(cin, cout * exp, 1, 1, 0, bias=False)
    self.down = nn.Conv2d(cout * exp, cout, 1, 1, 0, bias=False)
    self.gate = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.up(x).gelu()
    return self.down(xx) * self.gate(x).sigmoid()

class TokenMixer:
  def __init__(self, dim:int, stage:int):
    self.conv = RecConv(dim, kernel_size=5, levels=3-stage)

  def __call__(self, x:Tensor) -> Tensor:
    return self.conv(x)

class ConvBlock:
  def __init__(self, dim:int, stage:int, dropout:float=0.0):
    self.dropout = dropout

    self.tnorm = BatchNorm(dim)
    self.token_mixer = TokenMixer(dim, stage)

    self.cnorm = BatchNorm(dim)
    self.channel_mixer = ChannelMixer(dim)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.token_mixer(self.tnorm(x))
    x = x + xx.dropout(self.dropout)

    xx = self.channel_mixer(self.cnorm(x))
    x = x + xx.dropout(self.dropout)

    return x

class AttnBlock:
  def __init__(self, dim:int, sideband_dim:int, sideband_only:bool=False, sideband_channel_mixer=None, dropout:float=0.0):
    self.dropout = dropout
    self.sideband, self.sideband_dim, self.sideband_only = sideband_dim // dim, sideband_dim, sideband_only

    self.cpe_norm = BatchNorm(dim)
    self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    self.tnorm = nn.RMSNorm(dim)
    self.token_mixer = Attention(dim, dim // 4, heads=1, out="mod", dropout=dropout)

    if not sideband_only:
      self.cnorm = BatchNorm(dim)
      self.channel_mixer = ChannelMixer(dim)

    if sideband_channel_mixer is not None:
      self.sideband_channel_mixer = lambda sb: sideband_channel_mixer(sb)
    else:
      self.sideband_channel_mixer = FFNBlock(sideband_dim, exp=2, norm=True, bias=False, dropout=dropout)

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

    # run channel mixer if not sideband only
    if not self.sideband_only:
      xx = self.channel_mixer(self.cnorm(x))
      x = x + xx.dropout(self.dropout)

    # run sideband channel mixer
    sbsb = self.sideband_channel_mixer(sb)
    sb = sb + sbsb.dropout(self.dropout)

    return x, sb

class Downsample:
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

class ConvStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, stage:int, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [ConvBlock(cout, stage, dropout=dropout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "downsample"):
      x = self.downsample(x)

    return x.sequential(self.blocks)

class AttnStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, sideband_dim:int, sideband_only:bool=False, sideband_channel_mixer=None, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [AttnBlock(cout, sideband_dim, sideband_only and i == num_blocks - 1, sideband_channel_mixer, dropout=dropout) for i in range(num_blocks)]

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    if hasattr(self, "downsample"):
      x = self.downsample(x)

    for block in self.blocks:
      x, sb = block(x, sb)

    return x, sb

class Stem:
  def __init__(self, cin:int, cout:int, exp:float=2):
    cmid = int(cout * exp)
    self.conv1 = ConvNorm(cin, cout, 5, 2, 2, bias=False)
    self.conv2 = ConvNorm(cout, cmid, 5, 2, 2, bias=False)
    self.proj = ConvNorm(cmid, cout, 1, 1, 0, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x).gelu()
    x = self.conv2(x).gelu()
    return self.proj(x)

class Patcher:
  def __init__(self, patch_size:int):
    self.patch_size = patch_size

  def __call__(self, x:Tensor) -> Tensor:
    for _ in range(int(math.log2(self.patch_size))):
      x = self._dwt(x)
    return x

  def _dwt(self, x:Tensor) -> Tensor:
    if not hasattr(self, "wavelets"):
      self.wavelets = Tensor([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=dtypes.float32, device=x.device)
      self.flips = Tensor([1, -1], dtype=dtypes.float32, device=x.device)
    h = self.wavelets
    n = h.shape[0]
    g = x.shape[1]
    hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).cast(x.dtype)
    hh = (h * self.flips).reshape(1, 1, -1).repeat(g, 1, 1).cast(x.dtype)

    x = x.pad((n - 2, n - 1, n - 2, n - 1), mode="reflect")
    xl = x.conv2d(hl.unsqueeze(2), groups=g, stride=(1, 2))
    xh = x.conv2d(hh.unsqueeze(2), groups=g, stride=(1, 2))
    xll = xl.conv2d(hl.unsqueeze(3), groups=g, stride=(2, 1))
    xlh = xl.conv2d(hh.unsqueeze(3), groups=g, stride=(2, 1))
    xhl = xh.conv2d(hl.unsqueeze(3), groups=g, stride=(2, 1))
    xhh = xh.conv2d(hh.unsqueeze(3), groups=g, stride=(2, 1))

    out = xll.cat(xlh, xhl, xhh, dim=1)
    return out / 2

class Backbone:
  def __init__(self, cin:int, cstage:list[int], stages:list[int|tuple[int, int]], sideband_dim:int, sideband_only:bool, shared_sideband_channel_mixer:bool=True, patch_size:int=2, dropout:float=0.0):
    self.patcher = Patcher(patch_size)
    self.stem = Stem(cin * patch_size * patch_size, cstage[0])

    if shared_sideband_channel_mixer:
      self.sideband_channel_mixer = FFNBlock(sideband_dim, exp=2, norm=True, bias=False, dropout=dropout)
    else:
      self.sideband_channel_mixer = None

    self.sideband_token = Tensor.zeros(1, sideband_dim)
    self.stage0 = ConvStage(cstage[0], cstage[0], stages[0], 0, dropout=dropout)
    self.stage1 = ConvStage(cstage[0], cstage[1], stages[1], 1, dropout=dropout)
    self.stage20 = ConvStage(cstage[1], cstage[2], stages[2][0], 2, dropout=dropout)
    self.stage21 = AttnStage(cstage[2], cstage[2], stages[2][1], sideband_dim=sideband_dim, sideband_channel_mixer=self.sideband_channel_mixer, dropout=dropout)
    self.stage3 = AttnStage(cstage[2], cstage[3], stages[3], sideband_dim=sideband_dim, sideband_only=sideband_only, sideband_channel_mixer=self.sideband_channel_mixer, dropout=dropout)

  def __call__(self, img:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # image normalization
    img = img.cast(dtypes.default_float).permute(0, 3, 1, 2).div(255)
    img_mean, img_std = img.mean([2, 3], keepdim=True), img.std([2, 3], keepdim=True)
    x = img.sub(img_mean).div(img_std.add(1e-6))

    x = self.patcher(x)
    x = self.stem(x)

    sb = self.sideband_token.expand(x.shape[0], -1)

    x0 = self.stage0(x)
    x1 = self.stage1(x0)
    _x2 = self.stage20(x1)
    x2, sb = self.stage21(_x2, sb)
    x3, sb = self.stage3(x2, sb)

    return x0, x1, x2, x3, sb

class FeatureAdapter:
  def __init__(self, cin:int, cout:int):
    cmid = cout // 4
    self.proj_in = ConvNorm(cin, cmid * 2, 1, 1, 0, bias=False)
    self.dw1 = ConvNorm(cmid, cmid, 3, 1, 1, groups=cmid, bias=False)
    self.dw2 = ConvNorm(cmid, cmid, 3, 1, 1, groups=cmid, bias=False)
    self.proj = nn.Linear(4 * cmid, cout)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.proj_in(x).relu()
    x0, x1 = channel_shuffle(x)
    x2 = self.dw1(x1).relu()
    x3 = self.dw2(x2).relu()
    x = Tensor.cat(x0, x1, x2, x3, dim=1)
    return self.proj(x.mean((2, 3)))

class Summarizer:
  def __init__(self, cstage:list[int], sideband_dim:int, dim:int, dropout:float=0.0):
    self.x0_adapter = FeatureAdapter(cstage[0], dim)
    self.x1_adapter = FeatureAdapter(cstage[1], dim)
    self.x2_adapter = FeatureAdapter(cstage[2], dim)
    self.x3_adapter = FeatureAdapter(cstage[3], dim)
    self.sb_adapter = nn.Linear(sideband_dim, dim)

    self.attention_norm = nn.RMSNorm(dim)
    self.attention = Attention(dim, dim, heads=4, dropout=dropout)
    self.ffn = FFN(dim, dim, dim, exp=2, blocks=2, norm=True, dropout=dropout)

  def __call__(self, features:tuple[Tensor, ...]) -> Tensor:
    x0 = self.x0_adapter(features[0]).relu()
    x1 = self.x1_adapter(features[1]).relu()
    x2 = self.x2_adapter(features[2]).relu()
    x3 = self.x3_adapter(features[3]).relu()
    sb = self.sb_adapter(features[4]).relu()

    f = Tensor.stack(x0, x1, x2, x3, sb, dim=1)
    sb = sb + self.attention(self.attention_norm(f))[:, -1, :]
    return self.ffn(sb)

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
      if not hasattr(self, "twohot_weights"):
        self.twohot_weights = Tensor.linspace(self.low, self.high, self.bins, device=x.device).reshape(1, 1, -1)
      mu = logits.softmax().mul(self.twohot_weights).sum(-1)
      var = log_var.exp()

      mu = mu.flatten(1)
      var = var.flatten(1)

      return mu, var
    else:
      return logits, log_var

class Heads:
  def __init__(self, in_dim:int, mid_dim:int=32, dropout:float=0.0):
    self.det_head = CLSHead(in_dim, 2, mid_dim*1, dropout=dropout)
    self.color_head = CLSHead(in_dim, 4, mid_dim*1, dropout=dropout)
    self.number_head = CLSHead(in_dim, 7, mid_dim*1, dropout=dropout)
    self.plate_head = THRegHead(in_dim, 10, mid_dim*4, 128, -2, 2, dropout=dropout)

  def __call__(self, f:Tensor):
    det = self.det_head(f)
    color = self.color_head(f)
    number = self.number_head(f)
    plate_logits_mu, plate_log_var = self.plate_head(f)

    if not Tensor.training:
      return Tensor.cat(det, color, number, plate_logits_mu, plate_log_var, dim=1)
    else:
      return det, color, number, plate_logits_mu, plate_log_var

class Model:
  def __init__(self, dim:int=512, cstage:list[int]=[32, 64, 128, 256], stages:list[int|tuple[int, int]]=[2, 2, (6, 3), 2], sideband_dim:int=512, dropout:float=0.0):
    self.backbone = Backbone(cin=3, cstage=cstage, stages=stages, sideband_dim=sideband_dim, sideband_only=False, shared_sideband_channel_mixer=True, dropout=dropout)
    self.summarizer = Summarizer(cstage, sideband_dim, dim, dropout=dropout)
    self.heads = Heads(dim, dropout=dropout)

  def __call__(self, img:Tensor):
    xs = self.backbone(img)
    f = self.summarizer(xs)
    return self.heads(f)

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad.helpers import GlobalCounters, getenv
  from tinygrad.engine.jit import TinyJit
  from functools import partial

  BS = 1
  if getenv("HALF", 0):
    dtypes.default_float = dtypes.float16
  if getenv("TRAIN", 0):
    Tensor.training = True
    BS = 256

  model = Model()

  @partial(TinyJit, prune=True)
  def run(x:Tensor):
    return model(x)
  x = Tensor.randn(BS, 256, 512, 3).realize()
  GlobalCounters.reset()
  run(x)
  x = Tensor.randn(BS, 256, 512, 3).realize()
  GlobalCounters.reset()
  run(x)

  # full run
  x = Tensor.randn(BS, 256, 512, 3).realize()
  GlobalCounters.reset()
  x = run(x)
  print(x)

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
  print(f"backbone parameters: {sum(p.numel() for p in get_parameters(model.backbone))}")
  print(f"summarizer parameters: {sum(p.numel() for p in get_parameters(model.summarizer))}")
  print(f"head parameters: {sum(p.numel() for p in get_parameters(model.heads))}")

  print(f"model gflops: {GlobalCounters.global_ops * 1e-9:.2f} GFLOPs")
