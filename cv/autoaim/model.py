from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import BatchNorm, LayerNorm, ConvNorm, Attention, SE

def nonlinear(x:Tensor) -> Tensor: return x.gelu()

class TokenMixer:
  def __init__(self, dim:int, attn:bool=False):
    self.has_attn = attn

    if self.has_attn:
      self.attn = Attention(dim, dim // 4, heads=1, out="mod")
    else:
      self.conv7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    if self.has_attn:
      b, c, h, w = x.shape
      x = x.flatten(2).transpose(1, 2)
      x = self.attn(x).transpose(1, 2).reshape(b, c, h, w)
    else:
      x = self.conv7x7(x)

    return x

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=2, mix:bool=False):
    if cout == 0: cout = cin
    self.proj = nn.Conv2d(cin, cout * exp, 1, 1, 0, bias=False)
    if mix: self.mix = nn.Conv2d(cout * exp, cout * exp, 3, 1, 1, groups=cout * exp, bias=False)
    self.out = nn.Conv2d((cout * exp)//2 if mix else cout * exp, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.proj(x)
    if hasattr(self, "mix"):
      x, gate = self.mix(x).chunk(2, dim=1)
      x = x * nonlinear(gate)
    else:
      x = nonlinear(x)
    return self.out(x)

class Block:
  def __init__(self, dim:int, attn:bool=False):
    self.tnorm = BatchNorm(dim)
    self.token_mixer = TokenMixer(dim, attn=attn)
    self.cnorm = BatchNorm(dim)
    self.channel_mixer = ChannelMixer(dim, mix=attn)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.token_mixer(self.tnorm(x))
    x = x + xx
    xx = self.channel_mixer(self.cnorm(xx))
    return x + xx

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
  def __init__(self, cin:int, cout:int, num_blocks:int, attn:bool=False):
    self.downsample = Downsample(cin, cout) if cin != cout else lambda x: x
    self.blocks = [Block(cout, attn=attn) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    x = self.downsample(x)
    return x.sequential(self.blocks)

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
  def __init__(self, dim:int, exp:int=1):
    self.up = nn.Linear(dim, dim * exp, bias=False)
    self.down = nn.Linear(dim * exp, dim * 2, bias=False)
  def __call__(self, x:Tensor) -> Tensor:
    xx = nonlinear(self.up(x))
    xx, gate = self.down(xx).chunk(2, dim=-1)
    xx = xx * nonlinear(gate)
    return nonlinear(x + xx)

class FFN:
  def __init__(self, in_dim:int, out_dim:int, mid_dim:int, exp:int=1, blocks:int=1):
    self.proj = nn.Linear(in_dim, mid_dim, bias=False)
    self.blocks = [FFNBlock(mid_dim, exp) for _ in range(blocks)]
    self.out = nn.Linear(mid_dim, out_dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = nonlinear(self.proj(x))
    x = x.sequential(self.blocks)
    return self.out(x)

class DecoderBlock:
  def __init__(self, dim:int, has_self:bool=True):
    if has_self:
      self.self_attn_norm = LayerNorm(dim)
      self.self_attn = Attention(dim, dim, heads=4)
    self.cross_attn_norm = LayerNorm(dim)
    self.cross_attn = Attention(dim, dim, heads=4, out="mod")
    self.ffn_norm = LayerNorm(dim)
    self.ffn = FFN(dim, dim, dim*2, blocks=0)

  def __call__(self, x:Tensor, kv:Tensor) -> Tensor:
    if hasattr(self, "self_attn"):
      xx = self.self_attn(self.self_attn_norm(x))
      x = x + xx
    xx = self.cross_attn(self.cross_attn_norm(x), kv)
    x = x + xx
    xx = self.ffn(self.ffn_norm(x))
    return x + xx

class Decoder:
  def __init__(self, cin:int, cout:int, blocks:int, token_count:int):
    self.proj = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
    self.token_count = token_count
    self.pos_emb = nn.Embedding(token_count, cout)
    self.query_emb = nn.Embedding(1, cout)
    self.blocks = [DecoderBlock(cout, has_self=False) for _ in range(blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    x = nonlinear(self.proj(x))

    # flatten
    x = x.flatten(2).transpose(1, 2)
    assert x.shape[1] == self.token_count, f"expected {self.token_count} tokens, got {x.shape[1]}"

    # add positional embedding
    if not hasattr(self, "pos_arange"):
      self.pos_arange = Tensor.arange(self.token_count).unsqueeze(0)
    x = x + self.pos_emb(self.pos_arange).expand(x.shape[0], -1, -1)

    # query
    if not hasattr(self, "query"):
      self.query = Tensor([0]).unsqueeze(0)
    query = self.query_emb(self.query).expand(x.shape[0], -1, -1)

    # blocks
    for block in self.blocks:
      x = block(query, x)

    return x.sum(1)

class Head:
  def __init__(self, in_dim:int, mid_dim:int, out_dim:int):
    self.ffn = FFN(in_dim, mid_dim, out_dim, blocks=1)

  def __call__(self, x:Tensor) -> Tensor:
    return self.ffn(x)

class Heads:
  def __init__(self, in_dim:int):
    self.cls_head = Head(in_dim, 2, 64)
    self.x_head = Head(in_dim, 512, 64)
    self.y_head = Head(in_dim, 256, 64)
    # self.dist_head = Head(in_dim, 64, 64)

  def __call__(self, f:Tensor):
    cl = self.cls_head(f)
    x = self.x_head(f)
    y = self.y_head(f)
    # dist = self.dist_head(f)

    if not Tensor.training:
      cl = cl.softmax(1)
      clm, clp = cl.argmax(1, keepdim=True), cl.max(1, keepdim=True).float()

      if not hasattr(self, "x_arange"): self.x_arange = Tensor.arange(512).unsqueeze(1)
      if not hasattr(self, "y_arange"): self.y_arange = Tensor.arange(256).unsqueeze(1)
      x = (x.softmax() @ self.x_arange).float()
      y = (y.softmax() @ self.y_arange).float()
      # dist = (dist.softmax() @ Tensor.arange(64).unsqueeze(1)).float() / 4

      return Tensor.cat(clm, clp, x, y, dim=1)

    return cl, x, y

class Model:
  def __init__(self, dim:int=128, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 6, 2]):
    # feature extractor
    self.backbone = Backbone(cin=6, cstage=cstage, stages=stages, attn=True)

    # decoder
    self.decoder = Decoder(cstage[-1], dim, blocks=2, token_count=32)

    # heads
    self.heads = Heads(dim)

  def __call__(self, img:Tensor):
    # image normalization
    img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    # feature extraction
    xs = self.backbone(img)

    # decoder
    f = self.decoder(xs[-1])

    # heads
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
  run(Tensor.randn(1, 128, 256, 6))
  GlobalCounters.reset()
  run(Tensor.randn(1, 128, 256, 6))

  # full run
  GlobalCounters.reset()
  run(Tensor.randn(1, 128, 256, 6))

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
  print(f"backbone parameters: {sum(p.numel() for p in get_parameters(model.backbone))}")
  print(f"decoder parameters: {sum(p.numel() for p in get_parameters(model.decoder))}")
  print(f"head parameters: {sum(p.numel() for p in get_parameters(model.heads))}")
