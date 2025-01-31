from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import channel_shuffle, pixel_unshuffle, norm, telu
from ..common.nn import BatchNorm, ConvNorm, ConvTransposeNorm, Attention, SE, UpsampleConvNorm

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
    x = x * telu(gate)

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

class Upsample:
  def __init__(self, cin:int, cout:int):
    self.cnorm = BatchNorm(cin)
    self.channel_mixer = ChannelMixer(cin, cout)
    self.tnorm = BatchNorm(cout)
    self.token_mixer = TokenMixer(cout, stride=1)
  def __call__(self, x:Tensor) -> Tensor:
    bs, c, py, px = x.shape
    x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
    x = self.channel_mixer(self.cnorm(x))
    xx = self.token_mixer(self.tnorm(x))
    return x + xx

class EncoderStem:
  def __init__(self, cin:int, cout:int):
    self.conv1 = ConvNorm(cin, cout // 2, 3, 2, 1, bias=False)
    self.conv2 = ConvNorm(cout // 2, cout, 3, 2, 1, bias=False)
    self.se = SE(cout, max(4, cout // 16))

  def __call__(self, x: Tensor) -> Tensor:
    x = telu(self.conv1(x))
    x = self.conv2(x)
    return self.se(x)

class DecoderStem:
  def __init__(self, cin:int, cout:int):
    self.conv1 = UpsampleConvNorm(cin, cout // 2, 3, 2, 1, bias=False)
    self.conv2 = UpsampleConvNorm(cout // 2, cout, 3, 2, 1, bias=False)
  def __call__(self, x:Tensor) -> Tensor:
    x = telu(self.conv1(x))
    return self.conv2(x).sigmoid()

class EncoderStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, attn:bool=False):
    self.downsample = Downsample(cin, cout) if cin != cout else lambda x: x
    self.blocks = [Block(cout, attn=attn) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    x = self.downsample(x)
    return x.sequential(self.blocks)

class DecoderStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, attn:bool=False):
    self.blocks = [Block(cin, attn) for _ in range(num_blocks)]
    self.upsample = Upsample(cin, cout) if cin != cout else lambda x: x
  def __call__(self, x:Tensor) -> Tensor:
    x = x.sequential(self.blocks)
    return self.upsample(x)

class Encoder:
  def __init__(self, cin:int=3, cout:int=16, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 7, 2]):
    self.stem = EncoderStem(cin, cstage[0])

    self.stage0 = EncoderStage(cstage[0], cstage[0], stages[0])
    self.stage1 = EncoderStage(cstage[0], cstage[1], stages[1])
    self.stage2 = EncoderStage(cstage[1], cstage[2], stages[2], attn=True)
    self.stage3 = EncoderStage(cstage[2], cstage[3], stages[3], attn=True)

    self.proj = nn.Conv2d(cstage[3], cout, 1, 1, 0)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.stem(x)

    x0 = self.stage0(x)
    x1 = self.stage1(x0)
    x2 = self.stage2(x1)
    x3 = self.stage3(x2)

    return self.proj(x3)

class Decoder:
  def __init__(self, cin:int=16, cout:int=3, cstage:list[int]=[128, 64, 32, 16], stages:list[int]=[2, 7, 2, 2]):
    self.proj = nn.Conv2d(cin, cstage[0], 1, 1, 0)

    self.stage0 = DecoderStage(cstage[0], cstage[0], stages[0], attn=True)
    self.stage1 = DecoderStage(cstage[0], cstage[1], stages[1], attn=True)
    self.stage2 = DecoderStage(cstage[1], cstage[2], stages[2])
    self.stage3 = DecoderStage(cstage[2], cstage[3], stages[3])

    self.stem = DecoderStem(cstage[3], cout)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.proj(x)

    x = self.stage0(x)
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)

    return self.stem(x)

class VQQuantizer:
  def __init__(self, embed_dim, n_embed):
    self.embed_dim, self.n_embed = embed_dim, n_embed
    self.embed = nn.Embedding(n_embed, embed_dim)
    self.embed.weight = Tensor.uniform(self.embed.weight.shape, low=-1 / n_embed, high=1 / n_embed, dtype=dtypes.float32)

  @staticmethod
  def get_very_efficient_rotation(u:Tensor, q:Tensor, e:Tensor) -> Tensor:
    w = ((u + q) / norm(u + q, axis=1, keepdim=True)).detach()
    e = e - 2 * e.matmul(w.unsqueeze(-1)).matmul(w.unsqueeze(1)) + 2 * e.matmul(u.unsqueeze(-1).detach()).matmul(q.unsqueeze(1).detach())
    return e

  def __call__(self, z: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    z = z.permute(0, 2, 3, 1)
    z_flat = z.reshape(-1, self.embed_dim)

    # compute cosine distance
    z_flat_norm = z_flat / norm(z_flat, 1, keepdim=True).add(1e-5)
    embed_weight_norm = self.embed.weight / norm(self.embed.weight, 1, keepdim=True).add(1e-5)
    d = z_flat_norm @ embed_weight_norm.T

    # find nearest embedding
    indices = d.argmax(axis=1).detach()
    z_q = self.embed(indices)

    # losses
    embed_loss = (z_q - z_flat.detach()).square().mean()
    commit_loss = (z_q.detach() - z_flat).square().mean()
    ortho_loss = embed_weight_norm @ embed_weight_norm.T - Tensor.eye(self.n_embed, device=embed_weight_norm.device)
    ortho_loss = ortho_loss.square().sum() / (self.n_embed ** 2)

    # ste
    z_q = z_flat + (z_q - z_flat).detach()

    # rotation trick
    pre_norm_q = VQQuantizer.get_very_efficient_rotation(z_flat_norm, z_q / norm(z_q, axis=1, keepdim=True).add(1e-5), z_flat.unsqueeze(1)).squeeze()
    z_q = pre_norm_q * (norm(z_q, axis=1, keepdim=True) / z_flat_norm).detach()

    # keep embeddings on hypersphere
    z_q = z_q / norm(z_q, 1, keepdim=True).add(1e-5)

    # reshape back to match z
    z_q = z_q.reshape(z.shape)
    z_q = z_q.permute(0, 3, 1, 2)

    return z_q, (embed_loss, commit_loss, ortho_loss)

  def quantize(self, z: Tensor) -> Tensor:
    B, _, H, W = z.shape
    z = z.permute(0, 2, 3, 1)
    z_flat = z.reshape(-1, self.embed_dim)
    z_flat_norm = z_flat / norm(z_flat, 1, keepdim=True).add(1e-5)
    embed_weight_norm = self.embed.weight / norm(self.embed.weight, 1, keepdim=True).add(1e-5)
    return (z_flat_norm @ embed_weight_norm.T).argmax(axis=1).reshape(B, H*W)
  def dequantize(self, indices: Tensor) -> Tensor: return self.embed(indices)

class Model:
  def __init__(self, embed_dim:int=16, n_embed:int=4096, cstage:list[int]=[16, 32, 64, 128], stages:list[int]=[2, 2, 7, 2]):
    self.encoder = Encoder(cin=3, cout=embed_dim, cstage=cstage, stages=stages)
    self.quantizer = VQQuantizer(embed_dim, n_embed)
    self.decoder = Decoder(cin=embed_dim, cout=3, cstage=list(reversed(cstage)), stages=list(reversed(stages)))

  def __call__(self, img:Tensor):
    # image normalization
    img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    z_e = self.encoder(img).cast(dtypes.float32)
    z = z_e / norm(z_e, 1, keepdim=True).add(1e-5)
    z_q, losses = self.quantizer(z)
    img_hat = self.decoder(z_q.cast(dtypes.default_float))
    return img_hat, img, losses

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

  run(Tensor.randn(1, 128, 128, 3))
  GlobalCounters.reset()
  run(Tensor.randn(1, 128, 128, 3))
  GlobalCounters.reset()
  print(run(Tensor.randn(1, 128, 128, 3)))

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
