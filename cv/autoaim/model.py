from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad.helpers import round_up, prod
from tinygrad.device import is_dtype_supported

class BatchNorm:
  def __init__(self, dim:int, eps=1e-5): self.n = nn.BatchNorm(dim, eps=eps)
  def __call__(self, x:Tensor) -> Tensor: return self.n(x.float()).cast(dtypes.default_float)

class AllNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Tensor | None = Tensor.ones(sz) if affine else None
    self.bias: Tensor | None = Tensor.zeros(sz) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype='long' if is_dtype_supported(dtypes.long) else 'int', requires_grad=False)
    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(1, requires_grad=False), Tensor.ones(1, requires_grad=False)

  def calc_stats(self, x:Tensor) -> tuple[Tensor, Tensor]:
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    if self.track_running_stats and not Tensor.training: return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
    batch_mean = x.mean()
    y = (x - batch_mean.detach().reshape(shape=shape_mask))
    batch_var = (y*y).mean()
    return batch_mean, batch_var

  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = self.calc_stats(x)
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(x.shape)/(prod(x.shape)-x.shape[1]) * batch_var.detach())
      self.num_batches_tracked += 1
    # reshape and expand batch_mean and batch_var
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    batch_mean = batch_mean.reshape(shape=shape_mask).expand(list(x.shape[i] if shape_mask[i] == -1 else 1 for i in range(x.ndim)))
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt())

class ConvNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, groups:int=1, dilation:int=1, bias:bool=False):
    self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=bias)
    self.n = BatchNorm(out_channels)
  def __call__(self, x:Tensor) -> Tensor: return self.n(self.c(x))

def channel_shuffle(x: Tensor, r:int=2) -> list[Tensor]:
  b, c, h, w = x.shape
  assert c % 4 == 0
  x = x.reshape(b * c // r, r, h * w).permute(1, 0, 2)
  x = x.reshape(r, -1, c // r, h, w)
  return list(x[i] for i in range(r))

def pixel_unshuffle(x:Tensor, factor:int) -> Tensor:
  b, c, h, w = x.shape
  oc, oh, ow = c*(factor*factor), h//factor, w//factor
  x = x.reshape(b, c, oh, factor, ow, factor)
  x = x.permute(0, 1, 3, 5, 2, 4)
  return x.reshape(b, oc, oh, ow)

def pixel_shuffle(x:Tensor, factor: int) -> Tensor:
  b, c, h, w = x.shape
  oc, oh, ow = c//(factor*factor), h*factor, w*factor
  x = x.reshape(b, oc, factor, factor, h, w)
  x = x.permute(0, 1, 4, 2, 5, 3)
  return x.reshape(b, oc, oh, ow)

def upsample(x:Tensor, scale:int) -> Tensor:
  bs, c, py, px = x.shape
  return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale, px, scale).reshape(bs, c, py * scale, px * scale)

def symlog(x:Tensor) -> Tensor:
  return x.sign() * x.abs().add(1).log()
def symexp(x:Tensor) -> Tensor:
  return x.sign() * x.abs().exp().sub(1)

def twohot(x:Tensor, bins:int) -> Tensor:
  k0 = x.floor().clamp(0, bins-1)
  k1 = x.ceil().clamp(0, bins-1)

  equal = k0 == k1
  to_below = equal.where(1, (k0 - x).abs())
  to_above = equal.where(0, (k1 - x).abs())

  total = to_below + to_above
  w_below = to_above / total
  w_above = to_below / total

  ar = Tensor.arange(bins).reshape(1, bins).expand(x.shape[0], bins)
  th = (ar == k0.reshape(x.shape[0], 1)).where(w_below.reshape(x.shape[0], 1), 0)
  th = th + (ar == k1.reshape(x.shape[0], 1)).where(w_above.reshape(x.shape[0], 1), 0)
  return th

class SE:
  def __init__(self, dim:int, cmid:int):
    self.cv1 = nn.Conv2d(dim, cmid, kernel_size=1, bias=False)
    self.cv2 = nn.Conv2d(cmid, dim, kernel_size=1, bias=False)
  def __call__(self, x: Tensor):
    xx = x.mean((2, 3), keepdim=True)
    xx = self.cv1(xx).relu()
    xx = self.cv2(xx).sigmoid()
    return x * xx

class TokenMixer:
  def __init__(self, dim:int, stride:int=1):
    self.stride = stride
    self.conv7x7 = ConvNorm(dim // 4, dim // 4, 7, stride, 3, groups=dim // 4, bias=True)
    self.conv3x3 = ConvNorm(dim // 4, dim // 4, 3, stride, 1, groups=dim // 4, bias=True)
    self.conv2x2 = ConvNorm(dim // 4, dim // 4, 2, stride, 1, dilation=2, groups=dim // 4, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    x0, x1, x2, x3 = channel_shuffle(x, 4)

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
    self.in_proj = ConvNorm(cin, cin * 2, 1, 1, 0, bias=False)
    self.mix = nn.Conv2d(cin * 2, cin * 2, 3, 1, 1, groups=cin * 2, bias=False)
    self.out_proj = ConvNorm(cin, cin if cout == 0 else cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.in_proj(x)

    x, gate = self.mix(x).chunk(2, dim=1)
    x = x * gate.gelu()

    x = self.out_proj(x)
    return x

class Block:
  def __init__(self, dim:int):
    self.token_mixer = TokenMixer(dim)
    self.norm = BatchNorm(dim)
    self.se = SE(dim, max(16, dim // 16))
    self.channel_mixer = ChannelMixer(dim)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.norm(self.token_mixer(x))
    xx = self.se(xx)
    return x + self.channel_mixer(xx)

class Downsample:
  def __init__(self, cin:int, cout:int):
    self.channel_mixer = ChannelMixer(cin, cout)
    self.cnorm = BatchNorm(cout)
    self.token_mixer = TokenMixer(cout, stride=2)
    self.tnorm = BatchNorm(cout)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.cnorm(self.channel_mixer(x))
    xx = self.tnorm(self.token_mixer(xx))

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

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x).gelu()
    return self.conv2(x)

class Stage:
  def __init__(self, cin:int, cout:int, num_blocks:int):
    self.downsample = Downsample(cin, cout) if cin != cout else lambda x: x
    self.blocks = [Block(cout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor) -> Tensor:
    x = self.downsample(x)
    return x.sequential(self.blocks)

class Backbone:
  def __init__(self, cin:int=3, c_mult:int=1):
    self.stem = Stem(cin, 16 * c_mult)

    self.stage0 = Stage(16 * c_mult, 16 * c_mult, 2)
    self.stage1 = Stage(16 * c_mult, 32 * c_mult, 2)
    self.stage2 = Stage(32 * c_mult, 64 * c_mult, 5)
    self.stage3 = Stage(64 * c_mult, 128 * c_mult, 1)

  def __call__(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    x = self.stem(x)

    x0 = self.stage0(x)
    x1 = self.stage1(x0)
    x2 = self.stage2(x1)
    x3 = self.stage3(x2)

    return x0, x1, x2, x3

class Neck:
  def __init__(self, cins:list[int], cout:int, cmid:int=256):
    self.x0 = ConvNorm(cins[0], cmid, 1, 1, 0, bias=False)
    self.x1 = ConvNorm(cins[1], cmid, 1, 1, 0, bias=False)
    self.x2 = ConvNorm(cins[2], cmid, 1, 1, 0, bias=False)
    self.x3 = ConvNorm(cins[3], cmid, 1, 1, 0, bias=False)

    self.feature = Tensor.kaiming_normal(1, 2, cmid)

    self.heads = 8
    self.head_size = cmid // self.heads
    self.q = nn.Linear(cmid, cmid)
    self.k = nn.Linear(cmid, cmid)
    self.v = nn.Linear(cmid, cmid)
    self.out = nn.Linear(cmid, cmid)

    self.proj = nn.Linear(cmid * 2, cout, bias=False)
    self.norm = BatchNorm(cout)

  def __call__(self, x0:Tensor, x1:Tensor, x2:Tensor, x3:Tensor) -> Tensor:
    x0 = self.x0(x0).mean((2, 3))
    x1 = self.x1(x1).mean((2, 3))
    x2 = self.x2(x2).mean((2, 3))
    x3 = self.x3(x3).mean((2, 3))
    x = Tensor.stack(x0, x1, x2, x3, dim=1)

    # cat with feature
    x = x.cat(self.feature.expand(x.shape[0], -1, -1), dim=1)

    q = self.q(x).reshape(x.shape[0], -1, self.heads, self.head_size).transpose(1, 2)
    k = self.k(x).reshape(x.shape[0], -1, self.heads, self.head_size).transpose(1, 2)
    v = self.v(x).reshape(x.shape[0], -1, self.heads, self.head_size).transpose(1, 2)
    attn = Tensor.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.shape[0], -1, self.heads * self.head_size)
    out = self.out(attn)

    return self.norm(self.proj(out[:, -self.feature.shape[1]:].flatten(1)))

class Head:
  def __init__(self, outputs: int, mid: int = 64):
    self.proj = nn.Linear(512, mid)

    self.up = nn.Linear(mid, mid * 2)
    self.down = nn.Linear(mid * 2, mid)

    self.out = nn.Linear(mid, outputs)

  def __call__(self, x: Tensor):
    x = self.proj(x).gelu()
    xx = self.up(x).gelu()
    x = (x + self.down(xx)).gelu()
    x = self.out(x)
    return x

class Model:
  def __init__(self):
    self.backbone = Backbone(cin=6)
    self.neck = Neck([16, 32, 64, 128], 512)

    # heads
    self.cls_head = Head(2)
    self.x_head = Head(512)
    self.y_head = Head(256)
    # self.dist_head = Head(20)

  def __call__(self, img:Tensor):
    # image normalization
    if Tensor.training: img = img.float()
    else: img = img.cast(dtypes.default_float)
    img = img.permute(0, 3, 1, 2) / 255

    x0, x1, x2, x3 = self.backbone(img)
    f = self.neck(x0, x1, x2, x3)

    cl = self.cls_head(f),
    x = self.x_head(f)
    y = self.y_head(f)
    # dist = self.dist_head(x)

    if not Tensor.training:
      cl = (cl[0].sigmoid().argmax(1), cl[0].sigmoid()[:, cl[0].argmax(1)])
      x = (x.softmax() @ Tensor.arange(512)).float()
      y = (y.softmax() @ Tensor.arange(256)).float()
      # dist = (dist.softmax() @ Tensor.arange(20)).float()

    return *cl, x, y

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad.helpers import GlobalCounters

  model = Model()
  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")

  Tensor.realize(*model(Tensor.zeros(1, 128, 256, 6)))
  GlobalCounters.reset()
  Tensor.realize(*model(Tensor.zeros(1, 128, 256, 6)))

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
