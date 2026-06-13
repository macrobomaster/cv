import math

from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import Attention, FFNBlock, FFN
from ..common.nn.fuse import FusedBlock
from ..common.nn.norm import GRN, RMSNorm2d
from .common import IMG_H, IMG_W, X3_H, X3_W, N_X3_TOKENS, N_FEAT_TOKENS, T

class LayerScale:
  """Per-channel learnable residual scaling. Init small (~1e-4) so initial residual contribution
  is tiny — keeps residual stream bounded at init. From CaiT/DeiT III (Touvron et al. 2021).
  Stored 1D so it routes to AdamW (not Muon) under any ndim>=2 split. Gamma kept in fp32 and
  the residual add done in fp32 so small contributions survive when activations are bf16."""
  def __init__(self, dim:int, init:float=1e-4, dims_2d:bool=False):
    self.gamma = Tensor.ones(dim, dtype=dtypes.float32) * init
    self.dims_2d = dims_2d

  def __call__(self, x:Tensor, xx:Tensor, dropout:float=0.0) -> Tensor:
    g = self.gamma.reshape(1, -1, 1, 1) if self.dims_2d else self.gamma
    out = x.cast(dtypes.float32) + (xx.cast(dtypes.float32) * g).dropout(dropout)
    return out.cast(x.dtype)

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=3):
    if cout == 0: cout = cin
    self.up = nn.Conv2d(cin, cin * exp, 1, 1, 0, bias=False)
    self.grn = GRN(cin * exp)
    self.down = nn.Conv2d(cin * exp, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.down(self.grn(self.up(x).gelu()))

class ConvBlock:
  def __init__(self, dim:int, dropout:float=0.0):
    self.dropout = dropout

    self.tnorm = RMSNorm2d(dim)
    self.token_mixer = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
    self.ls1 = LayerScale(dim, dims_2d=True)

    self.cnorm = RMSNorm2d(dim)
    self.channel_mixer = ChannelMixer(dim)
    self.ls2 = LayerScale(dim, dims_2d=True)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.token_mixer(self.tnorm(x))
    x = x + self.ls1(xx).dropout(self.dropout)

    xx = self.channel_mixer(self.cnorm(x))
    x = x + self.ls2(xx).dropout(self.dropout)

    return x

class AttnBlock:
  def __init__(self, dim:int, sideband_dim:int, dropout:float=0.0):
    self.dropout = dropout
    self.sideband, self.sideband_dim = sideband_dim // dim, sideband_dim

    self.cpe = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

    self.tnorm = nn.RMSNorm(dim)
    self.token_mixer = Attention(dim, dim // 4, heads=2, kv_heads=1, out="mod", dropout=dropout)
    self.ls_attn_x = LayerScale(dim, dims_2d=True)
    self.ls_attn_sb = LayerScale(sideband_dim)

    self.cnorm = RMSNorm2d(dim)
    self.channel_mixer = ChannelMixer(dim)
    self.ls_chm = LayerScale(dim, dims_2d=True)

    self.sideband_channel_mixer = FFNBlock(sideband_dim, exp=2, norm=True, bias=False, dropout=dropout)
    self.ls_sbm = LayerScale(sideband_dim)

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    b, c, h, w = x.shape

    # conditional positional encoding
    xx = self.cpe(x)
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
    sb = sb + self.ls_attn_sb(sbsb.reshape(b, -1))
    x = x + self.ls_attn_x(xx)

    # run channel mixer
    xx = self.channel_mixer(self.cnorm(x))
    x = x + self.ls_chm(xx).dropout(self.dropout)

    # run sideband channel mixer
    sbsb = self.sideband_channel_mixer(sb)
    sb = sb + self.ls_sbm(sbsb).dropout(self.dropout)

    return x, sb

class Downsample(FusedBlock):
  def __init__(self, cin:int, cout:int, shortcut:bool=True):
    self.cout, self.shortcut = cout, shortcut
    # pre-norm at block input — one norm per block
    self.norm = RMSNorm2d(cin)
    self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=True)
    self.dw3x3 = nn.Conv2d(cout, cout, 3, 2, 1, groups=cout, bias=True)
    self.dw7x7 = nn.Conv2d(cout, cout, 7, 2, 3, groups=cout, bias=True)
    self.ls = LayerScale(cout, dims_2d=True)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.norm(x)
    xx = self.pw(xx).gelu()
    if not self.fused:
      xx = (self.dw3x3(xx) + self.dw7x7(xx)).gelu()
    else:
      xx = self.conv(xx).gelu()

    # shortcut uses raw x (not normed) — preserves residual stream scale
    if self.shortcut:
      x = pixel_unshuffle(x, 2)
      b, c, h, w = x.shape
      x = x.reshape(b, xx.shape[1], c // xx.shape[1], h, w)
      x = x.mean(2)
      x = x + self.ls(xx)
    else:
      x = xx

    return x

  def fuse(self) -> bool:
    if not (was_fused := super().fuse()):
      dw7x7_w = self.dw7x7.weight
      dw3x3_w = self.dw3x3.weight.pad((2, 2, 2, 2))
      w = dw3x3_w + dw7x7_w

      dw7x7_b = self.dw7x7.bias
      dw3x3_b = self.dw3x3.bias
      b = dw3x3_b + dw7x7_b

      self.conv = nn.Conv2d(self.cout, self.cout, 7, 2, 3, groups=self.cout, bias=True)
      self.conv.weight.replace(w)
      self.conv.bias.replace(b)

      del self.dw3x3
      del self.dw7x7
    return was_fused

class ConvStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [ConvBlock(cout, dropout=dropout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    return x.sequential(self.blocks), sb

class AttnStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, sideband_dim:int, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [AttnBlock(cout, sideband_dim, dropout=dropout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    for block in self.blocks:
      x, sb = block(x, sb)
    return x, sb

class Stem:
  def __init__(self, cin:int, cout:int, patch_size:int=2, temporal_size:int=1):
    assert patch_size in (1, 2, 4, 8), "patch_size must be one of 1, 2, 4, or 8"
    assert temporal_size in (1, 2, 4, 8), "temporal_size must be one of 1, 2, 4, or 8"
    self.patch_size = patch_size
    self.temporal_size = temporal_size
    self.conv = nn.Conv2d(temporal_size * cin * patch_size * patch_size, cout, 5, 2, 2, bias=True)
    self.norm = RMSNorm2d(cout)

  def __call__(self, frames:list[Tensor]) -> Tensor:
    assert len(frames) == self.temporal_size
    x = self._temporal_dwt(frames) if self.temporal_size > 1 else frames[0]
    for _ in range(int(math.log2(self.patch_size))):
      x = self._dwt(x)
    return self.norm(self.conv(x)).gelu()

  def _temporal_dwt(self, frames:list[Tensor]) -> Tensor:
    while len(frames) > 1:
      new_frames = []
      for i in range(0, len(frames), 2):
        a, b = frames[i], frames[i + 1]
        low = (a + b) / math.sqrt(2)
        high = (a - b) / math.sqrt(2)
        new_frames.append(Tensor.cat(low, high, dim=1))
      frames = new_frames
    return frames[0]

  def _dwt(self, x:Tensor) -> Tensor:
    if not hasattr(self, "wavelets"):
      self.wavelets = Tensor([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=dtypes.float32, device=x.device).is_param_(False)
      self.flips = Tensor([1, -1], dtype=dtypes.float32, device=x.device).is_param_(False)
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
  def __init__(self, cin:int, cstage:list[int], stages:list[tuple[int, int]], sideband_dim:int,
               patch_size:int=2, temporal_size:int=1, dropout:float=0.0):
    self.temporal_size = temporal_size
    self.stem = Stem(cin, cstage[0], patch_size=patch_size, temporal_size=temporal_size)

    self.sideband_token = Tensor.randn(sideband_dim) * 0.02

    self.stage0 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[0][0] > 0: self.stage0[0] = ConvStage(cstage[0], cstage[0], stages[0][0], dropout=dropout)
    if stages[0][1] > 0: self.stage0[1] = AttnStage(cstage[0], cstage[0], stages[0][1], sideband_dim=sideband_dim, dropout=dropout)
    self.stage1 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[1][0] > 0: self.stage1[0] = ConvStage(cstage[0], cstage[1], stages[1][0], dropout=dropout)
    if stages[1][1] > 0: self.stage1[1] = AttnStage(cstage[1], cstage[1], stages[1][1], sideband_dim=sideband_dim, dropout=dropout)
    self.stage2 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[2][0] > 0: self.stage2[0] = ConvStage(cstage[1], cstage[2], stages[2][0], dropout=dropout)
    if stages[2][1] > 0: self.stage2[1] = AttnStage(cstage[2], cstage[2], stages[2][1], sideband_dim=sideband_dim, dropout=dropout)
    self.stage3 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[3][0] > 0: self.stage3[0] = ConvStage(cstage[2], cstage[3], stages[3][0], dropout=dropout)
    if stages[3][1] > 0: self.stage3[1] = AttnStage(cstage[2], cstage[3], stages[3][1], sideband_dim=sideband_dim, dropout=dropout)

  def __call__(self, img:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if img.ndim == 4: img = img.unsqueeze(1)
    B, T, H, W, C = img.shape

    # per frame img normalization
    img = img.cast(dtypes.default_float).permute(0, 1, 4, 2, 3).div(255).reshape(B * T, C, H, W)
    mean, std = img.mean([1, 2, 3], keepdim=True), img.std([1, 2, 3], keepdim=True)
    img = img.sub(mean).div(std.add(1e-6)).reshape(B, T, C, H, W)
    frames = [c.squeeze(1) for c in img.chunk(T, dim=1)]

    x = self.stem(frames)

    sb = self.sideband_token.reshape(1, -1).expand(x.shape[0], -1)

    # stages
    x0, sb = self.stage0[1](*self.stage0[0](x, sb))
    x1, sb = self.stage1[1](*self.stage1[0](x0, sb))
    x2, sb = self.stage2[1](*self.stage2[0](x1, sb))
    x3, sb = self.stage3[1](*self.stage3[0](x2, sb))

    return x0, x1, x2, x3, sb

class FeatureTokenizer:
  def __init__(self, cstage:list[int], sideband_dim:int, dim:int):
    self.norm_x3 = nn.RMSNorm(cstage[3])
    self.norm_sb = nn.RMSNorm(sideband_dim)
    self.proj_x3 = nn.Linear(cstage[3], dim, bias=False)
    self.proj_sb = nn.Linear(sideband_dim, dim, bias=False)
    self.x3_pos_emb = Tensor.randn(N_X3_TOKENS, dim) * 0.02

  def __call__(self, x3:Tensor, sb:Tensor) -> Tensor:
    x3_tokens = self.proj_x3(self.norm_x3(x3.flatten(2).transpose(1, 2))) + self.x3_pos_emb
    sb_token = self.proj_sb(self.norm_sb(sb)).unsqueeze(1)
    return Tensor.cat(x3_tokens, sb_token, dim=1)

class DecoderBlock:
  def __init__(self, dim:int, heads:int=4, kv_heads:int=1, dropout:float=0.0):
    self.attn_norm = nn.RMSNorm(dim)
    self.attn = Attention(dim, dim, heads=heads, kv_heads=kv_heads, out="proj", dropout=dropout)
    self.ls1 = LayerScale(dim)
    self.ffn = FFNBlock(dim, exp=2, norm=True, bias=False, dropout=dropout)
    self.ls2 = LayerScale(dim)

  def __call__(self, x:Tensor) -> Tensor:
    x = x + self.ls1(self.attn(self.attn_norm(x)))
    x = x + self.ls2(self.ffn(x))
    return x

NUM_CLASSES = 17
N_CORNERS = 4
N_BINS = 64

class Decoder:
  def __init__(self, dim:int, n_layers:int=4, n_bins:int=N_BINS, dropout:float=0.0):
    self.n_bins = n_bins
    self.pos_emb = Tensor.randn(N_FEAT_TOKENS, dim) * 0.02
    self.class_token = Tensor.randn(dim) * 0.02
    self.corner_tokens = Tensor.randn(N_CORNERS, dim) * 0.02

    self.blocks = [DecoderBlock(dim, dropout=dropout) for _ in range(n_layers)]
    self.ln_out = nn.RMSNorm(dim)

    self.class_proj = nn.Linear(dim, NUM_CLASSES, bias=False)
    self.corner_mlp = FFN(dim, 2 * n_bins, dim, exp=2, blocks=2, norm=True, bias=False, dropout=dropout)

  def __call__(self, feat_tokens:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # feat_tokens: (B, N_FEAT_TOKENS, D)
    B, _, D = feat_tokens.shape

    feat_tokens = feat_tokens + self.pos_emb.reshape(1, N_FEAT_TOKENS, D).expand(B, -1, -1)

    class_tok = self.class_token.reshape(1, 1, D).expand(B, 1, D)
    corner_toks = self.corner_tokens.reshape(1, N_CORNERS, D).expand(B, N_CORNERS, D)
    x = feat_tokens.cat(class_tok, corner_toks, dim=1)
    x = x.sequential(self.blocks)
    x = self.ln_out(x)

    class_tok = x[:, N_FEAT_TOKENS, :]                                                # (B, D)
    corner_toks = x[:, N_FEAT_TOKENS + 1 : N_FEAT_TOKENS + 1 + N_CORNERS, :]           # (B, 4, D)

    class_logits = self.class_proj(class_tok)    # (B, NUM_CLASSES)
    corner_dists = self.corner_mlp(corner_toks)  # (B, 4, 2*n_bins)
    dist_x = corner_dists[:, :, :self.n_bins]    # (B, 4, n_bins)
    dist_y = corner_dists[:, :, self.n_bins:]    # (B, 4, n_bins)
    return class_logits, dist_x, dist_y

# Unified class encoding:
# 0: no plate
# 1: 1_blank, 2: 3_blank, 3: 4_blank, 4: 5_blank
# 5: 1_red, 6: 2_red, 7: 3_red, 8: 4_red, 9: 5_red, 10: 6_red
# 11: 1_blue, 12: 2_blue, 13: 3_blue, 14: 4_blue, 15: 5_blue, 16: 6_blue

# decode from class_id to plate
CLASS_DECODE_TABLE = [
  (0, 0, 0),  # 0: no plate
  (1, 0, 1),  # 1: 1_blank
  (1, 0, 3),  # 2: 3_blank
  (1, 0, 4),  # 3: 4_blank
  (1, 0, 5),  # 4: 5_blank
  (1, 1, 1),  # 5: 1_red
  (1, 1, 2),  # 6: 2_red
  (1, 1, 3),  # 7: 3_red
  (1, 1, 4),  # 8: 4_red
  (1, 1, 5),  # 9: 5_red
  (1, 1, 6),  # 10: 6_red
  (1, 2, 1),  # 11: 1_blue
  (1, 2, 2),  # 12: 2_blue
  (1, 2, 3),  # 13: 3_blue
  (1, 2, 4),  # 14: 4_blue
  (1, 2, 5),  # 15: 5_blue
  (1, 2, 6),  # 16: 6_blue
]

DFL_WEIGHT = 0.25
L1_WEIGHT = 1.0
CLASS_WEIGHT = 1.0

class Model:
  def __init__(self, dim:int=512, temporal_size:int=1, sideband_dim:int=512,
               cstage:list[int]=[32, 64, 128, 256], stages:list[tuple[int, int]]=[(2, 0), (2, 0), (6, 3), (0, 2)],
               dropout:float=0.0):
    self.temporal_size = temporal_size
    self.backbone = Backbone(cin=3, cstage=cstage, stages=stages, sideband_dim=sideband_dim,
                             temporal_size=temporal_size, dropout=dropout)
    self.feature_tokenizer = FeatureTokenizer(cstage, sideband_dim, dim)
    self.decoder = Decoder(dim, n_layers=4, dropout=dropout)

  def encode(self, img:Tensor) -> Tensor:
    """img: (B, T, H, W, 3) or (B, H, W, 3). Returns (B, N_FEAT_TOKENS, D) — temporal info
    is fused at the stem via Haar DWT, so output has no remaining temporal dim."""
    x0, x1, x2, x3, sb = self.backbone(img)
    return self.feature_tokenizer(x3, sb)

  def __call__(self, img:Tensor, y:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # img: (B, T, IMG_H, IMG_W, 3) — training only.
    # Returns (total_loss, class_loss, l1_loss, dfl_loss).
    assert Tensor.training, "Model.__call__ is training-only. Use encode() + corner_predict() for inference."
    feat_tokens = self.encode(img)  # (B, N_FEAT_TOKENS, D)
    return self.corner_loss(feat_tokens, y)

  def corner_loss(self, feat_tokens:Tensor, y:Tensor) -> Tensor:
    B = feat_tokens.shape[0]
    N = self.decoder.n_bins

    # y: [class_id, c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y, has_class, has_corners] — 11 values
    class_id = y[:, 0].cast(dtypes.int32)
    corners = y[:, 1:9].reshape(B, N_CORNERS, 2)  # normalized [0, 1]
    has_class = y[:, 9]
    has_corners = y[:, 10]

    class_logits, dist_x, dist_y = self.decoder(feat_tokens)

    ce = class_logits.cross_entropy(class_id, reduction="none")  # (B,)
    class_loss = (has_class * ce).sum() / has_class.sum().add(1e-6)

    # Stack x/y distributions to (B, 4, 2, N) — N is the class/bin dim
    dist_xy = Tensor.stack(dist_x, dist_y, dim=2)

    # L1 on expectation
    bin_centers = Tensor.arange(N, dtype=dtypes.default_float) / (N - 1)
    expected = (dist_xy.softmax(-1) * bin_centers).sum(-1)  # (B, 4, 2)
    l1 = (expected - corners).abs().mean(-1).mean(-1)  # (B,)
    l1_loss = (has_corners * l1).sum() / has_corners.sum().add(1e-6)

    # Distribution Focal Loss: target distribution splits mass between two flanking bins
    c_scaled = (corners * (N - 1)).clamp(0, N - 1)
    l_idx = c_scaled.floor().cast(dtypes.int32).clamp(0, N - 2)
    w_r = c_scaled - l_idx.cast(dtypes.default_float)
    target_dist = l_idx.one_hot(N).cast(dtypes.default_float) * (1.0 - w_r).unsqueeze(-1) \
                + (l_idx + 1).one_hot(N).cast(dtypes.default_float) * w_r.unsqueeze(-1)  # (B, 4, 2, N)
    # cross_entropy expects class dim at 1 — reshape (B,4,2,N) → (B*4*2, N)
    dfl = dist_xy.reshape(-1, N).cross_entropy(target_dist.reshape(-1, N), reduction="none").reshape(B, N_CORNERS, 2).mean(-1).mean(-1)
    dfl_loss = (has_corners * dfl).sum() / has_corners.sum().add(1e-6)

    total = CLASS_WEIGHT * class_loss + L1_WEIGHT * l1_loss + DFL_WEIGHT * dfl_loss
    return total, class_loss, l1_loss, dfl_loss

  def corner_predict(self, feat_tokens:Tensor) -> Tensor:
    """Returns (B, 10): [class_id, confidence, c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y]
    with corners normalized to [0, 1]."""
    B = feat_tokens.shape[0]
    N = self.decoder.n_bins

    class_logits, dist_x, dist_y = self.decoder(feat_tokens)

    class_probs = class_logits.softmax(-1)
    class_id = class_probs.argmax(-1, keepdim=True).cast(dtypes.default_float)
    class_conf = class_probs.max(-1, keepdim=True).cast(dtypes.default_float)

    bin_centers = Tensor.arange(N, dtype=dtypes.default_float) / (N - 1)
    px = dist_x.softmax(-1)
    py = dist_y.softmax(-1)
    cx = (px * bin_centers).sum(-1)  # (B, 4)
    cy = (py * bin_centers).sum(-1)  # (B, 4)
    corners = Tensor.stack(cx, cy, dim=-1).reshape(B, N_CORNERS * 2)  # (B, 8)
    return Tensor.cat(class_id, class_conf, corners, dim=-1)

  def fuse(self):
    def _find_fused_blocks(m, prefix=""):
      blocks = []
      for attr in dir(m):
        if attr.startswith("__"): continue
        if isinstance(getattr(m, attr), Tensor): continue
        if isinstance(getattr(m, attr), FusedBlock): blocks.append((getattr(m, attr), prefix + attr))
        if hasattr(getattr(m, attr), "__dict__"): blocks.extend(_find_fused_blocks(getattr(m, attr), prefix + attr + "."))
        if isinstance(getattr(m, attr), list):
          for i, b in enumerate(getattr(m, attr)):
            blocks.extend(_find_fused_blocks(b, f"{prefix}{attr}[{i}]."))
      return blocks

    for block in reversed(_find_fused_blocks(self)):
      print(f"Fusing block: {block[1]}")
      block[0].fuse()

if __name__ == "__main__":
  from tinygrad.nn.state import get_parameters
  from tinygrad.helpers import GlobalCounters, getenv, Context
  from tinygrad.engine.jit import TinyJit
  from functools import partial
  import time

  BS = 1
  if getenv("HALF"):
    dtypes.default_float = dtypes.float16
  if train := getenv("TRAIN"):
    Tensor.training = True
    BS = 64

  model = Model(temporal_size=T)
  if getenv("FUSE"): model.fuse()

  if train:
    @TinyJit
    def run(x:Tensor, y:Tensor):
      return model(x, y)

    with Context(DEBUG=getenv("DEBUG", 2)):
      x = Tensor.randn(BS, T, IMG_H, IMG_W, 3).realize()
      y = Tensor.randn(BS, 11).realize()
      GlobalCounters.reset()
      ret = run(x, y)
      x = Tensor.randn(BS, T, IMG_H, IMG_W, 3).realize()
      y = Tensor.randn(BS, 11).realize()
      GlobalCounters.reset()
      ret = run(x, y)
      print(ret)
  else:
    @partial(TinyJit, prune=True)
    def run_backbone(x:Tensor):
      return model.encode(x)

    @partial(TinyJit, prune=True)
    def run_decoder(feat_tokens:Tensor):
      return model.corner_predict(feat_tokens)

    with Context(DEBUG=getenv("DEBUG", 2)):
      x = Tensor.randn(BS, T, IMG_H, IMG_W, 3).realize()
      GlobalCounters.reset()
      feat = run_backbone(x)
      ret = run_decoder(feat)
      x = Tensor.randn(BS, T, IMG_H, IMG_W, 3).realize()
      GlobalCounters.reset()
      feat = run_backbone(x)
      ret = run_decoder(feat)
      print(ret)

  # full runs
  tms = []
  for _ in range(15):
    x = Tensor.randn(BS, T, IMG_H, IMG_W, 3).realize()
    y = Tensor.randn(BS, 11).realize()
    GlobalCounters.reset()
    st = time.perf_counter()
    if train:
      nret = run(x, y)
    else:
      feat = run_backbone(x)
      nret = run_decoder(feat)
    tms.append(time.perf_counter() - st)

  print("jit run successful")
  print()

  tms = tms[5:]

  print(f"average time: {sum(tms) / len(tms):.4f} seconds")
  print(f"average latency: {(sum(tms) / len(tms)) * 1000:.2f} ms")
  print(f"average fps: {BS / (sum(tms) / len(tms)):.2f} fps")

  print(f"fastest time: {min(tms):.4f} seconds")
  print(f"fastest latency: {min(tms) * 1000:.2f} ms")
  print(f"fastest fps: {BS / min(tms):.2f} fps")

  print(f"slowest time: {max(tms):.4f} seconds")
  print(f"slowest latency: {max(tms) * 1000:.2f} ms")
  print(f"slowest fps: {BS / max(tms):.2f} fps")

  print(f"model size: {sum(p.numel() * p.dtype.itemsize for p in get_parameters(model)) / 1024**2:.2f} MB")

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
  print(f"backbone parameters: {sum(p.numel() for p in get_parameters(model.backbone))}")
  print(f"feature_tokenizer parameters: {sum(p.numel() for p in get_parameters(model.feature_tokenizer))}")
  print(f"decoder parameters: {sum(p.numel() for p in get_parameters(model.decoder))}")

  print(f"model gflops: {GlobalCounters.global_ops * 1e-9:.2f} GFLOPs")
