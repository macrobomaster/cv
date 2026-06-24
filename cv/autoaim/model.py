import math

import numpy as np
from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import Attention, FFN, MLP, LayerScale, LayerScale2d
from ..common.nn.fuse import FusedBlock
from ..common.nn.norm import GRN, RMSNorm, RMSNorm2d
from ..common.losses import mal_loss
from .common import IMG_H, IMG_W, N_X2_TOKENS, N_X3_TOKENS, X2_H, X2_W, X3_H, X3_W, T, BIN_LO, BIN_HI, CANONICAL_CAMERA_MATRIX

class ChannelMixer:
  def __init__(self, cin:int, cout:int=0, exp:int=2):
    if cout == 0: cout = cin
    self.up = nn.Conv2d(cin, cin * exp, 1, 1, 0, bias=False)
    self.grn = GRN(cin * exp)
    self.down = nn.Conv2d(cin * exp, cout, 1, 1, 0, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.down(self.grn(self.up(x).gelu()))

class ConvTokenMixer(FusedBlock):
  def __init__(self, dim:int, stride:int=1):
    self.dim, self.stride = dim, stride
    self.dw3x3 = nn.Conv2d(dim, dim, 3, stride, 1, groups=dim)
    self.dw7x7 = nn.Conv2d(dim, dim, 7, stride, 3, groups=dim)

  def __call__(self, x:Tensor) -> Tensor:
    if not self.fused:
      return self.dw3x3(x) + self.dw7x7(x)
    else:
      return self.conv(x)

  def fuse(self) -> bool:
    if not (was_fused := super().fuse()):
      dw7x7_w = self.dw7x7.weight
      dw3x3_w = self.dw3x3.weight.pad((2, 2, 2, 2))
      w = dw3x3_w + dw7x7_w

      dw7x7_b = self.dw7x7.bias
      dw3x3_b = self.dw3x3.bias
      b = dw3x3_b + dw7x7_b

      self.conv = nn.Conv2d(self.dim, self.dim, 7, self.stride, 3, groups=self.dim)
      self.conv.weight.replace(w.realize())
      self.conv.bias.replace(b.realize())

      del self.dw3x3
      del self.dw7x7
    return was_fused

class ConvBlock:
  def __init__(self, dim:int, dropout:float=0.0):
    self.dropout = dropout

    self.tnorm = RMSNorm2d(dim)
    self.token_mixer = ConvTokenMixer(dim)
    self.ls1 = LayerScale2d(dim)

    self.cnorm = RMSNorm2d(dim)
    self.channel_mixer = ChannelMixer(dim)
    self.ls2 = LayerScale2d(dim)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.token_mixer(self.tnorm(x))
    x = self.ls1(x, xx, self.dropout)

    xx = self.channel_mixer(self.cnorm(x))
    x = self.ls2(x, xx, self.dropout)

    return x

class AttnBlock:
  def __init__(self, dim:int, sideband_dim:int, sr_ratio:int=1, dropout:float=0.0):
    self.dropout = dropout
    self.sideband, self.sideband_dim = sideband_dim // dim, sideband_dim
    self.sr_ratio = sr_ratio

    self.cpe = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

    self.tnorm = RMSNorm(dim)
    if sr_ratio > 1:
      self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio, groups=dim, bias=False)
      self.kvnorm = RMSNorm(dim)
    self.token_mixer = Attention(dim, dim // 4, heads=2, kv_heads=1, out="proj", dropout=dropout)
    self.ls_attn_x = LayerScale2d(dim)
    self.ls_attn_sb = LayerScale(sideband_dim)

    self.cnorm = RMSNorm2d(dim)
    self.channel_mixer = ChannelMixer(dim)
    self.ls_chm = LayerScale2d(dim)

    self.sideband_channel_mixer = FFN(sideband_dim, exp=2, norm=True, bias=False, dropout=dropout)
    self.ls_sbm = LayerScale(sideband_dim)

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    b, c, h, w = x.shape

    # conditional positional encoding
    xx = self.cpe(x)
    x = x + xx

    # query tokens: full-res spatial + sideband
    q = x.flatten(2).transpose(1, 2)
    q = self.tnorm(q.cat(sb.reshape(b, self.sideband, c), dim=1))

    # run token mixer (spatial-reduced K/V cross-attn when sr_ratio > 1, else self-attn)
    if self.sr_ratio > 1:
      kv = self.sr(x).flatten(2).transpose(1, 2)
      kv = self.kvnorm(kv.cat(sb.reshape(b, self.sideband, c), dim=1))
      xx = self.token_mixer(q, kv)
    else:
      xx = self.token_mixer(q)

    # split tokens and sideband
    xx, sbsb = xx.split([h * w, self.sideband], dim=1)
    xx = xx.transpose(1, 2).reshape(b, c, h, w)

    # residuals
    sb = self.ls_attn_sb(sb, sbsb.reshape(b, -1))
    x = self.ls_attn_x(x, xx)

    # run channel mixer
    xx = self.channel_mixer(self.cnorm(x))
    x = self.ls_chm(x, xx, self.dropout)

    # run sideband channel mixer
    sbsb = self.sideband_channel_mixer(sb)
    sb = self.ls_sbm(sb, sbsb, self.dropout)

    return x, sb

class Downsample:
  def __init__(self, cin:int, cout:int, shortcut:bool=True):
    self.cout, self.shortcut = cout, shortcut
    self.norm = RMSNorm2d(cin)
    self.pw = nn.Conv2d(cin, cout, 1, 1, 0)
    self.token_mixer = ConvTokenMixer(cout, stride=2)
    self.ls = LayerScale2d(cout)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.norm(x)
    xx = self.pw(xx).gelu()
    xx = self.token_mixer(xx).gelu()

    if self.shortcut:
      x = pixel_unshuffle(x, 2)
      b, c, h, w = x.shape
      x = x.reshape(b, xx.shape[1], c // xx.shape[1], h, w)
      x = x.mean(2)
      x = self.ls(x, xx)
    else:
      x = xx

    return x

class ConvStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [ConvBlock(cout, dropout=dropout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    return x.sequential(self.blocks), sb

class AttnStage:
  def __init__(self, cin:int, cout:int, num_blocks:int, sideband_dim:int, sr_ratio:int=1, dropout:float=0.0):
    if cin != cout: self.downsample = Downsample(cin, cout)
    self.blocks = [AttnBlock(cout, sideband_dim, sr_ratio=sr_ratio, dropout=dropout) for _ in range(num_blocks)]

  def __call__(self, x:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    if hasattr(self, "downsample"):
      x = self.downsample(x)
    for block in self.blocks:
      x, sb = block(x, sb)
    return x, sb

class Stem:
  def __init__(self, cin:int, cout:int, patch_size:int=2, temporal_size:int=1, kernel_size:int=3):
    assert patch_size in (1, 2, 4, 8), "patch_size must be one of 1, 2, 4, or 8"
    assert temporal_size in (1, 2, 4, 8), "temporal_size must be one of 1, 2, 4, or 8"
    self.patch_size = patch_size
    self.temporal_size = temporal_size
    self.conv = nn.Conv2d(temporal_size * cin * patch_size * patch_size, cout, kernel_size, 2, kernel_size//2)
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
               sr_ratios:list[int]=[1, 1, 1, 1], patch_size:int=2, temporal_size:int=1, dropout:float=0.0):
    self.temporal_size = temporal_size
    self.stem = Stem(cin, cstage[0], patch_size=patch_size, temporal_size=temporal_size)

    self.sideband_token = Tensor.randn(sideband_dim) * 0.02

    self.stage0 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[0][0] > 0: self.stage0[0] = ConvStage(cstage[0], cstage[0], stages[0][0], dropout=dropout)
    if stages[0][1] > 0: self.stage0[1] = AttnStage(cstage[0], cstage[0], stages[0][1], sideband_dim=sideband_dim, sr_ratio=sr_ratios[0], dropout=dropout)
    self.stage1 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[1][0] > 0: self.stage1[0] = ConvStage(cstage[0], cstage[1], stages[1][0], dropout=dropout)
    if stages[1][1] > 0: self.stage1[1] = AttnStage(cstage[1], cstage[1], stages[1][1], sideband_dim=sideband_dim, sr_ratio=sr_ratios[1], dropout=dropout)
    self.stage2 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[2][0] > 0: self.stage2[0] = ConvStage(cstage[1], cstage[2], stages[2][0], dropout=dropout)
    if stages[2][1] > 0: self.stage2[1] = AttnStage(cstage[2], cstage[2], stages[2][1], sideband_dim=sideband_dim, sr_ratio=sr_ratios[2], dropout=dropout)
    self.stage3 = [lambda x,sb:(x,sb), lambda x,sb:(x,sb)]
    if stages[3][0] > 0: self.stage3[0] = ConvStage(cstage[2], cstage[3], stages[3][0], dropout=dropout)
    if stages[3][1] > 0: self.stage3[1] = AttnStage(cstage[2], cstage[3], stages[3][1], sideband_dim=sideband_dim, sr_ratio=sr_ratios[3], dropout=dropout)

  def __call__(self, img:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if img.ndim == 4: img = img.unsqueeze(0)
    B, T, H, W, C = img.shape

    # per frame img normalization
    img = img.cast(dtypes.float32).permute(0, 1, 4, 2, 3).div(255).reshape(B * T, C, H, W)
    mean, std = img.mean([1, 2, 3], keepdim=True), img.std([1, 2, 3], keepdim=True)
    img = img.sub(mean).div(std.add(1e-6)).cast(dtypes.default_float).reshape(B, T, C, H, W)
    frames = [c.squeeze(1) for c in img.chunk(T, dim=1)]

    x = self.stem(frames)

    sb = self.sideband_token.reshape(1, -1).expand(x.shape[0], -1)

    # stages
    x0, sb = self.stage0[1](*self.stage0[0](x, sb))
    x1, sb = self.stage1[1](*self.stage1[0](x0, sb))
    x2, sb = self.stage2[1](*self.stage2[0](x1, sb))
    x3, sb = self.stage3[1](*self.stage3[0](x2, sb))

    return x0, x1, x2, x3, sb

def coord_grid(h:int, w:int) -> Tensor:
  u = (Tensor.arange(w, dtype=dtypes.float32) + 0.5) / w
  v = (Tensor.arange(h, dtype=dtypes.float32) + 0.5) / h
  grid = Tensor.stack(u.reshape(1, w).expand(h, w), v.reshape(h, 1).expand(h, w), dim=-1)  # (h,w,2)
  return grid.reshape(h * w, 2).is_param_(False)

class LearnedFourierPosEmb:
  def __init__(self, dim:int, pos_dim:int=2, f_dim:int=0, hidden:int=32):
    if f_dim == 0: f_dim = dim
    self.scale = 1.0 / math.sqrt(f_dim)
    self.freqs = nn.Linear(pos_dim, f_dim // 2, bias=False)
    self.mlp = MLP(f_dim, dim, hidden, blocks=0)

  def __call__(self, coords:Tensor) -> Tensor:
    r = self.freqs(coords)
    return self.mlp(Tensor.cat(r.cos(), r.sin(), dim=-1) * self.scale)

class FeatureTokenizer:
  def __init__(self, cstage:list[int], sideband_dim:int, dim:int):
    self.norm_x2 = RMSNorm(cstage[2])
    self.proj_x2 = nn.Linear(cstage[2], dim, bias=False)
    self.norm_x3 = RMSNorm(cstage[3])
    self.proj_x3 = nn.Linear(cstage[3], dim, bias=False)
    self.norm_sb = RMSNorm(sideband_dim)
    self.proj_sb = nn.Linear(sideband_dim, dim, bias=False)
    self.pos_emb = LearnedFourierPosEmb(dim)
    self.coords = Tensor.cat(coord_grid(X3_H, X3_W), coord_grid(X2_H, X2_W), dim=0).is_param_(False)  # [x3 ; x2]
    self.level = Tensor.randn(2, dim) * 0.02  # [x3, x2]

  def __call__(self, x2:Tensor, x3:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    pos = self.pos_emb(self.coords)
    x3_pos, x2_pos = pos[:N_X3_TOKENS], pos[N_X3_TOKENS:]
    x2_tokens = self.proj_x2(self.norm_x2(x2.flatten(2).transpose(1, 2))) + x2_pos + self.level[1]
    x3_tokens = self.proj_x3(self.norm_x3(x3.flatten(2).transpose(1, 2))) + x3_pos + self.level[0]
    sb_token = self.proj_sb(self.norm_sb(sb)).unsqueeze(1)  # global register -> decoder query, not memory
    mem = Tensor.cat(x3_tokens.contiguous(), x2_tokens.contiguous(), dim=1)  # [x3 ; x2] = 1440 (GPU-friendly)
    return mem, sb_token

class DecoderBlock:
  def __init__(self, dim:int, heads:int=4, kv_heads:int=1, dropout:float=0.0):
    self.sa_norm = RMSNorm(dim)
    self.self_attn = Attention(dim, dim, heads=heads, kv_heads=kv_heads, out="proj", dropout=dropout)
    self.ls_sa = LayerScale(dim)

    self.ca_norm_q = RMSNorm(dim)
    self.cross_attn = Attention(dim, dim, heads=heads, kv_heads=kv_heads, out="proj", kv=False, dropout=dropout)
    self.ls_ca = LayerScale(dim)

    self.ffn = FFN(dim, exp=3, norm=True, bias=False, dropout=dropout)
    self.ls_ffn = LayerScale(dim)

  def __call__(self, q:Tensor, kv:tuple[Tensor, Tensor], q_pos:Tensor|None=None) -> Tensor:
    q = self.ls_sa(q, self.self_attn(self.sa_norm(q)))
    ca_in = self.ca_norm_q(q)
    if q_pos is not None: ca_in = ca_in + q_pos  # reference-point positional query (out of residual stream)
    q = self.ls_ca(q, self.cross_attn(ca_in, kv))
    q = self.ls_ffn(q, self.ffn(q))
    return q

NUM_CLASSES = 17
N_CORNERS = 4
N_BINS = 96
def bin_centers(n_bins:int) -> Tensor:
  return BIN_LO + Tensor.arange(n_bins, dtype=dtypes.float32) / (n_bins - 1) * (BIN_HI - BIN_LO)

class Decoder:
  def __init__(self, dim:int, n_layers:int=4, n_bins:int=N_BINS, heads:int=4, kv_heads:int=1, dropout:float=0.0):
    self.n_bins = n_bins
    self.class_token = Tensor.randn(dim) * 0.02
    self.corner_tokens = Tensor.randn(N_CORNERS, dim) * 0.02
    # position-only query selection over memory tokens: pick the plate token, predict its corner box from its feature
    self.objness = nn.Linear(dim, 1, bias=False)  # scores each memory token for the target-color plate
    self.ref_head = nn.Linear(dim, N_CORNERS * 2, bias=False)  # selected token's feature -> size-aware corner offsets

    self.target_color_embed = nn.Embedding(2, dim)

    self.ca_norm_kv = RMSNorm(dim)
    self.kv_proj = Attention(dim, dim, heads=heads, kv_heads=kv_heads, q=False)

    self.blocks = [DecoderBlock(dim, heads=heads, kv_heads=kv_heads, dropout=dropout) for _ in range(n_layers)]
    self.ln_out = RMSNorm(dim)

    self.class_proj = nn.Linear(dim, NUM_CLASSES, bias=False)
    self.corner_mlp = MLP(dim, 2 * n_bins, dim // 2, blocks=1)  # shared residual head over absolute bins

  def __call__(self, mem:Tensor, sb:Tensor, target_color:Tensor, pos_emb, coords:Tensor) -> tuple[Tensor, list, Tensor]:
    # mem: (B, N_X3_TOKENS+N_X2_TOKENS, D) spatial memory [x3 ; x2]; sb: (B, 1, D) global token (last query)
    # target_color: (B,) int32; pos_emb/coords shared with FeatureTokenizer (same Fourier space as mem).
    # Returns (class_logits, [cum_logits per layer], obj_logit) — obj_logit scores tokens for query selection.
    B, _, D = mem.shape

    target_tok = self.target_color_embed(target_color).reshape(B, 1, D)  # (B, 1, D)
    kv = self.kv_proj.kv_cache(self.ca_norm_kv(mem))  # cross-attn still reads BOTH levels [x3 ; x2]

    # position-only query selection on the COARSE level only (x3 @ /32): propose coarse, refine fine.
    # x3 has the largest receptive field -> best for "where is the plate + how big"; its coarse coordinate is
    # recovered by the refinement loop, and x2 detail still drives the cross-attention. (a single top-1 commits
    # to one level anyway; pooling both levels just lets the finer x2 always win the nearest-centroid target.)
    x3_mem, x3_coords = mem[:, :N_X3_TOKENS], coords[:N_X3_TOKENS]
    obj_logit = self.objness(x3_mem + target_tok).squeeze(-1)  # (B, N_X3_TOKENS)
    onehot = obj_logit.argmax(-1).one_hot(N_X3_TOKENS).cast(mem.dtype)  # (B, N_X3_TOKENS) hard top-1
    sel_feat = onehot.unsqueeze(1) @ x3_mem  # (B, 1, D) selected token's feature
    c0 = onehot.unsqueeze(1) @ x3_coords  # (B, 1, 2) selected token's (u, v)
    ref = (c0 + self.ref_head(sel_feat).reshape(B, N_CORNERS, 2)).clip(0, 1)  # (B, 4, 2) initial corner refs

    class_tok = self.class_token.reshape(1, 1, D).expand(B, 1, D)
    corner_toks = self.corner_tokens.reshape(1, N_CORNERS, D).expand(B, N_CORNERS, D)  # content stays learnable
    q = target_tok.cat(class_tok, corner_toks, sb, dim=1)  # (B, 2+N_CORNERS+1, D) — sb (global register) last

    # fdr like thingy, with iterative reference refinement steering the cross-attention
    centers = bin_centers(self.n_bins)
    cum, cum_layers, qn = None, [], None
    for block in self.blocks:
      q_pos = (q[:, :2] * 0).cat(pos_emb(ref), q[:, :1] * 0, dim=1)  # pos on corner queries only (sharding-safe zeros)
      q = block(q, kv, q_pos)
      qn = self.ln_out(q)
      ct = qn[:, 2:2 + N_CORNERS, :]  # (B, 4, D)
      cum = self.corner_mlp(ct) if cum is None else cum + self.corner_mlp(ct)
      cum_layers.append(cum)
      expected = (cum.reshape(B, N_CORNERS, 2, self.n_bins).softmax(-1) * centers).sum(-1)  # (B, 4, 2)
      ref = expected.clip(0, 1).detach()  # next layer attends where this layer pointed

    class_logits = self.class_proj(qn[:, 1, :])  # (B, NUM_CLASSES)
    return class_logits, cum_layers, obj_logit

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
GOLSD_WEIGHT = 0.125
QUALITY_TAU = 0.4
MAL_GAMMA = 2
MIN_PLATE_SCALE = 0.02
GEOM_WEIGHT = 0.1
DEGEN_FRAC = 0.1  # vp non-degeneracy: penalize perimeter edges shorter than this fraction of the diagonal
OBJ_WEIGHT = 0.5  # query-selection objectness: find the plate token (NOT quality-weighted — must find small plates too)

# image of the absolute conic ω = (KKᵀ)⁻¹ in canonical pixel coords; a rectangle's two 3D edge
# directions are orthogonal, so their image vanishing points satisfy v_hᵀ ω v_v = 0.
_OMEGA = Tensor(np.linalg.inv(CANONICAL_CAMERA_MATRIX @ CANONICAL_CAMERA_MATRIX.T).astype(np.float32))

def _cross(a:Tensor, b:Tensor) -> Tensor:  # batched 3-vector cross product, (...,3)
  ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
  bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
  return Tensor.stack(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx, dim=-1)

def vp_orthogonality_loss(corners:Tensor) -> Tensor:
  # corners: (B, 4, 2) predicted, normalized image coords, order TL, TR, BL, BR. Returns per-sample (B,).
  # Penalizes the squared ω-cosine between the horizontal- and vertical-edge vanishing points, which is
  # 0 exactly when the quad is the perspective image of a right-angled (rectangular) plate.
  px = corners * Tensor([IMG_W, IMG_H], dtype=corners.dtype, device=corners.device)
  pts = px.cat(px[..., :1] * 0 + 1, dim=-1)  # homogeneous coord, derived from px so it shards with the batch
  tl, tr, bl, br = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
  v_h = _cross(_cross(tl, tr), _cross(bl, br))  # horizontal edges' vanishing point
  v_v = _cross(_cross(tl, bl), _cross(tr, br))  # vertical edges' vanishing point
  omega = _OMEGA.to(corners.device)  # match shard layout: a single-device const can't meet a sharded tensor
  def q(a, b): return ((a @ omega) * b).sum(-1)
  cos = q(v_h, v_v) / (q(v_h, v_h) * q(v_v, v_v)).sqrt().add(1e-9)
  # non-degeneracy: cos² is trivially 0 if any edge collapses (its vanishing point -> 0), so penalize
  # perimeter edges much shorter than the diagonal to stop the loss folding the quad into a triangle.
  def seg(i, j): return ((corners[:, i] - corners[:, j]) ** 2).sum(-1).sqrt()
  diag = (seg(0, 3) + seg(1, 2)) * 0.5 + 1e-6
  degen = sum((DEGEN_FRAC - seg(*e) / diag).relu() for e in ((0, 1), (1, 3), (3, 2), (2, 0))) / DEGEN_FRAC
  return cos * cos + degen

class Model:
  def __init__(self, dim:int=192, temporal_size:int=1, sideband_dim:int=512,
               cstage:list[int]=[32, 64, 128, 256], stages:list[tuple[int, int]]=[(1, 0), (2, 0), (6, 0), (0, 2)],
               sr_ratios:list[int]=[1, 1, 2, 1], dropout:float=0.0):
    self.temporal_size = temporal_size
    self.backbone = Backbone(cin=3, cstage=cstage, stages=stages, sideband_dim=sideband_dim,
                             sr_ratios=sr_ratios, temporal_size=temporal_size, dropout=dropout)
    self.feature_tokenizer = FeatureTokenizer(cstage, sideband_dim, dim)
    self.decoder = Decoder(dim, n_layers=3, dropout=dropout)

  def encode(self, img:Tensor) -> tuple[Tensor, Tensor]:
    x0, x1, x2, x3, sb = self.backbone(img)
    return self.feature_tokenizer(x2, x3, sb)

  def __call__(self, img:Tensor, y:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # img: (B, T, IMG_H, IMG_W, 3)
    # Returns (total_loss, class_loss, l1_loss, dfl_loss, lsd_loss, geom_loss, obj_loss).
    assert Tensor.training
    mem, sb = self.encode(img)
    return self.corner_loss(mem, sb, y)

  def corner_loss(self, mem:Tensor, sb:Tensor, y:Tensor) -> Tensor:
    B = mem.shape[0]
    N = self.decoder.n_bins

    # y: [class_id, c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y, has_class, has_corners, target_color]
    class_id = y[:, 0].cast(dtypes.int32)
    corners_f = y[:, 1:9].reshape(B, N_CORNERS, 2).cast(dtypes.float32)  # gt, [BIN_LO, BIN_HI]
    has_class = y[:, 9]
    has_corners = y[:, 10]
    target_color = y[:, 11].cast(dtypes.int32)

    ft = self.feature_tokenizer
    class_logits, cum_layers, obj_logit = self.decoder(mem, sb, target_color, ft.pos_emb, ft.coords)

    denom = has_corners.sum().add(1e-6)
    centers = bin_centers(N)

    # plate scale
    plate_scale = (corners_f.max(1) - corners_f.min(1)).pow(2).sum(-1).sqrt().maximum(MIN_PLATE_SCALE)  # (B,)

    # dfl target (two-hot over bins)
    c_scaled = ((corners_f - BIN_LO) / (BIN_HI - BIN_LO) * (N - 1)).clamp(0, N - 1)  # (B, 4, 2)
    l_idx = c_scaled.floor().cast(dtypes.int32).clamp(0, N - 2)
    w_r = c_scaled - l_idx.cast(dtypes.float32)
    target_dist = l_idx.one_hot(N).cast(dtypes.float32) * (1.0 - w_r).unsqueeze(-1) + (l_idx + 1).one_hot(N).cast(dtypes.float32) * w_r.unsqueeze(-1)

    # deep supervision, l1 scale normalized
    weights = [i + 1 for i in range(len(cum_layers))]
    wsum = float(sum(weights))
    l1_loss, dfl_loss, expected = 0.0, 0.0, None
    for i, cum in enumerate(cum_layers):
      logits = cum.reshape(B, N_CORNERS, 2, N).cast(dtypes.float32)
      expected = (logits.softmax(-1) * centers).sum(-1)  # (B, 4, 2)
      l1_i = (expected - corners_f).abs().mean(-1).mean(-1) / plate_scale  # (B,)
      dfl_i = logits.reshape(-1, N).cross_entropy(target_dist.reshape(-1, N), reduction="none").reshape(B, N_CORNERS, 2).mean(-1).mean(-1)
      l1_loss += (weights[i] / wsum) * (has_corners * l1_i).sum() / denom
      dfl_loss += (weights[i] / wsum) * (has_corners * dfl_i).sum() / denom

    # class loss
    rel_err = (expected - corners_f).abs().mean(-1).mean(-1) / plate_scale  # (B,), final layer
    quality = has_corners * (-rel_err / QUALITY_TAU).exp() + (1 - has_corners)  # (B,)
    y_onehot = class_id.one_hot(NUM_CLASSES).cast(dtypes.float32)
    mal = mal_loss(class_logits.cast(dtypes.float32), y_onehot, quality.reshape(B, 1), gamma=MAL_GAMMA)  # (B,)
    class_loss = (has_class * mal).sum() / has_class.sum().add(1e-6)

    # go-lsd self-distillation loss
    teacher = cum_layers[-1].reshape(B, N_CORNERS, 2, N).cast(dtypes.float32).softmax(-1).detach()
    lsd_loss = 0.0
    for i in range(len(cum_layers) - 1):
      student = cum_layers[i].reshape(B, N_CORNERS, 2, N).cast(dtypes.float32)
      lsd_i = student.reshape(-1, N).cross_entropy(teacher.reshape(-1, N), reduction="none").reshape(B, N_CORNERS, 2).mean(-1).mean(-1)
      lsd_loss += (has_corners * lsd_i).sum() / denom
    lsd_loss = lsd_loss / max(1, len(cum_layers) - 1)

    # geometric prior — final-layer corners should be the perspective image of a rectangle (vp-orthogonality)
    geom_loss = (has_corners * vp_orthogonality_loss(expected)).sum() / denom

    # query-selection objectness — pick the x3 token nearest the plate centroid (softmax over the coarse level)
    centroid = corners_f.mean(1)  # (B, 2)
    x3_coords = ft.coords[:N_X3_TOKENS]  # selection pool is the coarse level only
    d2 = ((x3_coords.reshape(1, -1, 2) - centroid.reshape(B, 1, 2)) ** 2).sum(-1)  # (B, N_X3_TOKENS)
    obj_target = d2.argmin(-1).one_hot(obj_logit.shape[-1]).cast(dtypes.float32)  # (B, N_X3_TOKENS)
    obj_i = obj_logit.cast(dtypes.float32).cross_entropy(obj_target, reduction="none")  # (B,)
    obj_loss = (has_corners * obj_i).sum() / denom

    total = CLASS_WEIGHT * class_loss + L1_WEIGHT * l1_loss + DFL_WEIGHT * dfl_loss + GOLSD_WEIGHT * lsd_loss + GEOM_WEIGHT * geom_loss + OBJ_WEIGHT * obj_loss
    return total, class_loss, l1_loss, dfl_loss, lsd_loss, geom_loss, obj_loss

  def corner_predict(self, mem:Tensor, sb:Tensor, target_color:Tensor) -> Tensor:
    B = mem.shape[0]
    N = self.decoder.n_bins

    ft = self.feature_tokenizer
    class_logits, cum_layers, _ = self.decoder(mem, sb, target_color, ft.pos_emb, ft.coords)

    class_probs = class_logits.cast(dtypes.float32).sigmoid()  # one-vs-all; max = quality-aware confidence
    class_id = class_probs.argmax(-1, keepdim=True).cast(dtypes.default_float)
    class_conf = class_probs.max(-1, keepdim=True).cast(dtypes.default_float)

    centers = bin_centers(N)
    dist = cum_layers[-1].reshape(B, N_CORNERS, 2, N).cast(dtypes.float32).softmax(-1)  # (B,4,2,N)
    mean = (dist * centers).sum(-1)  # (B,4,2)
    cdf = dist.cumsum(-1)
    def _quant(q:float) -> Tensor:
      idx = (cdf < q).sum(-1).clip(0, N - 1).cast(dtypes.int32)  # (B,4,2)
      return (idx.one_hot(N).cast(dtypes.float32) * centers).sum(-1)
    corners = mean.reshape(B, N_CORNERS * 2).cast(dtypes.default_float)
    q_lo = _quant(0.159).reshape(B, N_CORNERS * 2).cast(dtypes.default_float)
    q_hi = _quant(0.841).reshape(B, N_CORNERS * 2).cast(dtypes.default_float)
    return Tensor.cat(class_id, class_conf, corners, q_lo, q_hi, dim=-1)

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
  from .common import pred, TemporalInference
  import time

  if getenv("HALF"):
    dtypes.default_float = dtypes.float16

  model = Model(temporal_size=T)
  if getenv("FUSE"): model.fuse()
  infer = TemporalInference(pred, T=T, model=model)

  with Context(DEBUG=getenv("DEBUG", 2)):
    model_fn = infer.warmup()

  infer = TemporalInference(model_fn, T=T)

  # full runs
  tms = []
  fake_frame = Tensor.empty(IMG_H * IMG_W * 3, dtype=dtypes.uint8, device="PYTHON").clone().realize()
  for _ in range(15):
    GlobalCounters.reset()
    st = time.perf_counter()
    ret = infer(fake_frame, 0)
    tms.append(time.perf_counter() - st)

  print("jit run successful")
  print()

  tms = tms[5:]

  print(f"average time: {sum(tms) / len(tms):.4f} seconds")
  print(f"average latency: {(sum(tms) / len(tms)) * 1000:.2f} ms")
  print(f"average fps: {1 / (sum(tms) / len(tms)):.2f} fps")

  print(f"fastest time: {min(tms):.4f} seconds")
  print(f"fastest latency: {min(tms) * 1000:.2f} ms")
  print(f"fastest fps: {1 / min(tms):.2f} fps")

  print(f"slowest time: {max(tms):.4f} seconds")
  print(f"slowest latency: {max(tms) * 1000:.2f} ms")
  print(f"slowest fps: {1 / max(tms):.2f} fps")

  print(f"model size: {sum(p.numel() * p.dtype.itemsize for p in get_parameters(model)) / 1024**2:.2f} MB")

  print(f"model parameters: {sum(p.numel() for p in get_parameters(model))}")
  print(f"backbone parameters: {sum(p.numel() for p in get_parameters(model.backbone))}")
  print(f"feature_tokenizer parameters: {sum(p.numel() for p in get_parameters(model.feature_tokenizer))}")
  print(f"decoder parameters: {sum(p.numel() for p in get_parameters(model.decoder))}")

  print(f"model gflops: {GlobalCounters.global_ops * 1e-9:.2f} GFLOPs")
