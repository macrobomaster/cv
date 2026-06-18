import math

from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from ..common.tensor import pixel_unshuffle
from ..common.nn import Attention, FFN, MLP, LayerScale, LayerScale2d
from ..common.nn.fuse import FusedBlock
from ..common.nn.norm import GRN, RMSNorm, RMSNorm2d
from .common import IMG_H, IMG_W, N_X2_TOKENS, N_X3_TOKENS, N_FEAT_TOKENS, T, BIN_LO, BIN_HI

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
  def __init__(self, dim:int, sideband_dim:int, dropout:float=0.0):
    self.dropout = dropout
    self.sideband, self.sideband_dim = sideband_dim // dim, sideband_dim

    self.cpe = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

    self.tnorm = RMSNorm(dim)
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

    # concat sideband to tokens
    xx = x.flatten(2).transpose(1, 2)
    xx = xx.cat(sb.reshape(b, self.sideband, c), dim=1)

    # run token mixer
    xx = self.token_mixer(self.tnorm(xx))

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

class Downsample(FusedBlock):
  def __init__(self, cin:int, cout:int, shortcut:bool=True):
    self.cout, self.shortcut = cout, shortcut
    self.norm = RMSNorm2d(cin)
    self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=True)
    self.dw3x3 = nn.Conv2d(cout, cout, 3, 2, 1, groups=cout, bias=True)
    self.dw7x7 = nn.Conv2d(cout, cout, 7, 2, 3, groups=cout, bias=True)
    self.ls = LayerScale2d(cout)

  def __call__(self, x:Tensor) -> Tensor:
    xx = self.norm(x)
    xx = self.pw(xx).gelu()
    if not self.fused:
      xx = (self.dw3x3(xx) + self.dw7x7(xx)).gelu()
    else:
      xx = self.conv(xx).gelu()

    if self.shortcut:
      x = pixel_unshuffle(x, 2)
      b, c, h, w = x.shape
      x = x.reshape(b, xx.shape[1], c // xx.shape[1], h, w)
      x = x.mean(2)
      x = self.ls(x, xx)
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

class FeatureTokenizer:
  def __init__(self, cstage:list[int], sideband_dim:int, dim:int):
    self.norm_x2 = RMSNorm(cstage[2])
    self.proj_x2 = nn.Linear(cstage[2], dim, bias=False)
    self.x2_pos_emb = Tensor.randn(N_X2_TOKENS, dim) * 0.02

    self.norm_x3 = RMSNorm(cstage[3])
    self.proj_x3 = nn.Linear(cstage[3], dim, bias=False)
    self.x3_pos_emb = Tensor.randn(N_X3_TOKENS, dim) * 0.02

    self.norm_sb = RMSNorm(sideband_dim)
    self.proj_sb = nn.Linear(sideband_dim, dim, bias=False)

  def __call__(self, x2:Tensor, x3:Tensor, sb:Tensor) -> tuple[Tensor, Tensor]:
    fine_tokens = self.proj_x2(self.norm_x2(x2.flatten(2).transpose(1, 2))) + self.x2_pos_emb

    x3_tokens = self.proj_x3(self.norm_x3(x3.flatten(2).transpose(1, 2))) + self.x3_pos_emb
    sb_token = self.proj_sb(self.norm_sb(sb)).unsqueeze(1)
    feat_tokens = Tensor.cat(x3_tokens, sb_token, dim=1)

    return feat_tokens, fine_tokens

class DecoderBlock:
  def __init__(self, dim:int, heads:int=4, kv_heads:int=1, dropout:float=0.0):
    self.sa_norm = RMSNorm(dim)
    self.self_attn = Attention(dim, dim, heads=heads, kv_heads=kv_heads, out="proj", dropout=dropout)
    self.ls_sa = LayerScale(dim)

    self.ca_norm_q = RMSNorm(dim)
    self.ca_norm_kv = RMSNorm(dim)
    self.cross_attn = Attention(dim, dim, heads=heads, kv_heads=kv_heads, out="proj", dropout=dropout)
    self.ls_ca = LayerScale(dim)

    self.ffn = FFN(dim, exp=2, norm=True, bias=False, dropout=dropout)
    self.ls_ffn = LayerScale(dim)

  def __call__(self, q:Tensor, mem:Tensor) -> Tensor:
    q = self.ls_sa(q, self.self_attn(self.sa_norm(q)))
    q = self.ls_ca(q, self.cross_attn(self.ca_norm_q(q), self.ca_norm_kv(mem)))
    q = self.ls_ffn(q, self.ffn(q))
    return q

NUM_CLASSES = 17
N_CORNERS = 4
N_BINS = 96
def bin_centers(n_bins:int) -> Tensor:
  return BIN_LO + Tensor.arange(n_bins, dtype=dtypes.float32) / (n_bins - 1) * (BIN_HI - BIN_LO)

class Decoder:
  def __init__(self, dim:int, n_layers:int=4, n_bins:int=N_BINS, dropout:float=0.0):
    self.n_bins = n_bins
    self.pos_emb = Tensor.randn(N_FEAT_TOKENS, dim) * 0.02
    self.class_token = Tensor.randn(dim) * 0.02
    self.corner_tokens = Tensor.randn(N_CORNERS, dim) * 0.02

    self.target_color_embed = nn.Embedding(2, dim)

    self.blocks = [DecoderBlock(dim, dropout=dropout) for _ in range(n_layers)]
    self.ln_out = RMSNorm(dim)

    self.class_proj = nn.Linear(dim, NUM_CLASSES, bias=False)
    self.corner_mlp = MLP(dim, 2 * n_bins, dim // 2, blocks=1)  # shared residual head over absolute bins

  def __call__(self, feat_tokens:Tensor, fine_tokens:Tensor, target_color:Tensor) -> tuple[Tensor, list]:
    # feat_tokens: (B, N_FEAT_TOKENS, D) - coarse /32 (+sb); fine_tokens: (B, N_X2_TOKENS, D) - fine /16
    # target_color: (B,) int32. Returns (class_logits, [cum_logits per layer]) - accumulated corner logits.
    B, _, D = feat_tokens.shape

    feat_tokens = feat_tokens + self.pos_emb.reshape(1, N_FEAT_TOKENS, D).expand(B, -1, -1)
    mem = feat_tokens.cat(fine_tokens, dim=1)  # (B, N_FEAT+N_X2, D)

    target_tok = self.target_color_embed(target_color).reshape(B, 1, D)  # (B, 1, D)
    class_tok = self.class_token.reshape(1, 1, D).expand(B, 1, D)
    corner_toks = self.corner_tokens.reshape(1, N_CORNERS, D).expand(B, N_CORNERS, D)
    q = target_tok.cat(class_tok, corner_toks, dim=1)  # (B, 2+N_CORNERS, D)

    # fdr like thingy
    cum, cum_layers, qn = None, [], None
    for block in self.blocks:
      q = block(q, mem)
      qn = self.ln_out(q)
      ct = qn[:, 2:2 + N_CORNERS, :]  # (B, 4, D)
      cum = self.corner_mlp(ct) if cum is None else cum + self.corner_mlp(ct)
      cum_layers.append(cum)

    class_logits = self.class_proj(qn[:, 1, :])  # (B, NUM_CLASSES)
    return class_logits, cum_layers

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

class Model:
  def __init__(self, dim:int=512, temporal_size:int=1, sideband_dim:int=512,
               cstage:list[int]=[32, 64, 128, 256], stages:list[tuple[int, int]]=[(2, 0), (2, 0), (6, 2), (0, 2)],
               dropout:float=0.0):
    self.temporal_size = temporal_size
    self.backbone = Backbone(cin=3, cstage=cstage, stages=stages, sideband_dim=sideband_dim,
                             temporal_size=temporal_size, dropout=dropout)
    self.feature_tokenizer = FeatureTokenizer(cstage, sideband_dim, dim)
    self.decoder = Decoder(dim, n_layers=4, dropout=dropout)

  def encode(self, img:Tensor) -> tuple[Tensor, Tensor]:
    x0, x1, x2, x3, sb = self.backbone(img)
    return self.feature_tokenizer(x2, x3, sb)

  def __call__(self, img:Tensor, y:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # img: (B, T, IMG_H, IMG_W, 3)
    # Returns (total_loss, class_loss, l1_loss, dfl_loss, lsd_loss).
    assert Tensor.training
    tokens = self.encode(img)  # (feat_tokens, fine_tokens)
    return self.corner_loss(tokens, y)

  def corner_loss(self, tokens:tuple[Tensor, Tensor], y:Tensor) -> Tensor:
    feat_tokens, fine_tokens = tokens
    B = feat_tokens.shape[0]
    N = self.decoder.n_bins

    # y: [class_id, c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y, has_class, has_corners, target_color]
    class_id = y[:, 0].cast(dtypes.int32)
    corners_f = y[:, 1:9].reshape(B, N_CORNERS, 2).cast(dtypes.float32)  # gt, [BIN_LO, BIN_HI]
    has_class = y[:, 9]
    has_corners = y[:, 10]
    target_color = y[:, 11].cast(dtypes.int32)

    class_logits, cum_layers = self.decoder(feat_tokens, fine_tokens, target_color)

    # class loss
    ce = class_logits.cast(dtypes.float32).cross_entropy(class_id, reduction="none")  # (B,)
    class_loss = (has_class * ce).sum() / has_class.sum().add(1e-6)

    denom = has_corners.sum().add(1e-6)
    centers = bin_centers(N)

    # dfl loss
    c_scaled = ((corners_f - BIN_LO) / (BIN_HI - BIN_LO) * (N - 1)).clamp(0, N - 1)  # (B, 4, 2)
    l_idx = c_scaled.floor().cast(dtypes.int32).clamp(0, N - 2)
    w_r = c_scaled - l_idx.cast(dtypes.float32)
    target_dist = l_idx.one_hot(N).cast(dtypes.float32) * (1.0 - w_r).unsqueeze(-1) + (l_idx + 1).one_hot(N).cast(dtypes.float32) * w_r.unsqueeze(-1)

    # deep supervision
    weights = [i + 1 for i in range(len(cum_layers))]
    wsum = float(sum(weights))
    l1_loss, dfl_loss = 0.0, 0.0
    for i, cum in enumerate(cum_layers):
      logits = cum.reshape(B, N_CORNERS, 2, N).cast(dtypes.float32)
      expected = (logits.softmax(-1) * centers).sum(-1)  # (B, 4, 2)
      l1_i = (expected - corners_f).abs().mean(-1).mean(-1)  # (B,)
      dfl_i = logits.reshape(-1, N).cross_entropy(target_dist.reshape(-1, N), reduction="none").reshape(B, N_CORNERS, 2).mean(-1).mean(-1)
      l1_loss += (weights[i] / wsum) * (has_corners * l1_i).sum() / denom
      dfl_loss += (weights[i] / wsum) * (has_corners * dfl_i).sum() / denom

    # go-lsd self-distillation loss
    teacher = cum_layers[-1].reshape(B, N_CORNERS, 2, N).cast(dtypes.float32).softmax(-1).detach()
    lsd_loss = 0.0
    for i in range(len(cum_layers) - 1):
      student = cum_layers[i].reshape(B, N_CORNERS, 2, N).cast(dtypes.float32)
      lsd_i = student.reshape(-1, N).cross_entropy(teacher.reshape(-1, N), reduction="none").reshape(B, N_CORNERS, 2).mean(-1).mean(-1)
      lsd_loss += (has_corners * lsd_i).sum() / denom
    lsd_loss = lsd_loss / max(1, len(cum_layers) - 1)

    total = CLASS_WEIGHT * class_loss + L1_WEIGHT * l1_loss + DFL_WEIGHT * dfl_loss + GOLSD_WEIGHT * lsd_loss
    return total, class_loss, l1_loss, dfl_loss, lsd_loss

  def corner_predict(self, tokens:tuple[Tensor, Tensor], target_color:Tensor) -> Tensor:
    feat_tokens, fine_tokens = tokens
    B = feat_tokens.shape[0]
    N = self.decoder.n_bins

    class_logits, cum_layers = self.decoder(feat_tokens, fine_tokens, target_color)

    class_probs = class_logits.cast(dtypes.float32).softmax(-1)
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
  from .common import pred
  import time

  if getenv("HALF"):
    dtypes.default_float = dtypes.float16

  model = Model(temporal_size=T)
  if getenv("FUSE"): model.fuse()

  with Context(DEBUG=getenv("DEBUG", 2)):
    for _ in range(3):
      fake_frames = Tensor.empty(T, IMG_H, IMG_W, 3, dtype=dtypes.uint8).realize()
      fake_frame = Tensor.empty(IMG_H * IMG_W * 3, dtype=dtypes.uint8, device="PYTHON").realize()
      fake_target = Tensor([0], dtype=dtypes.int32, device="PYTHON").realize()
      GlobalCounters.reset()
      ret = pred(model, fake_frames, fake_frame, fake_target)
      print(ret)

  # full runs
  tms = []
  for _ in range(15):
    fake_frames = Tensor.empty(T, IMG_H, IMG_W, 3, dtype=dtypes.uint8).realize()
    fake_frame = Tensor.empty(IMG_H * IMG_W * 3, dtype=dtypes.uint8, device="PYTHON").realize()
    fake_target = Tensor([0], dtype=dtypes.int32, device="PYTHON").realize()
    GlobalCounters.reset()
    st = time.perf_counter()
    ret = pred(model, fake_frames, fake_frame, fake_target)
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
