"""Native-tinygrad BiSeNetV2 for floor / free-space segmentation (occupancyd Phase 2).

Mirrors github.com/CoinCheung/BiSeNet `lib/models/bisenetv2.py` module-for-module so the
pretrained ADE20K state_dict loads 1:1 — EVAL path only (detail + segment + bga + head);
the `aux*` heads are training-only and dropped (`load_state_dict(strict=False)`). Pure
Conv2d + BatchNorm + ReLU with nearest/bilinear upsample — no attention/deformable/custom
ops. Convert weights with `cv/floorseg/convert.py`; consumed by occupancyd's seg backend.
"""
from tinygrad import nn
from tinygrad.tensor import Tensor

from ..common.tensor import upsample   # exact nearest x-scale (block replicate), == nn.Upsample(nearest)

# ADE20K (150-class) indices treated as walkable ground → "free". floor=3 is primary;
# road/sidewalk/earth/rug/field/path included so varied indoor mats still read as floor.
# Tune per venue (or fine-tune the head). Everything else → obstacle.
ADE_FREE_CLASSES = (3, 6, 11, 13, 28, 29, 52)

# CoinCheung ADE20K normalisation (RGB, /255 then (x-mean)/std) — from configs/bisenetv2_ade20k.py.
IM_MEAN = (0.49343686, 0.46819251, 0.43106987)
IM_STD = (0.25680734, 0.25051928, 0.24334113)

class ConvBNReLU:
  def __init__(self, cin:int, cout:int, ks:int=3, stride:int=1, padding:int=1, dilation:int=1, groups:int=1):
    self.conv = nn.Conv2d(cin, cout, ks, stride, padding, dilation=dilation, groups=groups, bias=False)
    self.bn = nn.BatchNorm(cout)
  def __call__(self, x:Tensor) -> Tensor: return self.bn(self.conv(x)).relu()

class DetailBranch:
  def __init__(self):
    self.S1 = [ConvBNReLU(3, 64, 3, 2), ConvBNReLU(64, 64, 3, 1)]
    self.S2 = [ConvBNReLU(64, 64, 3, 2), ConvBNReLU(64, 64, 3, 1), ConvBNReLU(64, 64, 3, 1)]
    self.S3 = [ConvBNReLU(64, 128, 3, 2), ConvBNReLU(128, 128, 3, 1), ConvBNReLU(128, 128, 3, 1)]
  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential(self.S1).sequential(self.S2).sequential(self.S3)

class StemBlock:
  def __init__(self):
    self.conv = ConvBNReLU(3, 16, 3, 2)
    self.left = [ConvBNReLU(16, 8, 1, 1, 0), ConvBNReLU(8, 16, 3, 2)]
    self.fuse = ConvBNReLU(32, 16, 3, 1)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.conv(x)
    left = x.sequential(self.left)
    right = x.max_pool2d(kernel_size=3, stride=2, padding=1)   # `right` branch (no params)
    return self.fuse(left.cat(right, dim=1))

class CEBlock:
  def __init__(self):
    self.bn = nn.BatchNorm(128)
    self.conv_gap = ConvBNReLU(128, 128, 1, 1, 0)
    self.conv_last = ConvBNReLU(128, 128, 3, 1)
  def __call__(self, x:Tensor) -> Tensor:
    feat = self.conv_gap(self.bn(x.mean((2, 3), keepdim=True)))
    return self.conv_last(feat + x)

class GELayerS1:
  def __init__(self, cin:int, cout:int, exp:int=6):
    mid = cin * exp
    self.conv1 = ConvBNReLU(cin, cin, 3, 1)
    self.dwconv = [nn.Conv2d(cin, mid, 3, 1, 1, groups=cin, bias=False), nn.BatchNorm(mid)]
    self.conv2 = [nn.Conv2d(mid, cout, 1, 1, 0, bias=False), nn.BatchNorm(cout)]
  def __call__(self, x:Tensor) -> Tensor:
    feat = self.conv1(x)
    feat = self.dwconv[1](self.dwconv[0](feat)).relu()
    feat = self.conv2[1](self.conv2[0](feat))
    return (feat + x).relu()

class GELayerS2:
  def __init__(self, cin:int, cout:int, exp:int=6):
    mid = cin * exp
    self.conv1 = ConvBNReLU(cin, cin, 3, 1)
    self.dwconv1 = [nn.Conv2d(cin, mid, 3, 2, 1, groups=cin, bias=False), nn.BatchNorm(mid)]
    self.dwconv2 = [nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False), nn.BatchNorm(mid)]
    self.conv2 = [nn.Conv2d(mid, cout, 1, 1, 0, bias=False), nn.BatchNorm(cout)]
    self.shortcut = [nn.Conv2d(cin, cin, 3, 2, 1, groups=cin, bias=False), nn.BatchNorm(cin),
                     nn.Conv2d(cin, cout, 1, 1, 0, bias=False), nn.BatchNorm(cout)]
  def __call__(self, x:Tensor) -> Tensor:
    feat = self.conv1(x)
    feat = self.dwconv1[1](self.dwconv1[0](feat))
    feat = self.dwconv2[1](self.dwconv2[0](feat)).relu()
    feat = self.conv2[1](self.conv2[0](feat))
    s = self.shortcut[1](self.shortcut[0](x))
    s = self.shortcut[3](self.shortcut[2](s))
    return (feat + s).relu()

class SegmentBranch:
  def __init__(self):
    self.S1S2 = StemBlock()
    self.S3 = [GELayerS2(16, 32), GELayerS1(32, 32)]
    self.S4 = [GELayerS2(32, 64), GELayerS1(64, 64)]
    self.S5_4 = [GELayerS2(64, 128), GELayerS1(128, 128), GELayerS1(128, 128), GELayerS1(128, 128)]
    self.S5_5 = CEBlock()
  def __call__(self, x:Tensor) -> Tensor:   # eval path needs only the deepest feature (feat_s)
    feat = self.S1S2(x)
    feat = feat.sequential(self.S3).sequential(self.S4).sequential(self.S5_4)
    return self.S5_5(feat)

class BGALayer:
  def __init__(self):
    self.left1 = [nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False), nn.BatchNorm(128), nn.Conv2d(128, 128, 1, 1, 0, bias=False)]
    self.left2 = [nn.Conv2d(128, 128, 3, 2, 1, bias=False), nn.BatchNorm(128)]
    self.right1 = [nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm(128)]
    self.right2 = [nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False), nn.BatchNorm(128), nn.Conv2d(128, 128, 1, 1, 0, bias=False)]
    self.conv = [nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm(128)]
  def __call__(self, x_d:Tensor, x_s:Tensor) -> Tensor:
    left1 = self.left1[2](self.left1[1](self.left1[0](x_d)))
    left2 = self.left2[1](self.left2[0](x_d)).avg_pool2d(kernel_size=3, stride=2, padding=1)
    right1 = self.right1[1](self.right1[0](x_s))
    right2 = self.right2[2](self.right2[1](self.right2[0](x_s)))
    left = left1 * upsample(right1, 4).sigmoid()      # up1 (nearest x4)
    right = upsample(left2 * right2.sigmoid(), 4)      # up2 (nearest x4)
    return self.conv[1](self.conv[0](left + right)).relu()

class _Identity:
  def __call__(self, x:Tensor) -> Tensor: return x

class _Bilinear:
  def __init__(self, factor:int): self.factor = factor
  def __call__(self, x:Tensor) -> Tensor:
    return x.interpolate((x.shape[2] * self.factor, x.shape[3] * self.factor), mode="linear")

class SegmentHead:
  """Main (non-aux) head: ConvBNReLU → classifier 1x1 → bilinear x8. conv_out[1] is the
  classifier (matches the checkpoint key `head.conv_out.1.*`); [0]/[2] are param-less."""
  def __init__(self, cin:int, mid:int, n_classes:int, up_factor:int=8):
    self.conv = ConvBNReLU(cin, mid, 3, 1)
    self.conv_out = [_Identity(), nn.Conv2d(mid, n_classes, 1, 1, 0, bias=True), _Bilinear(up_factor)]
  def logits(self, x:Tensor) -> Tensor:   # class logits at /8 (pre-upsample) — cheaper for occupancyd
    return self.conv_out[1](self.conv(x))
  def __call__(self, x:Tensor) -> Tensor:
    return self.conv_out[2](self.logits(x))

class BiSeNetV2:
  def __init__(self, n_classes:int=150):
    self.detail = DetailBranch()
    self.segment = SegmentBranch()
    self.bga = BGALayer()
    self.head = SegmentHead(128, 1024, n_classes, up_factor=8)

  def __call__(self, x:Tensor) -> Tensor:                # full-res class logits (matches ref eval)
    return self.head(self.bga(self.detail(x), self.segment(x)))

  def logits8(self, x:Tensor) -> Tensor:                 # class logits at /8 (occupancyd uses this)
    return self.head.logits(self.bga(self.detail(x), self.segment(x)))

# --- ADE20K 150 class names (subset; the ground-like ones matter for ADE_FREE_CLASSES) ---
ADE_NAMES = {0: "wall", 3: "floor", 5: "ceiling", 6: "road", 9: "grass", 11: "sidewalk",
             13: "earth/ground", 28: "rug", 29: "field", 46: "sand", 52: "path", 53: "stairs",
             54: "runway", 68: "hill", 91: "dirt-track", 94: "land"}

if __name__ == "__main__":
  # Inspect what the seg net predicts on a real arena frame → tune ADE_FREE_CLASSES.
  #   python -m cv.floorseg.model <image.png>
  import sys
  from pathlib import Path
  import cv2, numpy as np
  from tinygrad.nn.state import safe_load, load_state_dict
  Tensor.training = False
  m = BiSeNetV2(150)
  weights = Path(__file__).resolve().parents[2] / "weights" / "floorseg.safetensors"
  load_state_dict(m, safe_load(str(weights)), strict=False, verbose=False)
  img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
  h, w = img.shape[:2]; h -= h % 32; w -= w % 32; img = cv2.resize(img, (w, h))
  mean, std = np.asarray(IM_MEAN, np.float32) * 255, np.asarray(IM_STD, np.float32) * 255
  x = ((img.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
  ids = m(Tensor(x)).numpy()[0].argmax(0)                # full-res class ids
  u, c = np.unique(ids, return_counts=True)
  print(f"{sys.argv[1]}  {w}x{h}  top classes:")
  for i in np.argsort(-c)[:12]:
    print(f"  class {u[i]:3d}  {ADE_NAMES.get(int(u[i]), '?'):14s} {100*c[i]/ids.size:5.1f}%")
  print(f"floor frac (ADE_FREE_CLASSES={ADE_FREE_CLASSES}): {np.isin(ids, ADE_FREE_CLASSES).mean():.3f}")
  ov = img.copy(); ov[np.isin(ids, ADE_FREE_CLASSES)] = (0, 255, 0)
  out = "/tmp/floorseg_overlay.png"; cv2.imwrite(out, cv2.cvtColor((0.5 * img + 0.5 * ov).astype(np.uint8), cv2.COLOR_RGB2BGR))
  print(f"overlay (green=free) → {out}")
