from functools import partial
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import csv

from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.helpers import tqdm, getenv

from ..common import BASE_PATH

# Input image dimensions used across the autoaim pipeline (model, data, syndata, camerad path).
IMG_H, IMG_W = 256, 512

# Backbone reduces spatial dims by this factor: stem(/4 — DWT × stride-2 conv) × 3 stage downsamples(/8).
# Update if cstage / patch_size / num stages change.
BACKBONE_STRIDE = 32
X3_H, X3_W = IMG_H // BACKBONE_STRIDE, IMG_W // BACKBONE_STRIDE
N_X3_TOKENS = X3_H * X3_W
N_FEAT_TOKENS = N_X3_TOKENS + 1  # +1 for the sideband token

# Temporal window (number of frames fused at the stem). Configurable via env var so it stays consistent
# across data loading, training, and inference (model/data/train/autoaimd all import from here).
T = getenv("T", 1)

MODEL_VERSION = 16

@partial(TinyJit, prune=True)
def pred_backbone(model, frames):
  """frames: (1, T, H, W, 3) — T raw frames. Returns feat tokens (1, N_FEAT_TOKENS, D)."""
  frames = frames.to(Device.DEFAULT)
  return model.encode(frames).to("CPU")

@partial(TinyJit, prune=True)
def pred_decoder(model, feat_tokens):
  """Feature tokens (1, N_FEAT_TOKENS, D) → output (1, 10):
  [class_id, confidence, c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y] with corners in [0, 1]."""
  feat_tokens = feat_tokens.to(Device.DEFAULT)
  return model.corner_predict(feat_tokens).to("CPU")

class TemporalInference:
  """Stateful temporal inference: buffers the last T raw frames and fuses them at the stem
  via Haar temporal DWT. One backbone pass per call (not T)."""
  def __init__(self, backbone_fn, decoder_fn, model, T:int):
    self.backbone_fn = backbone_fn
    self.decoder_fn = decoder_fn
    self.model = model
    self.T = T
    self.frame_buffer: deque = deque(maxlen=T)

  def reset(self):
    self.frame_buffer.clear()

  def __call__(self, img) -> list:
    """Process a single frame with temporal context. Returns model output as a list."""
    self.frame_buffer.append(img)
    # cold-start: pad with the oldest available frame until the buffer is full
    while len(self.frame_buffer) < self.T:
      self.frame_buffer.appendleft(self.frame_buffer[0])

    frames = Tensor.stack(*list(self.frame_buffer), dim=0).unsqueeze(0)  # (1, T, H, W, 3)
    feat_tokens = self.backbone_fn(self.model, frames)
    return self.decoder_fn(self.model, feat_tokens).tolist()[0]

@dataclass
class Annotation:
  detected: int
  x: float
  y: float

annotations_csv = {}
def get_annotation(img_file) -> Annotation:
  global annotations_csv

  # default
  detected, x, y = 0, 0.0, 0.0

  # if there is a img_file.txt file, read that
  if Path(img_file).with_suffix(".txt").exists():
    with open(Path(img_file).with_suffix(".txt"), "r") as f:
      line = f.readline().strip()
      line = line.split(" ")
      detected, x, y = int(line[0]), float(line[1]), float(line[2])
  else:
    basename = ".".join(Path(img_file).name.split(".")[:-2])
    if basename not in annotations_csv:
      with open(BASE_PATH / "data" / basename / f"{basename}.csv", "r") as f:
        tqdm.write(f"reading annotation file {BASE_PATH / 'data' / basename / f'{basename}.csv'}")
        # read the annotation file
        reader = csv.reader(f)
        # skip the header
        _ = next(reader)
        annotations_csv[basename] = [(int(row[0]), int(row[1]), float(row[2]), float(row[3])) for row in reader]

    # check that frame index matches
    frame_index = int(Path(img_file).name.split(".")[-2]) - 1
    assert frame_index == annotations_csv[basename][frame_index][0]

    detected, x, y = annotations_csv[basename][frame_index][1:]

  # return annotation
  return Annotation(detected, x, y)
