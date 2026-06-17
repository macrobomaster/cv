from functools import partial
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import csv

import numpy as np
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import tqdm, getenv

from ..common import BASE_PATH

# input image dims
IMG_H, IMG_W = 256, 512

# Backbone reduces spatial dims by this factor: stem(/4 — DWT × stride-2 conv) × 3 stage downsamples(/8).
# Update if cstage / patch_size / num stages change.
BACKBONE_STRIDE = 32
X3_H, X3_W = IMG_H // BACKBONE_STRIDE, IMG_W // BACKBONE_STRIDE
N_X3_TOKENS = X3_H * X3_W
N_FEAT_TOKENS = N_X3_TOKENS + 1  # +1 for the sideband token

# Fine feature map (x2, one stage up from x3) — /16 stride. Corner queries cross-attend to this
# finer map for localization; x3 at /32 (32px tokens) is too coarse to resolve small-plate corners.
X2_STRIDE = 16
X2_H, X2_W = IMG_H // X2_STRIDE, IMG_W // X2_STRIDE
N_X2_TOKENS = X2_H * X2_W

# Temporal window (number of frames fused at the stem). Configurable via env var so it stays consistent
# across data loading, training, and inference (model/data/train/autoaimd all import from here).
T = getenv("T", 1)

# corner output valid range
BIN_LO, BIN_HI = -0.5, 1.5

MODEL_VERSION = 27

# canonical camera
CANONICAL_FX_FY = 650
CANONICAL_CX, CANONICAL_CY = IMG_W / 2, IMG_H / 2
CANONICAL_CAMERA_MATRIX = np.array([[CANONICAL_FX_FY, 0, CANONICAL_CX],
                                    [0, CANONICAL_FX_FY, CANONICAL_CY],
                                    [0, 0, 1]], dtype=np.float32)
CANONICAL_DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)

# Ballistics — drag-free projectile model.
MUZZLE_VELOCITY = 28.0   # m/s, effective. TODO: calibrate.
GRAVITY = 9.81           # m/s^2

# Pipeline latency budget for the aim/fire prediction. Seconds (rad/s where noted).
DELTA_INPUT = 0.020      # s, UART → main-board command pickup. TODO: measure.
DELTA_TRIGGER = 0.060    # s, feeder + flywheel pickup. TODO: measure.
GIMBAL_TAU = 0.040       # s, gimbal first-order motor constant. TODO: fit.
GIMBAL_OMEGA_MAX = 12.0  # rad/s, gimbal slew rate ceiling. TODO: fit.

# Mount: camera frame → gimbal-end-effector frame (R applied to a column vector).
R_MOUNT = np.eye(3)      # rotation. TODO: measure.
T_MOUNT = np.zeros(3)    # translation, m. TODO: measure.

@partial(TinyJit, prune=True)
def pred(model, frames, frame, target_color):
  frame = frame.to(Device.DEFAULT).reshape(256, 512, 3)
  frames.assign(Tensor.cat(frames[1:], frame.unsqueeze(0))).realize()

  target_color = target_color.to(Device.DEFAULT)
  tokens = model.encode(frames.unsqueeze(0))
  return model.corner_predict(tokens, target_color).float().to("CPU")

class TemporalInference:
  def __init__(self, model_fn, T:int, model=None):
    self.model_fn, self.model = model_fn, model
    self.T = T
    self.frames = Tensor.zeros(T, IMG_H, IMG_W, 3, dtype=dtypes.uint8).clone()

  def __call__(self, img, target_color:int=0) -> list:
    target_color_t = Tensor([target_color], dtype=dtypes.int32, device="PYTHON")
    return self.model_fn(self.model, self.frames, img, target_color_t).tolist()[0]

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
