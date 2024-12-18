import math, time, glob, tempfile
from typing import Tuple

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import GlobalCounters, tqdm, trange, getenv
import wandb
import cv2
import numpy as np
import albumentations as A

from .model import Model
from .common import get_annotation
from ..common.tensor import twohot
from ..common import BASE_PATH
from ..common.optim import CLAMB
from ..common.image import bgr_to_yuv420
from ..common.dataloader import batch_load, BatchDesc

# main augments
A_PIPELINE = A.Compose([
  A.HorizontalFlip(p=0.5),
  A.Perspective(p=0.25),
  A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
  A.OneOf([
    A.RandomCrop(256, 512, p=0.4),
    A.Compose([
      A.LongestMaxSize(max_size=512, p=1),
      A.RandomCrop(256, 512, p=1)
    ], p=0.6),
  ], p=1),
  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
  A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.5),
  A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
  A.OneOf([
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.RandomToneCurve(p=0.5),
  ], p=0.2),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

# for data with distance we can only do horizontal flips and very small non-affine transforms
D_PIPELINE = A.Compose([
  A.HorizontalFlip(p=0.5),
  A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.01, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
  A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.5),
  A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
  A.OneOf([
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.RandomToneCurve(p=0.5),
  ], p=0.2),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

def get_train_files():
  return glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
def load_single_file(file):
  img = cv2.imread(file)
  img = cv2.resize(img, (512, 256))

  detected, x, y, dist = get_annotation(file)

  # transform points
  x, y = x * img.shape[1], (1 - y) * img.shape[0]

  # augment
  if dist > 0:
    transformed = D_PIPELINE(image=img, keypoints=[(x, y)])
  else:
    transformed = A_PIPELINE(image=img, keypoints=[(x, y)])
  img, x, y = transformed["image"], transformed["keypoints"][0][0], transformed["keypoints"][0][1]

  # convert to yuv420
  img = bgr_to_yuv420(img)

  return {
    "x": img.tobytes(),
    "y": np.array((detected, x, y, dist), dtype=np.float32).tobytes(),
  }

BS = 64
WARMUP_STEPS = 500
WARMPUP_LR = 1e-5
START_LR = 1e-3
END_LR = 1e-5
EPOCHS = 100
STEPS_PER_EPOCH = len(get_train_files())//BS

def focal_loss(pred:Tensor, y:Tensor, alpha:float=0.25, gamma:float=2):
  p, ce = pred.sigmoid(), pred.binary_crossentropy_logits(y, reduction="none")
  pt = p * y + (1 - p) * (1 - y)
  alpha_ = y * alpha + (1 - y) * (1 - alpha)
  loss = ce * ((1 - pt) ** gamma) * alpha_
  return loss.mean()

def loss_fn(pred: Tuple[Tensor, Tensor, Tensor, Tensor], y: Tensor):
  cl_loss = focal_loss(pred[0], y[:, 0].cast(dtypes.int32).one_hot(2))
  x_loss = (pred[1].cross_entropy(twohot(y[:, 1], 512), reduction="none") * y[:, 0].detach()).mean()
  y_loss = (pred[2].cross_entropy(twohot(y[:, 2], 256), reduction="none") * y[:, 0].detach()).mean()
  dist_loss = (y[:, 3] > 0).detach().where(pred[3].cross_entropy(twohot(y[:, 3] * 10, 200), reduction="none") * y[:, 0].detach(), 0.0).mean()

  return cl_loss + x_loss + y_loss + dist_loss

@TinyJit
def train_step(x, y, lr):
  pred = model(x)
  loss = loss_fn(pred, y)

  optim.lr.assign(lr)
  optim.zero_grad()
  loss.backward()
  optim.step()

  return loss.float().realize()

warming_up = True
def get_lr(step:int) -> float:
  global warming_up
  if warming_up:
    lr = START_LR * (step / WARMUP_STEPS) + WARMPUP_LR * (1 - step / WARMUP_STEPS)
    if step >= WARMUP_STEPS: warming_up = False
  else: lr = END_LR + 0.5 * (START_LR - END_LR) * (1 + math.cos(((step - WARMUP_STEPS) / ((EPOCHS * STEPS_PER_EPOCH) - WARMUP_STEPS)) * math.pi))
  return lr

if __name__ == "__main__":
  Tensor.no_grad = False
  Tensor.training = True
  # dtypes.default_float = dtypes.float16

  if getenv("WANDB", 0):
    wandb.init(project="mrm_cv_autoaim")
    wandb.config.update({
      "warmup_steps": WARMUP_STEPS,
      "warmup_lr": WARMPUP_LR,
      "start_lr": START_LR,
      "end_lr": END_LR,
      "epochs": EPOCHS,
      "bs": BS,
      "steps_per_epoch": STEPS_PER_EPOCH,
    })

  model = Model()

  if (ckpt := getenv("CKPT", "")) != "":
    print(f"loading checkpoint {BASE_PATH / 'intermediate' / f'model_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"model_{ckpt}.safetensors")
    load_state_dict(model, state_dict, strict=False)

  parameters = get_parameters(model)
  optim = CLAMB(parameters, weight_decay=0.1, b1=0.9, b2=0.994, adam=True)

  def single_batch(iter):
    d, c = next(iter)
    return d["x"].to(Device.DEFAULT), d["y"].to(Device.DEFAULT), c

  steps = 0
  for epoch in trange(EPOCHS):
    batch_iter = iter(tqdm(batch_load(
      {
        "x": BatchDesc(shape=(128, 256, 6), dtype=dtypes.uint8),
        "y": BatchDesc(shape=(4,), dtype=dtypes.float32),
      },
      load_single_file, get_train_files, bs=BS, shuffle=True,
    ), total=STEPS_PER_EPOCH, desc=f"epoch {epoch}"))
    i, proc = 0, single_batch(batch_iter)
    while proc is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      lr = get_lr(steps)
      loss = train_step(proc[0], proc[1], Tensor([lr], dtype=dtypes.default_float))
      pt = time.perf_counter()

      try: next_proc = single_batch(batch_iter)
      except StopIteration: next_proc = None
      dt = time.perf_counter()

      loss = loss.item()
      at = time.perf_counter()

      tqdm.write(
        f"{i:5} {((at - st)) * 1000.0:7.2f} ms step, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
        f"{loss:11.6f} loss, {lr:.6f} lr, "
        f"{GlobalCounters.mem_used / 1e9:7.2f} GB used, {GlobalCounters.mem_used * 1e-9 / (at - st):9.2f} GB/s, {GlobalCounters.global_ops * 1e-9 / (at - st):9.2f} GFLOPS"
      )

      if getenv("WANDB", 0):
        wandb.log({
          "epoch": epoch + (i + 1) / STEPS_PER_EPOCH,
          "step_time": at - st, "python_time": pt - st, "data_time": dt - pt, "accel_time": at - dt,
          "loss": loss, "lr": lr,
          "gb": GlobalCounters.mem_used / 1e9, "gbps": GlobalCounters.mem_used * 1e-9 / (at - st), "gflops": GlobalCounters.global_ops * 1e-9 / (at - st)
        })

      proc, next_proc = next_proc, None
      i += 1
      steps += 1

    safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{epoch}.safetensors"))

  # copy the last intermediate to the final model
  with open(BASE_PATH / "intermediate" / f"model_{epoch}.safetensors", "rb") as f:
    with open(BASE_PATH / "model.safetensors", "wb") as f2: f2.write(f.read())
