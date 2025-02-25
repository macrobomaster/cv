import math, time, glob
from typing import Tuple

from tinygrad.tensor import Tensor, _to_np_dtype
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
from .lpips import VGG16Loss
from ..common import BASE_PATH
from ..common.optim import CLAMB
from ..common.dataloader import batch_load, BatchDesc

WIDTH = 128
HEIGHT = 128

def get_train_files():
  return glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)

A_PIPELINE = None
def load_single_file(file):
  global A_PIPELINE
  if A_PIPELINE is None:
    A_PIPELINE = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.Perspective(p=0.25),
      A.Affine(translate_percent=(-0.2, 0.2), scale=(0.9, 1.1), rotate=(-45, 45), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
      A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.5),
      A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
      A.OneOf([
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomToneCurve(p=0.5),
      ], p=0.2),
    ])

  img = cv2.imread(file)
  if img.shape[0] != HEIGHT or img.shape[1] != WIDTH:
    img = cv2.resize(img, (WIDTH, HEIGHT))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # augment
  transformed = A_PIPELINE(image=img)
  img = transformed["image"]

  return {
    "x": img.tobytes(),
  }

BS = 32
WARMUP_STEPS = 100
WARMPUP_LR = 1e-5
START_LR = 1e-3
END_LR = 1e-5
EPOCHS = 1
STEPS_PER_EPOCH = len(get_train_files())//BS

def loss_fn(x_hat: Tensor, x: Tensor):
  l1_loss = (x_hat - x).abs().mean()
  l2_loss = (x_hat - x).square().mean()
  wp_loss = 0.01 * (x_hat[0] - x).flatten(1).square().max(axis=-1).mean()
  # lpips_loss = lpips(x_hat, x).mean()

  return l1_loss + l2_loss + wp_loss

@TinyJit
def train_step(x, lr):
  x_hat, x, (embed_loss, commit_loss, ortho_loss) = model(x)
  recon_loss = loss_fn(x_hat, x)
  quant_loss = 10 * (embed_loss + 0.25 * commit_loss + ortho_loss)
  loss = recon_loss + quant_loss

  optim.lr.assign(lr)
  optim.zero_grad()
  loss.backward()
  optim.step()

  return loss.float().realize(), recon_loss.float().realize(), quant_loss.float().realize()

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
    wandb.init(project="mrm_cv_vae")
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
    print(f"loading checkpoint {BASE_PATH / 'intermediate' / f'vae_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"vae_{ckpt}.safetensors")
    load_state_dict(model, state_dict, strict=False)

  parameters = get_parameters(model)
  optim = CLAMB(parameters, weight_decay=0.1, b1=0.9, b2=0.994, adam=True)

  # initialize lpips loss
  lpips = VGG16Loss()
  load_state_dict(lpips, safe_load("weights/lpips.safetensors"))

  def single_batch(iter):
    d, c = next(iter)
    return d["x"].to(Device.DEFAULT), c

  steps = 0
  for epoch in trange(EPOCHS):
    batch_iter = iter(tqdm(batch_load(
      {
        "x": BatchDesc(shape=(HEIGHT, WIDTH, 3), dtype=dtypes.uint8),
      },
      load_single_file, get_train_files, bs=BS, shuffle=True,
    ), total=STEPS_PER_EPOCH, desc=f"epoch {epoch}"))
    i, proc = 0, single_batch(batch_iter)
    while proc is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      lr = get_lr(steps)
      loss, recon_loss, quant_loss = train_step(proc[0], Tensor([lr], dtype=dtypes.float32))
      pt = time.perf_counter()

      try: next_proc = single_batch(batch_iter)
      except StopIteration: next_proc = None
      dt = time.perf_counter()

      loss, recon_loss, quant_loss = loss.item(), recon_loss.item(), quant_loss.item()
      at = time.perf_counter()

      tqdm.write(
        f"{i:5} {((at - st)) * 1000.0:7.2f} ms step, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
        f"{loss:11.6f} loss, {recon_loss:11.6f} recon_loss, {quant_loss:11.6f} quant_loss, "
        f"{lr:.6f} lr, "
        f"{GlobalCounters.mem_used / 1e9:7.2f} GB used, {GlobalCounters.mem_used * 1e-9 / (at - st):9.2f} GB/s, {GlobalCounters.global_ops * 1e-9 / (at - st):9.2f} GFLOPS"
      )

      if getenv("WANDB", 0):
        wandb.log({
          "epoch": epoch + (i + 1) / STEPS_PER_EPOCH,
          "step_time": at - st, "python_time": pt - st, "data_time": dt - pt, "accel_time": at - dt,
          "loss": loss, "recon_loss": recon_loss, "quant_loss": quant_loss,
          "lr": lr,
          "gb": GlobalCounters.mem_used / 1e9, "gbps": GlobalCounters.mem_used * 1e-9 / (at - st), "gflops": GlobalCounters.global_ops * 1e-9 / (at - st)
        })

      proc, next_proc = next_proc, None
      i += 1
      steps += 1

    safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/vae_{epoch}.safetensors"))

  # copy the last intermediate to the final model
  with open(BASE_PATH / "intermediate" / f"vae_{epoch}.safetensors", "rb") as f:
    with open(BASE_PATH / "vae.safetensors", "wb") as f2: f2.write(f.read())
