import time, glob, math, os

from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
import wandb

from ..system.core.logging import logger
from ..common.dataloader import BatchDesc, Dataloader
from ..common.tensor import twohot, masked_cross_entropy, mal_loss
from ..common.optim import CLAMB, CosineWarmupLR
from ..common.image import rgb_to_yuv420_tensor
from .common import BASE_PATH
from .model import Model
from .data import get_train_files

BS = 256
WARMUP_STEPS = 400
WARMPUP_LR = 1e-7
START_LR = 1e-3
END_LR = 1e-4
EPOCHS = 20
STEPS_PER_EPOCH = len(get_train_files())//BS

def loss_fn(pred: tuple[Tensor, ...], y: Tensor):
  B = y.shape[0]

  y_color = y[:, 0].cast(dtypes.int32)
  y_xc = y[:, 1]
  y_yc = y[:, 2]
  y_xtl = y[:, 3]
  y_ytl = y[:, 4]
  y_xtr = y[:, 5]
  y_ytr = y[:, 6]
  y_xbl = y[:, 7]
  y_ybl = y[:, 8]
  y_xbr = y[:, 9]
  y_ybr = y[:, 10]
  y_number = y[:, 11].cast(dtypes.int32)

  # center keypoint loss
  xc_loss = pred[1].cross_entropy(twohot((y_xc + 256) / 2, 512))
  yc_loss = pred[2].cross_entropy(twohot((y_yc + 128) / 2, 256))

  # box keypoint loss
  xtl_loss = pred[3].cross_entropy(twohot((y_xtl + 256) / 2, 512))
  ytl_loss = pred[4].cross_entropy(twohot((y_ytl + 128) / 2, 256))
  xtr_loss = pred[5].cross_entropy(twohot((y_xtr + 256) / 2, 512))
  ytr_loss = pred[6].cross_entropy(twohot((y_ytr + 128) / 2, 256))
  xbl_loss = pred[7].cross_entropy(twohot((y_xbl + 256) / 2, 512))
  ybl_loss = pred[8].cross_entropy(twohot((y_ybl + 128) / 2, 256))
  xbr_loss = pred[9].cross_entropy(twohot((y_xbr + 256) / 2, 512))
  ybr_loss = pred[10].cross_entropy(twohot((y_ybr + 128) / 2, 256))

  keypoint_loss = xc_loss + yc_loss + xtl_loss + ytl_loss + xtr_loss + ytr_loss + xbl_loss + ybl_loss + xbr_loss + ybr_loss
  keypoint_loss = keypoint_loss / 10

  # quality factor from center keypoint
  if not hasattr(loss_fn, "x_arange"): setattr(loss_fn, "x_arange", Tensor.arange(512))
  if not hasattr(loss_fn, "y_arange"): setattr(loss_fn, "y_arange", Tensor.arange(256))
  point_xc = (pred[1].softmax() @ getattr(loss_fn, "x_arange")).float().mul(2).sub(256) / 512
  point_yc = (pred[2].softmax() @ getattr(loss_fn, "y_arange")).float().mul(2).sub(128) / 256
  point_dist = (point_xc.sub(y_xc / 512).square() + point_yc.sub(y_yc / 256).square()).sqrt()
  quality = (1 - point_dist.clamp(0, 1))

  # color loss
  target_cls = y_color.one_hot(4)
  target_quality = target_cls[:, :1].cat(quality.unsqueeze(-1).expand(B, 3), dim=1)
  color_loss = mal_loss(pred[0], target_cls, target_quality, gamma=1.5)

  # number loss
  target_cls = y_number.one_hot(6)
  target_quality = target_cls[:, :1].cat(quality.unsqueeze(-1).expand(B, 5), dim=1)
  number_loss = mal_loss(pred[11], target_cls, target_quality, gamma=1.5)

  return color_loss + keypoint_loss + number_loss

@TinyJit
def train_step(model, optim, lr_sched, x, y):
  optim.zero_grad()

  yuv = rgb_to_yuv420_tensor(x)

  pred = model(yuv)
  loss = loss_fn(pred, y)

  loss.backward()

  optim.step()
  lr_sched.step()

  return loss.float()

warming_up = True
def get_lr(step:int) -> float:
  global warming_up
  if warming_up:
    lr = START_LR * (step / WARMUP_STEPS) + WARMPUP_LR * (1 - step / WARMUP_STEPS)
    if step >= WARMUP_STEPS: warming_up = False
  else: lr = END_LR + 0.5 * (START_LR - END_LR) * (1 + math.cos(((step - WARMUP_STEPS) / ((EPOCHS * STEPS_PER_EPOCH) - WARMUP_STEPS)) * math.pi))
  return lr

def run():
  Tensor.no_grad = False
  Tensor.training = True

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

  dataloader = Dataloader({
    "x": BatchDesc(shape=(256, 512, 3), dtype=dtypes.uint8),
    "y": BatchDesc(shape=(12,), dtype=dtypes.default_float),
  }, bs=BS, files_fn=get_train_files)

  model = Model()

  if (ckpt := getenv("CKPT", "")) != "":
    print(f"loading checkpoint {BASE_PATH / 'intermediate' / f'model_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"model_{ckpt}.safetensors")
    load_state_dict(model, state_dict, strict=False)

  parameters = get_parameters(model)
  optim = CLAMB(parameters, weight_decay=0.01, b1=0.9, b2=0.994, adam=True)
  lr_sched = CosineWarmupLR(optim, WARMUP_STEPS, WARMPUP_LR, START_LR, END_LR, EPOCHS, STEPS_PER_EPOCH)

  steps = 0
  for epoch in range(EPOCHS):
    dataloader.load()
    i, d = 0, dataloader.next(Device.DEFAULT)
    while d is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      loss = train_step(model, optim, lr_sched, *d[:-1])
      pt = time.perf_counter()

      try: next_d = dataloader.next(Device.DEFAULT)
      except StopIteration: next_d = None
      dt = time.perf_counter()

      lr = optim.lr.item()
      loss = loss.item()
      at = time.perf_counter()

      # logging
      logger.info(
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

      d, next_d = next_d, None
      i += 1
      steps += 1

    safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{epoch}.safetensors"))
    safe_save(get_state_dict(optim), str(BASE_PATH / f"intermediate/optim_{epoch}.safetensors"))

  # copy the last intermediate to the final model
  with open(BASE_PATH / "intermediate" / f"model_{epoch}.safetensors", "rb") as f:
    with open(BASE_PATH / "model.safetensors", "wb") as f2: f2.write(f.read())

  wandb.finish()
