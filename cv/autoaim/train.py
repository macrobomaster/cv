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
from ..common.tensor import twohot, masked_cross_entropy, mal_loss, masked_mdn_loss, masked_twohot_uncertainty_loss
from ..common.optim import CLAMB, CosineWarmupLR, Schedule, CosineSchedule, ExpSchedule, grad_clip_norm, SwitchEMA
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
  y_color = y[:, 0].cast(dtypes.int32)
  y_keypoints = y[:, 2:12]
  y_number = y[:, 1].cast(dtypes.int32)

  det_gate = y_color > 0

  keypoint_loss = masked_twohot_uncertainty_loss(pred[2], pred[3], y_keypoints, det_gate, 64, -2, 2)

  # quality factor from center keypoint
  # if not hasattr(loss_fn, "x_arange"): setattr(loss_fn, "x_arange", Tensor.arange(512))
  # if not hasattr(loss_fn, "y_arange"): setattr(loss_fn, "y_arange", Tensor.arange(256))
  # point_xc = (pred[1].softmax() @ getattr(loss_fn, "x_arange")).float().mul(2).sub(256) / 512
  # point_yc = (pred[2].softmax() @ getattr(loss_fn, "y_arange")).float().mul(2).sub(128) / 256
  # point_dist = (point_xc.sub(y_xc / 512).square() + point_yc.sub(y_yc / 256).square()).sqrt()
  # quality = (1 - point_dist.clamp(0, 1))

  # color loss
  # target_cls = y_color.one_hot(4)
  # target_quality = target_cls[:, :1].cat(quality.unsqueeze(-1).expand(y.shape[0], 3), dim=1)
  # color_loss = mal_loss(pred[0], target_cls, target_quality, gamma=1.5)
  color_loss = pred[0].sparse_categorical_crossentropy(y_color)


  # number loss
  # target_cls = y_number.one_hot(6)
  # target_quality = target_cls[:, :1].cat(quality.unsqueeze(-1).expand(y.shape[0], 5), dim=1)
  # number_loss = mal_loss(pred[11], target_cls, target_quality, gamma=1.5)
  number_loss = pred[1].sparse_categorical_crossentropy(y_number)

  return color_loss + keypoint_loss + number_loss

@TinyJit
def train_step(model, optim, lr_sched, switch_ema, temp_sched, x, y):
  optim.zero_grad()

  yuv = rgb_to_yuv420_tensor(x)

  pred = model(yuv)
  loss = loss_fn(pred, y, temp_sched)

  loss.backward()

  global_norm = grad_clip_norm(optim)

  optim.step()
  lr_sched.step()

  switch_ema.update()

  return loss.float(), global_norm.float()

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
  model_ema = Model()

  if (ckpt := getenv("CKPT", "")) != "":
    logger.info(f"loading checkpoint {BASE_PATH / 'intermediate' / f'model_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"model_{ckpt}.safetensors")

    if getenv("PRETRAINED_BACKBONE"):
      logger.info(f"only loading backbone")
      # remove all keys that don't start with backbone and strip the backbone. prefix
      state_dict = {k[9:]: v for k,v in state_dict.items() if k.startswith("backbone.")}
      load_state_dict(model.backbone, state_dict, strict=False)
    else:
      logger.info(f"loading whole model")
      load_state_dict(model, state_dict, strict=False)

  parameters = get_parameters(model)
  optim = CLAMB(parameters, weight_decay=0.1, b1=0.9, b2=0.994, adam=True)
  lr_sched = CosineWarmupLR(optim, WARMUP_STEPS, WARMPUP_LR, START_LR, END_LR, EPOCHS, STEPS_PER_EPOCH)

  switch_ema = SwitchEMA(model, model_ema, alpha=0.999)

  temp_sched = ExpSchedule(20, 1, int(STEPS_PER_EPOCH * EPOCHS * (1 / 4)))

  steps = 0
  for epoch in range(EPOCHS):
    dataloader.load()
    i, d = 0, dataloader.next(Device.DEFAULT)
    while d is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      loss, global_norm = train_step(model, optim, lr_sched, switch_ema, temp_sched, *d[:-1])
      pt = time.perf_counter()

      try: next_d = dataloader.next(Device.DEFAULT)
      except StopIteration: next_d = None
      dt = time.perf_counter()

      lr = optim.lr.item()
      loss, global_norm = loss.item(), global_norm.item()
      at = time.perf_counter()

      # logging
      logger.info(
        f"{i:5} {((at - st)) * 1000.0:7.2f} ms step, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
        f"{loss:11.6f} loss, {global_norm:11.6f} global_norm, {lr:.6f} lr, "
        f"{GlobalCounters.mem_used / 1e9:7.2f} GB used, {GlobalCounters.mem_used * 1e-9 / (at - st):9.2f} GB/s, {GlobalCounters.global_ops * 1e-9 / (at - st):9.2f} GFLOPS"
      )

      if getenv("WANDB", 0):
        wandb.log({
          "epoch": epoch + (i + 1) / STEPS_PER_EPOCH,
          "step_time": at - st, "python_time": pt - st, "data_time": dt - pt, "accel_time": at - dt,
          "loss": loss, "global_norm": global_norm, "lr": lr,
          "gb": GlobalCounters.mem_used / 1e9, "gbps": GlobalCounters.mem_used * 1e-9 / (at - st), "gflops": GlobalCounters.global_ops * 1e-9 / (at - st)
        })

      d, next_d = next_d, None
      i += 1
      steps += 1

    switch_ema.switch()

    # save intermediate model
    safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{epoch}.safetensors"))
    safe_save(get_state_dict(optim), str(BASE_PATH / f"intermediate/optim_{epoch}.safetensors"))

  # copy the last intermediate to the final model
  with open(BASE_PATH / "intermediate" / f"model_{epoch}.safetensors", "rb") as f:
    with open(BASE_PATH / "model.safetensors", "wb") as f2: f2.write(f.read())

  wandb.finish()
