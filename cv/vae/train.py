import time

from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
import wandb

from ..system.core.logging import logger
from ..common.dataloader import BatchDesc, Dataloader
from ..common.optim import CLaProp, CosineWarmupLR, grad_clip_norm, SwitchEMA
from ..common.image import rgb_to_yuv420_tensor
from ..common import BASE_PATH
from ..autoaim.model import Backbone
from .model import Model
from .data import get_train_files
from .lpips import VGG16Loss

BS = 32
WARMUP_STEPS = 400
WARMPUP_LR = 1e-7
START_LR = 1e-3
END_LR = 1e-4
EPOCHS = 20
STEPS_PER_EPOCH = len(get_train_files())//BS

def loss_fn(x:Tensor, x_hat:Tensor, lpips) -> Tensor:
  l1_loss = (x - x_hat).abs().mean()
  l2_loss = (x - x_hat).square().mean()
  wp_loss = 0.01 * (x - x_hat).flatten(1).square().max(axis=-1).mean()
  lpips_loss = lpips(x_hat, x).mean()

  return l1_loss + l2_loss + wp_loss + lpips_loss

@TinyJit
def train_step(encoder, sideband, model, optim, lr_sched, switch_ema, lpips, x):
  optim.zero_grad()

  yuv = rgb_to_yuv420_tensor(x)
  yuv = yuv.cast(dtypes.default_float).permute(0, 3, 1, 2).div(255)
  yuv_mean, yuv_std = yuv.mean([2, 3], keepdim=True), yuv.std([2, 3], keepdim=True)
  yuv = yuv.sub(yuv_mean).div(yuv_std.add(1e-6))
  x_hat = x.cast(dtypes.float32).permute(0, 3, 1, 2).avg_pool2d(8, 8).div(255)

  with Tensor.train(False), Tensor.test(True):
    z = encoder(yuv, sideband.expand(yuv.shape[0], -1))[-1].detach()
  x = model(z)
  loss = loss_fn(x, x_hat, lpips)

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

  dataloader = Dataloader({
    "x": BatchDesc(shape=(256, 512, 3), dtype=dtypes.uint8),
  }, bs=BS, files_fn=get_train_files)

  encoder = Backbone(cin=6, cstage=[16, 32, 64, 128], stages=[2, 2, 6, 2], sideband=4, sideband_only=True, dropout=0)
  state_dict = safe_load(BASE_PATH / "model.safetensors")
  state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
  load_state_dict(encoder, state_dict)
  sideband = safe_load(BASE_PATH / "model.safetensors")["sideband"].to(Device.DEFAULT)

  model = Model()
  model_ema = Model()

  if (ckpt := getenv("CKPT", "")) != "":
    logger.info(f"loading checkpoint {BASE_PATH / 'intermediate' / f'vae_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"vae_{ckpt}.safetensors")
    load_state_dict(model, state_dict, strict=False)

  parameters = get_parameters(model)
  optim = CLaProp(parameters, weight_decay=0.1)
  lr_sched = CosineWarmupLR(optim, WARMUP_STEPS, WARMPUP_LR, START_LR, END_LR, EPOCHS, STEPS_PER_EPOCH)

  switch_ema = SwitchEMA(model, model_ema, momentum=0.999)

  lpips = VGG16Loss()
  lpips.load_from_pretrained()

  steps = 0
  for epoch in range(EPOCHS):
    dataloader.load()
    i, d = 0, dataloader.next(Device.DEFAULT)
    while d is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      loss, global_norm = train_step(encoder, sideband, model, optim, lr_sched, switch_ema, lpips, *d[:-1])
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
    safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/vae_{epoch}.safetensors"))
    safe_save(get_state_dict(optim), str(BASE_PATH / f"intermediate/vae_optim_{epoch}.safetensors"))

  # copy the last intermediate to the final model
  with open(BASE_PATH / "intermediate" / f"vae_{epoch}.safetensors", "rb") as f:
    with open(BASE_PATH / "vae.safetensors", "wb") as f2: f2.write(f.read())

  wandb.finish()
