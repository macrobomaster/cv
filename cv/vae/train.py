import time

from tinygrad.device import Device
from tinygrad.nn.optim import SGD
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
import wandb

from ..system.core.logging import logger
from ..common.dataloader import BatchDesc, Dataloader
from ..common.optim import CLaProp, CosineWarmupLR, StepSchedule, grad_clip_norm, SwitchEMA
from ..common.tensor import hinge_discriminator_loss, norm
from ..common import BASE_PATH
from ..autoaim.model import Backbone
from .model import Model
from .data import get_train_files
from .lpips import VGG16Loss
from .discriminator import Discriminator

BS = 256
WARMUP_STEPS = 100
WARMPUP_LR = 1e-7
START_LR = 4e-3
END_LR = 1e-5
EPOCHS = 100
STEPS_PER_EPOCH = len(get_train_files())//BS

def loss_fn(model, x:Tensor, x_hat:Tensor, lpips, disc, gram_schedule, disc_schedule) -> Tensor:
  l1_loss = 0.1 * (x - x_hat).abs().mean()
  l2_loss = (x - x_hat).square().mean()
  wp_loss = 0.01 * (x - x_hat).flatten(1).square().max(axis=-1).mean()

  lpips_loss, gram_loss = lpips(x, x_hat)
  lpips_loss = 0.1 * lpips_loss.mean()
  gram_loss = gram_schedule.get() * gram_loss.mean()
  gram_schedule.step()

  rec_loss = l1_loss + l2_loss + wp_loss + lpips_loss + gram_loss
  rec_loss = rec_loss.squeeze()

  with Tensor.train(False):
    logits_fake = disc(x)
  d_loss = 2 * logits_fake.mean().neg()
  r_grads = rec_loss.gradient(model.out.weight)[0]
  d_grads = d_loss.gradient(model.out.weight)[0]
  d_weight = norm(r_grads).div(norm(d_grads).add(1e-4)).clamp(0, 1e4).detach()
  d_loss = disc_schedule.get() * d_weight * d_loss

  return rec_loss + d_loss

@TinyJit
def train_step(encoder, model, optim, lr_sched, switch_ema, lpips, disc, disc_optim, disc_lr_sched, gram_schedule, disc_schedule, x:Tensor):
  x_hat = x.cast(dtypes.float32).permute(0, 3, 1, 2).interpolate((32, 64), mode="linear").div(255)

  with Tensor.train(False), Tensor.test(True):
    z = encoder(x)[-1].detach()
  x = model(z)

  rec_loss = loss_fn(model, x, x_hat, lpips, disc, gram_schedule, disc_schedule)

  optim.zero_grad()
  rec_loss.squeeze().backward()
  global_norm = grad_clip_norm(optim)
  optim.step()
  lr_sched.step()
  switch_ema.update()

  logits_real = disc(x_hat.requires_grad_())
  logits_fake = disc(x.detach())
  disc_loss = hinge_discriminator_loss(logits_real, logits_fake)
  r1_reg = 10 * logits_real.sum().gradient(x_hat)[0].square().sum((1, 2, 3)).mean()
  disc_loss = disc_loss + r1_reg
  disc_loss = disc_schedule.get() * disc_loss
  disc_schedule.step()

  disc_optim.zero_grad()
  disc_loss.squeeze().backward()
  disc_global_norm = grad_clip_norm(disc_optim)
  disc_optim.step()
  disc_lr_sched.step()

  return rec_loss.float(), disc_loss.float(), (global_norm + disc_global_norm).float()

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

  encoder = Backbone(cin=3, cstage=[16, 32, 64, 128], stages=[2, 2, 6, 2], sideband=4, sideband_only=True, dropout=0)
  state_dict = safe_load(BASE_PATH / "model.safetensors")
  state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
  load_state_dict(encoder, state_dict)

  model = Model()
  model_ema = Model()

  if (ckpt := getenv("CKPT", "")) != "":
    logger.info(f"loading checkpoint {BASE_PATH / 'intermediate' / f'vae_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"vae_{ckpt}.safetensors")
    load_state_dict(model, state_dict, strict=False)

  optim = CLaProp(get_parameters(model), weight_decay=0.01)
  lr_sched = CosineWarmupLR(optim, WARMUP_STEPS, WARMPUP_LR, START_LR, END_LR, EPOCHS, STEPS_PER_EPOCH)

  switch_ema = SwitchEMA(model, model_ema, momentum=0.999)

  lpips = VGG16Loss()
  lpips.load_from_pretrained()

  gram_schedule = StepSchedule(0, 0.05, (EPOCHS // 2) * STEPS_PER_EPOCH)
  disc_schedule = StepSchedule(0, 0.1, (EPOCHS // 2) * STEPS_PER_EPOCH)

  disc = Discriminator()
  disc_optim = CLaProp(get_parameters(disc), b1=0.5, b2=0.9, weight_decay=0.01)
  disc_lr_sched = CosineWarmupLR(optim, WARMUP_STEPS, WARMPUP_LR, START_LR / 10, END_LR / 10, EPOCHS, STEPS_PER_EPOCH)

  steps = 0
  for epoch in range(EPOCHS):
    dataloader.load()
    i, d = 0, dataloader.next(Device.DEFAULT)
    while d is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      rec_loss, disc_loss, global_norm = train_step(encoder, model, optim, lr_sched, switch_ema, lpips, disc, disc_optim, disc_lr_sched, gram_schedule, disc_schedule, *d[:-1])
      pt = time.perf_counter()

      try: next_d = dataloader.next(Device.DEFAULT)
      except StopIteration: next_d = None
      dt = time.perf_counter()

      lr = optim.lr.item()
      rec_loss, disc_loss, global_norm = rec_loss.item(), disc_loss.item(), global_norm.item()
      at = time.perf_counter()

      # logging
      logger.info(
        f"{i:5} {((at - st)) * 1000.0:7.2f} ms step, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
        f"{rec_loss:11.6f} rec_loss, {disc_loss:11.6f} disc_loss, {global_norm:11.6f} global_norm, {lr:.6f} lr, "
        f"{GlobalCounters.mem_used / 1e9:7.2f} GB used, {GlobalCounters.mem_used * 1e-9 / (at - st):9.2f} GB/s, {GlobalCounters.global_ops * 1e-9 / (at - st):9.2f} GFLOPS"
      )

      if getenv("WANDB", 0):
        wandb.log({
          "epoch": epoch + (i + 1) / STEPS_PER_EPOCH,
          "step_time": at - st, "python_time": pt - st, "data_time": dt - pt, "accel_time": at - dt,
          "rec_loss": rec_loss, "disc_loss": disc_loss, "global_norm": global_norm, "lr": lr,
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
