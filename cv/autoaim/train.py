import time

from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.device import Device
import wandb

from ..system.core.logging import logger
from ..system.core.keyvalue import kv_put
from ..common.dataloader import BatchDesc, Dataloader
from ..common.losses import twohot_loss, cross_entropy, mal_loss, gaussian_uncertainty, masked_mean
from ..common.optim import CLaProp, CosineWarmupLR, grad_clip_norm, SwitchEMA
from .common import BASE_PATH
from .model import Model
from .data import get_train_files

GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1)))
BS = 256 * len(GPUS)
WARMUP_STEPS = 400
WARMPUP_LR = 1e-7
START_LR = 2e-3 * len(GPUS)
END_LR = 1e-5
EPOCHS = 50
STEPS_PER_EPOCH = len(get_train_files())//BS

def loss_fn(model, pred:tuple[Tensor, ...], y:Tensor):
  y_det = y[:, 0].cast(dtypes.int32)
  y_color = y[:, 1].cast(dtypes.int32)
  y_number = y[:, 2].cast(dtypes.int32)
  y_plate = y[:, 3:13]

  has_det = y[:, 13] > 0
  has_color = y[:, 14] > 0
  has_number = y[:, 15] > 0
  has_plate = y[:, 16] > 0

  plate_area = y[:, 17]

  area_weight = 1 / plate_area
  area_weight = area_weight / area_weight.mean()
  area_weight = (plate_area > 1).where(area_weight, 1)

  plate_loss = twohot_loss(pred[3], y_plate, model.heads.plate_head.bins, model.heads.plate_head.low, model.heads.plate_head.high)
  plate_loss = gaussian_uncertainty(plate_loss, pred[4])
  plate_loss = plate_loss.mean(-1) * area_weight
  plate_loss = masked_mean(plate_loss, has_plate)

  # quality factor from center keypoint
  if not hasattr(loss_fn, "plate_twohot_weights"):
    setattr(loss_fn, "plate_twohot_weights", Tensor.linspace(model.heads.plate_head.low, model.heads.plate_head.high, model.heads.plate_head.bins, device=y.device).reshape(1, 1, -1))
  point_c = pred[3][:, 0:2, :].softmax().mul(getattr(loss_fn, "plate_twohot_weights")).sum(-1)
  point_dist = point_c.square().sum(-1).sqrt()
  quality = (1 - point_dist.clamp(0, 1))

  # det loss
  target_cls = y_det.one_hot(2)
  target_quality = target_cls[:, :1].cat(quality.unsqueeze(-1).expand(y.shape[0], 1), dim=1)
  det_loss = mal_loss(pred[0], target_cls, target_quality, gamma=1.5)
  det_loss = det_loss * area_weight
  det_loss = masked_mean(det_loss, has_det)

  # color loss
  target_cls = y_color.one_hot(4)
  color_loss = cross_entropy(pred[1], target_cls)
  color_loss = color_loss * area_weight
  color_loss = masked_mean(color_loss, has_color)

  # number loss
  target_cls = y_number.one_hot(7)
  number_loss = cross_entropy(pred[2], target_cls)
  number_loss = number_loss * area_weight
  number_loss = masked_mean(number_loss, has_number)

  return det_loss + plate_loss + color_loss + number_loss

@TinyJit
def train_step(model, optim, lr_sched, switch_ema, x, y) -> Tensor:
  optim.zero_grad()

  pred = model(x)
  loss = loss_fn(model, pred, y)

  loss.backward()

  global_norm = grad_clip_norm(optim)

  optim.step()
  lr_sched.step()

  switch_ema.update()

  return Tensor.cat(loss.float().reshape(1), global_norm.float().reshape(1), optim.lr.float().reshape(1))

def run():
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
    "y": BatchDesc(shape=(18,), dtype=dtypes.float32),
  }, bs=BS, files_fn=get_train_files)

  model = Model()
  for _, x in get_state_dict(model).items(): x.to_(GPUS)
  model_ema = Model()
  for _, x in get_state_dict(model_ema).items(): x.to_(GPUS)

  parameters = get_parameters(model)
  optim = CLaProp(parameters, weight_decay=0.01)
  lr_sched = CosineWarmupLR(optim, WARMUP_STEPS, WARMPUP_LR, START_LR, END_LR, EPOCHS, STEPS_PER_EPOCH)

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

    if getenv("CKPT_OPTIM"):
      logger.info(f"loading optimizer {BASE_PATH / 'intermediate' / f'optim_{ckpt}.safetensors'}")
      state_dict = safe_load(BASE_PATH / "intermediate" / f"optim_{ckpt}.safetensors")
      load_state_dict(optim, state_dict, strict=False)

  switch_ema = SwitchEMA(model, model_ema, EPOCHS, STEPS_PER_EPOCH, momentum=0.999)

  steps = 0
  for epoch in range(EPOCHS):
    dataloader.load()
    i, d = 0, dataloader.next(GPUS)
    while d is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      out = train_step(model, optim, lr_sched, switch_ema, *d[:-1])
      pt = time.perf_counter()

      try: next_d = dataloader.next(GPUS)
      except StopIteration: next_d = None
      dt = time.perf_counter()

      loss, global_norm, lr = out.tolist()
      at = time.perf_counter()

      # logging
      logger.info(
        f"{epoch:3} {i:5}/{STEPS_PER_EPOCH} {((at - st)) * 1000.0:7.2f} ms step, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
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

    # save intermediate model
    safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{epoch}.safetensors"))
    safe_save(get_state_dict(optim), str(BASE_PATH / f"intermediate/optim_{epoch}.safetensors"))

  # copy the last intermediate to the final model
  with open(BASE_PATH / "intermediate" / f"model_{epoch}.safetensors", "rb") as f:
    with open(BASE_PATH / "model.safetensors", "wb") as f2: f2.write(f.read())

  wandb.finish()
  kv_put("global", "do_shutdown", True)
