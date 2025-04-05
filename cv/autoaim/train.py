import math, time
from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import GlobalCounters, tqdm, trange, getenv
import wandb
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from .model import Model
from .data import get_train_files, load_single_file, single_batch
from ..common.tensor import twohot, masked_cross_entropy, mal_loss
from ..common import BASE_PATH
from ..common.optim import CLAMB
from ..common.image import rgb_to_yuv420_tensor
from ..common.dataloader import batch_load, BatchDesc

BS = 256
WARMUP_STEPS = 400
WARMPUP_LR = 1e-7
START_LR = 1e-3
END_LR = 1e-4
EPOCHS = 20
STEPS_PER_EPOCH = len(get_train_files())//BS

def loss_fn(pred, y: Tensor):
  B = y.shape[0]

  y_color = y[:, 0]
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
  y_number = y[:, 11]

  det_gate = (y_color > 0).detach()

  # center keypoint loss
  xc_loss = masked_cross_entropy(pred[1], twohot((y_xc + 256) / 2, 512), det_gate)
  yc_loss = masked_cross_entropy(pred[2], twohot((y_yc + 128) / 2, 256), det_gate)

  # box keypoint loss
  xtl_loss = masked_cross_entropy(pred[3], twohot((y_xtl + 256) / 2, 512), det_gate)
  ytl_loss = masked_cross_entropy(pred[4], twohot((y_ytl + 128) / 2, 256), det_gate)
  xtr_loss = masked_cross_entropy(pred[5], twohot((y_xtr + 256) / 2, 512), det_gate)
  ytr_loss = masked_cross_entropy(pred[6], twohot((y_ytr + 128) / 2, 256), det_gate)
  xbl_loss = masked_cross_entropy(pred[7], twohot((y_xbl + 256) / 2, 512), det_gate)
  ybl_loss = masked_cross_entropy(pred[8], twohot((y_ybl + 128) / 2, 256), det_gate)
  xbr_loss = masked_cross_entropy(pred[9], twohot((y_xbr + 256) / 2, 512), det_gate)
  ybr_loss = masked_cross_entropy(pred[10], twohot((y_ybr + 126) / 2, 256), det_gate)

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
  target_cls = y_color.cast(dtypes.int32).one_hot(4)
  target_quality = target_cls[:, :1].cat(quality.unsqueeze(-1).expand(B, 3), dim=1)
  color_loss = mal_loss(pred[0], target_cls, target_quality, gamma=1.5)

  # number loss
  target_cls = y_number.cast(dtypes.int32).one_hot(6)
  target_quality = target_cls[:, :1].cat(quality.unsqueeze(-1).expand(B, 5), dim=1)
  number_loss = mal_loss(pred[11], target_cls, target_quality, gamma=1.5)

  return color_loss + keypoint_loss + number_loss

@TinyJit
def train_step(x, x2, y, lr):
  optim.zero_grad()

  yuv = rgb_to_yuv420_tensor(x)
  yuv2 = rgb_to_yuv420_tensor(x2)

  pred = model((yuv, yuv2))
  loss = loss_fn(pred, y)

  # (loss * 1024).backward()
  # for p in optim.params:
  #   p.grad = p.grad / 1024
  loss.backward()

  optim.lr.assign(lr.to(optim.lr.device))
  optim.step()

  if getenv("NAN", 0):
    gsums = {k: v.grad.square().sum() for k,v in get_state_dict(model).items() if v.grad is not None}
    wsums = {k: v.square().sum() for k,v in get_state_dict(model).items() if v.grad is not None}
    loss = loss.float().realize()
    for s in gsums.values(): s.realize()
    for s in wsums.values(): s.realize()
    return loss, gsums, wsums

  return loss.float().to("CPU")

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
  dtypes.default_float = dtypes.float32

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

  if getenv("PRETRAINED_BACKBONE"):
    print(f"loading pretrained backbone")
    state_dict = safe_load(Path(__file__).parent.parent.parent / "weights/model.safetensors")
    # remove all keys that don't start with backbone and strip the backbone. prefix
    state_dict = {k[9:]: v for k,v in state_dict.items() if k.startswith("backbone.")}
    load_state_dict(model.backbone, state_dict, strict=False)

  if getenv("PRETRAINED_DECODER"):
    print(f"loading pretrained decoder")
    state_dict = safe_load(Path(__file__).parent.parent.parent / "weights/model.safetensors")
    # remove all keys that don't start with decoder and strip the decoder. prefix
    state_dict = {k[8:]: v for k,v in state_dict.items() if k.startswith("decoder.")}
    load_state_dict(model.decoder, state_dict, strict=False)

  if (ckpt := getenv("CKPT", "")) != "":
    print(f"loading checkpoint {BASE_PATH / 'intermediate' / f'model_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"model_{ckpt}.safetensors")
    load_state_dict(model, state_dict, strict=False)

  parameters = get_parameters(model)
  optim = CLAMB(parameters, weight_decay=0.1, b1=0.9, b2=0.994, adam=True)

  steps = 0
  for epoch in trange(EPOCHS):
    batch_iter = iter(tqdm(batch_load(
      {
        "x": BatchDesc(shape=(256, 512, 3), dtype=dtypes.uint8),
        "x2": BatchDesc(shape=(256, 512, 3), dtype=dtypes.uint8),
        "y": BatchDesc(shape=(12,), dtype=dtypes.default_float),
      },
      load_single_file, get_train_files, bs=BS, shuffle=True,
    ), total=STEPS_PER_EPOCH, desc=f"epoch {epoch}"))
    i, proc = 0, single_batch(batch_iter)
    while proc is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      lr = get_lr(steps)
      loss = train_step(proc[0], proc[1], proc[2], Tensor([lr], dtype=dtypes.float32, device="PYTHON"))
      if getenv("NAN", 0):
        loss, gsums, wsums = loss
      pt = time.perf_counter()

      try: next_proc = single_batch(batch_iter)
      except StopIteration: next_proc = None
      dt = time.perf_counter()

      loss = loss.item()
      at = time.perf_counter()

      # check for NaNs
      if getenv("NAN", 0):
        has_nan = False
        for p,s in gsums.items():
          if math.isnan(s.item()):
            print(f"{p}.grad: NaN")
            has_nan = True
          else:
            print(f"{p}.grad: {math.sqrt(s.item())}")
        for p,s in wsums.items():
          if math.isnan(s.item()):
            print(f"{p}: NaN")
            has_nan = True
          else:
            print(f"{p}: {math.sqrt(s.item())}")
        if has_nan: exit(1)

      # logging
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
    safe_save(get_state_dict(optim), str(BASE_PATH / f"intermediate/optim_{epoch}.safetensors"))

  # copy the last intermediate to the final model
  with open(BASE_PATH / "intermediate" / f"model_{epoch}.safetensors", "rb") as f:
    with open(BASE_PATH / "model.safetensors", "wb") as f2: f2.write(f.read())
