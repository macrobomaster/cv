import math, time

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
WARMUP_STEPS = 200
WARMPUP_LR = 1e-7
START_LR = 1e-3
END_LR = 1e-5
EPOCHS = 30
STEPS_PER_EPOCH = len(get_train_files())//BS

def loss_fn(pred: tuple[Tensor, Tensor, Tensor, Tensor], y: Tensor):
  det_gate = (y[:, 0] > 0).detach()

  x_loss = masked_cross_entropy(pred[1], twohot(y[:, 1], 512), det_gate)
  y_loss = masked_cross_entropy(pred[2], twohot(y[:, 2], 256), det_gate)

  # dist_loss = masked_cross_entropy(pred[3], twohot(y[:, 3] * 4, 64), (det_gate & (y[:, 3] > 0)).detach())

  if not hasattr(loss_fn, "x_arange"): setattr(loss_fn, "x_arange", Tensor.arange(512))
  if not hasattr(loss_fn, "y_arange"): setattr(loss_fn, "y_arange", Tensor.arange(256))
  point_x = (pred[1].softmax() @ getattr(loss_fn, "x_arange")).float() / 512
  point_y = (pred[2].softmax() @ getattr(loss_fn, "y_arange")).float() / 256
  point_dist = (point_x.sub(y[:, 1] / 512).square() + point_y.sub(y[:, 2] / 256).square())
  point_loss = det_gate.where(point_dist, 0).sum() / det_gate.sum().add(1e-6)

  target_cls = y[:, 0].cast(dtypes.int32).one_hot(2)
  quality = (1 - point_dist.sqrt().clamp(0, 1))
  quality = target_cls[:, 0].stack(quality, dim=1)
  cl_loss = mal_loss(pred[0], target_cls, quality, gamma=1.5)

  return cl_loss + x_loss + y_loss + point_loss

@TinyJit
def train_step(x, y, lr):
  yuv = rgb_to_yuv420_tensor(x)

  pred = model(yuv)
  loss = loss_fn(pred, y)

  optim.lr.assign(lr)
  optim.zero_grad()
  loss.backward()
  optim.step()

  if getenv("NAN", 0):
    gsums = {k: v.grad.square().sum() for k,v in get_state_dict(model).items() if v.grad is not None}
    wsums = {k: v.square().sum() for k,v in get_state_dict(model).items() if v.grad is not None}
    loss = loss.float().realize()
    for s in gsums.values(): s.realize()
    for s in wsums.values(): s.realize()
    return loss, gsums, wsums

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
        "y": BatchDesc(shape=(4,), dtype=dtypes.default_float),
      },
      load_single_file, get_train_files, bs=BS, shuffle=True,
    ), total=STEPS_PER_EPOCH, desc=f"epoch {epoch}"))
    i, proc = 0, single_batch(batch_iter)
    while proc is not None:
      st = time.perf_counter()
      GlobalCounters.reset()

      lr = get_lr(steps)
      loss = train_step(proc[0], proc[1], Tensor([lr], dtype=dtypes.float32))
      if getenv("NAN", 0):
        loss, gsums, wsums = loss
      pt = time.perf_counter()

      try: next_proc = single_batch(batch_iter)
      except StopIteration: next_proc = None
      dt = time.perf_counter()

      loss = loss.item()
      if getenv("NAN", 0):
        # check for NaNs
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
