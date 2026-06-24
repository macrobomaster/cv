import time, math

from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.device import Device
import wandb

from ..system.core.logging import logger
from ..system.core.keyvalue import kv_put
from ..common.dataloader import BatchDesc, Dataloader
from tinygrad.nn.optim import OptimizerGroup, Muon
from ..common.optim import CLAMB, TrapezoidalWarmupLR, MasterWeights, CosineWarmupLR
from .common import BASE_PATH
from .model import Model
from .data import get_train_files
from .common import T, IMG_H, IMG_W

GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1)))
BS = 64 * len(GPUS)
WARMUP_STEPS = 100
WARMPUP_LR = 1e-6
ADAMW_START_LR = 1e-3 * math.sqrt(BS / 256)
ADAMW_END_LR = 1e-4
MUON_START_LR = 1e-2 * math.sqrt(BS / 256)
MUON_END_LR = 1e-3
EPOCHS = 100
STEPS_PER_EPOCH = len(get_train_files())//BS

def run():
  Tensor.training = True

  if getenv("WANDB", 0):
    wandb.init(project="mrm_cv_autoaim")
    wandb.config.update({
      "warmup_steps": WARMUP_STEPS,
      "warmup_lr": WARMPUP_LR,
      "muon_start_lr": MUON_START_LR,
      "muon_end_lr": MUON_END_LR,
      "adamw_start_lr": ADAMW_START_LR,
      "adamw_end_lr": ADAMW_END_LR,
      "epochs": EPOCHS,
      "bs": BS,
      "steps_per_epoch": STEPS_PER_EPOCH,
    })

  dataloader = Dataloader({
    "x": BatchDesc(shape=(T, IMG_H, IMG_W, 3), dtype=dtypes.uint8),
    "y": BatchDesc(shape=(12,), dtype=dtypes.float32),
  }, bs=BS, files_fn=get_train_files)

  model = Model(temporal_size=T)
  state_dict = get_state_dict(model)
  for _, x in state_dict.items(): x.to_(GPUS)

  _NODECAY_KW = ("norm", "gamma", "beta", "bias", "ls", "corner_tokens", "pos_emb", "embed", "level")
  _DECAY_PATHS = ("backbone.stem.conv", "decoder.class_proj", "decoder.corner_mlp.out", "decoder.objness", "decoder.ref_head")
  def _classify(name, p):
    if p.ndim < 2 or any(k in name for k in _NODECAY_KW): return "nodecay"
    if any(path in name for path in _DECAY_PATHS): return "decay"
    return "muon"
  muon_params, decay_params, nodecay_params = [], [], []
  for name, p in state_dict.items():
    if not p.is_param: continue
    {"muon": muon_params, "decay": decay_params, "nodecay": nodecay_params}[_classify(name, p)].append(p)
  logger.info(f"Muon params: {len(muon_params)}, AdamW-decay: {len(decay_params)}, AdamW-nodecay: {len(nodecay_params)}")

  muon_optim = MasterWeights(muon_params, Muon, lr=MUON_START_LR, weight_decay=0.01)
  adamw_decay = MasterWeights(decay_params, CLAMB, lr=ADAMW_START_LR, weight_decay=0.01, adam=True)
  adamw_nodecay = MasterWeights(nodecay_params, CLAMB, lr=ADAMW_START_LR, weight_decay=0.0, adam=True)
  optim = OptimizerGroup(muon_optim, adamw_decay, adamw_nodecay)
  muon_lr_sched = CosineWarmupLR(muon_optim, WARMUP_STEPS, WARMPUP_LR, MUON_START_LR, MUON_END_LR, EPOCHS, STEPS_PER_EPOCH)
  adamw_decay_sched = CosineWarmupLR(adamw_decay, WARMUP_STEPS, WARMPUP_LR, ADAMW_START_LR, ADAMW_END_LR, EPOCHS, STEPS_PER_EPOCH)
  adamw_nodecay_sched = CosineWarmupLR(adamw_nodecay, WARMUP_STEPS, WARMPUP_LR, ADAMW_START_LR, ADAMW_END_LR, EPOCHS, STEPS_PER_EPOCH)

  if (ckpt := getenv("CKPT", "")) != "":
    logger.info(f"loading checkpoint {BASE_PATH / 'intermediate' / f'model_{ckpt}.safetensors'}")
    state_dict = safe_load(BASE_PATH / "intermediate" / f"model_{ckpt}.safetensors")

    if getenv("PRETRAINED_BACKBONE"):
      logger.info("only loading backbone")
      state_dict = {k[9:]: v for k,v in state_dict.items() if k.startswith("backbone.")}
      load_state_dict(model.backbone, state_dict, strict=False)
    else:
      logger.info("loading whole model")
      load_state_dict(model, state_dict, strict=False)

    if getenv("CKPT_OPTIM"):
      logger.info(f"loading optimizer {BASE_PATH / 'intermediate' / f'optim_{ckpt}.safetensors'}")
      state_dict = safe_load(BASE_PATH / "intermediate" / f"optim_{ckpt}.safetensors")
      load_state_dict(optim, state_dict, strict=False)

  for p in optim.params:
    p.grad = p.zeros_like().contiguous()
  grads = [p.grad for p in optim.params]

  masters = [m for o in (muon_optim, adamw_decay, adamw_nodecay) for m in getattr(o, "master", ())]
  Tensor.realize(*optim.params, *masters)

  @TinyJit
  def train_step(x, y):
    total, class_l, l1_l, dfl_l, lsd_l, geom_l, obj_l = model(x, y)
    total.backward()
    loss_cpu = Tensor.stack(total, class_l, l1_l, dfl_l, lsd_l, geom_l, obj_l).float().to("CPU")
    return loss_cpu.realize(*grads)

  @TinyJit
  def optim_step():
    optim.step()
    grad_norm = Tensor.stack(*[g.float().square().sum() for g in grads]).sum().sqrt().contiguous()

    muon_lr_sched.step()
    adamw_decay_sched.step()
    adamw_nodecay_sched.step()

    for g in grads: g.assign(0)

    muon_lr_cpu = muon_optim.lr.float().to("CPU")
    adamw_lr_cpu = adamw_decay.lr.float().to("CPU")
    grad_norm_cpu = grad_norm.float().to("CPU")
    Tensor.realize(muon_lr_cpu, adamw_lr_cpu, grad_norm_cpu, *grads)

    return muon_lr_cpu, adamw_lr_cpu, grad_norm_cpu

  steps = 0
  for epoch in range(EPOCHS if not getenv("VIZ") else 1):
    dataloader.load()
    i, d = 0, dataloader.next(GPUS)
    while d is not None and (i < 6 or not getenv("VIZ")):
      GlobalCounters.reset()
      st = time.perf_counter()

      loss = train_step(*d[:-1])
      ret = optim_step()
      pt = time.perf_counter()

      try: next_d = dataloader.next(GPUS)
      except StopIteration: next_d = None
      dt = time.perf_counter()

      total_loss, class_loss, l1_loss, dfl_loss, lsd_loss, geom_loss, obj_loss = loss.tolist()
      muon_lr, adamw_lr, grad_norm, = map(lambda x: x.item(), ret)
      at = time.perf_counter()

      logger.info(
        f"{epoch:3} {i:5}/{STEPS_PER_EPOCH} {((at - st)) * 1000.0:7.2f} ms step, "
        f"{(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
        f"{total_loss:9.4f} loss (cls {class_loss:.4f} | l1 {l1_loss:.4f} | dfl {dfl_loss:.4f} | lsd {lsd_loss:.4f} | geom {geom_loss:.4f} | obj {obj_loss:.4f}), "
        f"{grad_norm:9.4f} grad_norm, {muon_lr:.6f} muon_lr, {adamw_lr:.6f} adamw_lr, "
        f"{GlobalCounters.mem_used / 1e9:7.2f} GB used, {GlobalCounters.mem_used * 1e-9 / (at - st):9.2f} GB/s, "
        f"{GlobalCounters.global_ops * 1e-9 / (at - st):9.2f} GFLOPS"
      )

      if getenv("WANDB", 0):
        wandb.log({
          "epoch": epoch + (i + 1) / STEPS_PER_EPOCH,
          "step_time": at - st, "python_time": pt - st, "data_time": dt - pt, "accel_time": at - dt,
          "loss": total_loss, "class_loss": class_loss, "l1_loss": l1_loss, "dfl_loss": dfl_loss, "lsd_loss": lsd_loss, "geom_loss": geom_loss, "obj_loss": obj_loss,
          "grad_norm": grad_norm, "muon_lr": muon_lr, "adamw_lr": adamw_lr,
          "gb": GlobalCounters.mem_used / 1e9, "gbps": GlobalCounters.mem_used * 1e-9 / (at - st),
          "gflops": GlobalCounters.global_ops * 1e-9 / (at - st)
        })

      d, next_d = next_d, None
      i += 1
      steps += 1

    if not getenv("VIZ"):
      logger.info(f"saving checkpoint for epoch {epoch} at {BASE_PATH / 'intermediate' / f'model_{epoch}.safetensors'}")
      safe_save(get_state_dict(model), str(BASE_PATH / f"intermediate/model_{epoch}.safetensors"))
      safe_save(get_state_dict(optim), str(BASE_PATH / f"intermediate/optim_{epoch}.safetensors"))

  if not getenv("VIZ"):
    logger.info(f"final save to {BASE_PATH / 'model.safetensors'}")
    with open(BASE_PATH / "intermediate" / f"model_{epoch}.safetensors", "rb") as f:
      with open(BASE_PATH / "model.safetensors", "wb") as f2: f2.write(f.read())

  wandb.finish()
  if not getenv("VIZ"): kv_put("global", "do_shutdown", True)
