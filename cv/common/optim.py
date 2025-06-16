import math

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import dedup, FUSE_OPTIM
from tinygrad.extra.lr_scheduler import LR_Scheduler
from tinygrad.nn.state import get_parameters, get_state_dict

class CLAMB(Optimizer):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, adam=False, fused=FUSE_OPTIM):
    super().__init__(params, lr, fused)
    self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, weight_decay, adam
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous() for _ in [b1, b2])
    self.m = self._new_optim_param()
    self.v = self._new_optim_param()

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, (t, g) in enumerate(zip(params, grads)):
      self.m[i].assign((self.b1 * self.m[i] + (1.0 - self.b1) * g).cast(t.dtype))
      self.v[i].assign((self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).cast(t.dtype))
      m_hat = self.m[i] / (1.0 - self.b1_t)
      v_hat = self.v[i] / (1.0 - self.b2_t)
      cmask = (m_hat * g > 0).cast(t.dtype)
      cmask = cmask / cmask.mean().clamp(min_=1e-3)
      up = ((m_hat * cmask) / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      ret.append((t.detach() - self.lr * r * up).cast(t.dtype))
    return ret, [self.b1_t, self.b2_t] + self.m + self.v

class CLaProp(Optimizer):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, fused=FUSE_OPTIM):
    super().__init__(params, lr, fused)
    self.b1, self.b2, self.eps, self.wd = b1, b2, eps, weight_decay
    self.b1_t, self.b2_t = (Tensor.zeros((1,), dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous().realize() for _ in [b1, b2])
    self.exp_avg = [t.realize() for t in self._new_optim_param()]
    self.exp_avg_sq = [t.realize() for t in self._new_optim_param()]

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    self.b1_t.assign(self.b1 * self.b1_t + (1 - self.b1) * self.lr)
    self.b2_t.assign(self.b2 * self.b2_t + (1 - self.b2))
    for i, (t, g) in enumerate(zip(params, grads)):
      self.exp_avg_sq[i].assign(self.b2 * self.exp_avg_sq[i] + (1 - self.b2) * g.square())

      bias_correction1 = self.b1_t / self.lr
      bias_correction2 = self.b2_t
      step_size = 1 / bias_correction1

      denom = self.exp_avg_sq[i].div(bias_correction2).sqrt().add(self.eps)
      step_of_this_grad = g / denom
      self.exp_avg[i].assign(self.b1 * self.exp_avg[i] + (1 - self.b1) * self.lr * step_of_this_grad)

      cmask = (self.exp_avg[i] * g > 0).cast(t.dtype)
      cmask = cmask / cmask.mean().clamp(min_=1e-3)

      ret.append((t.detach() - step_size * self.exp_avg[i] * cmask - self.lr * self.wd * t.detach()))
    return ret, [self.b1_t, self.b2_t] + self.exp_avg + self.exp_avg_sq

class GrokfastEMA:
  def __init__(self, params:list[Tensor], momentum, factor):
    self.params, self.momentum, self.factor = dedup([x for x in params if x.requires_grad]), momentum, factor
    self.t = [Tensor.zeros_like(p) for p in self.params]
  def update(self):
    for p, t in zip(self.params, self.t):
      assert p.grad is not None
      t.assign(self.momentum * t + (1 - self.momentum) * p.grad)
      p.grad.assign(p.grad + t * self.factor)

class SwitchEMA:
  def __init__(self, model, ema_model, epochs:int, steps_per_epoch:int, momentum:float=0.999):
    self.step_counter = Tensor([0], requires_grad=False, device=get_parameters(model)[0].device)
    self.epochs, self.steps_per_epoch = epochs, steps_per_epoch
    self.momentum = momentum
    self.model_params = {k:v for k,v in get_state_dict(model).items() if v.requires_grad}
    ema_state_dict = get_state_dict(ema_model)
    self.ema_model_params = {k:ema_state_dict[k] for k in self.model_params.keys()}
    for p_ema in self.ema_model_params.values(): p_ema.requires_grad = False

  def update(self):
    for p, p_ema in zip(self.model_params.values(), self.ema_model_params.values()):
      p_ema.assign(p_ema * self.momentum + p.detach() * (1 - self.momentum))
    for p, p_ema in zip(self.model_params.values(), self.ema_model_params.values()):
      p.assign((self.step_counter == self.steps_per_epoch - 1).where(p_ema, p).detach())
    self.step_counter.assign((self.step_counter == self.steps_per_epoch - 1).where(0, self.step_counter + 1))
    Tensor.realize(*self.model_params.values(), *self.ema_model_params.values(), self.step_counter)

class CosineWarmupLR(LR_Scheduler):
  def __init__(self, optimizer:Optimizer, warmup_steps:int, warmup_lr:float, start_lr:float, end_lr:float, epochs:int, steps_per_epoch:int):
    super().__init__(optimizer)
    self.warmup_steps, self.epochs, self.steps_per_epoch = warmup_steps, epochs, steps_per_epoch
    self.warmup_lr, self.start_lr, self.end_lr = warmup_lr, start_lr, end_lr
    self.optimizer.lr.assign(self.get_lr()).realize()

  def get_lr(self):
    return (self.epoch_counter < self.warmup_steps).where(
      self.start_lr * (self.epoch_counter / self.warmup_steps) + self.warmup_lr * (1 - self.epoch_counter / self.warmup_steps),
      self.end_lr + 0.5 * (self.start_lr - self.end_lr) * (1 + (((self.epoch_counter - self.warmup_steps) / ((self.epochs * self.steps_per_epoch) - self.warmup_steps)) * math.pi).cos())
    ).cast(self.optimizer.lr.dtype)

def grad_clip_norm(optim:Optimizer, max_norm:float=1.0):
  global_norm = None
  for p in optim.params:
    if p.grad is not None:
      norm = p.grad.cast(dtypes.float32).square().sum()
      if global_norm is None: global_norm = norm
      else: global_norm += norm
  global_norm = global_norm.sqrt().contiguous()
  for p in optim.params:
    if p.grad is not None:
      p.grad = (global_norm > max_norm).where(p.grad.div(global_norm), p.grad)
  return global_norm

class Schedule:
  def __init__(self, device=None):
    self.t = Tensor([0], requires_grad=False, dtype=dtypes.int32, device=device)
  def step(self):
    self.t.assign(self.t + 1).realize()
  def get(self):
    raise NotImplementedError

class CosineSchedule(Schedule):
  def __init__(self, start:float, end:float, steps:int, device=None):
    super().__init__(device)
    self.start, self.end, self.steps = start, end, steps

  def get(self):
    return (self.t <= self.steps).where(
      self.end + 0.5 * (self.start - self.end) * (1 + ((self.t / self.steps) * math.pi).cos()),
      self.end
    )

class ExpSchedule(Schedule):
  def __init__(self, start:float, end:float, steps:int, device=None):
    super().__init__(device)
    self.start, self.end, self.steps = start, end, steps

  def get(self):
    return (self.t <= self.steps).where(
      self.start * ((self.end / self.start) ** (self.t / self.steps)),
      self.end
    )

class StepSchedule(Schedule):
  def __init__(self, before:float, after:float, crossover:int, device=None):
    super().__init__(device)
    self.before, self.after, self.crossover = before, after, crossover

  def get(self):
    return (self.t <= self.crossover).where(self.before, self.after)
