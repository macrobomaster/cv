import math

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import dedup
from tinygrad.extra.lr_scheduler import LR_Scheduler

class CLAMB(Optimizer):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-5, weight_decay=0.0, adam=False):
    super().__init__(params, lr)
    self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, weight_decay, adam
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous() for _ in [b1, b2])
    self.m = [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous() for t in self.params]
    self.v = [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous() for t in self.params]

  def schedule_step_with_grads(self, grads:list[Tensor]) -> list[Tensor]:
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, (t, g) in enumerate(zip(self.params, grads)):
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
      t.assign((t.detach() - self.lr * r * up).cast(t.dtype))
    return [self.b1_t, self.b2_t] + self.m + self.v

class GrokfastEMA:
  def __init__(self, params:list[Tensor], momentum, factor):
    self.params, self.momentum, self.factor = dedup([x for x in params if x.requires_grad]), momentum, factor
    self.t = [Tensor.zeros_like(p) for p in self.params]
  def update(self):
    for p, t in zip(self.params, self.t):
      assert p.grad is not None
      t.assign(self.momentum * t + (1 - self.momentum) * p.grad)
      p.grad.assign(p.grad + t * self.factor)

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
