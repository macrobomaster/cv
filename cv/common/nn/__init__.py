from typing import Literal

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad import nn

from .norm import RMSNorm

class LayerScale:
  def __init__(self, dim:int, init:float=1e-4):
    self.gamma = Tensor.ones(dim, dtype=dtypes.float32) * init

  def __call__(self, x:Tensor, xx:Tensor, dropout:float=0.0) -> Tensor:
    return x + (xx * self.gamma.cast(x.dtype)).dropout(dropout)

class LayerScale2d(LayerScale):
  def __call__(self, x:Tensor, xx:Tensor, dropout:float=0.0) -> Tensor:
    return x + (xx * self.gamma.cast(x.dtype).reshape(1, -1, 1, 1)).dropout(dropout)

class Attention:
  """
  Cross and Self Attention with qk-norm, gqa, xsa, and configurable output modulation
  """
  def __init__(self, dim:int, qk_dim:int, heads:int, kv_heads:int=0, out:Literal["proj", "mod"]|None="proj", dropout:float=0.0):
    if kv_heads == 0: kv_heads = heads
    assert qk_dim % heads == 0, "qk_dim must be divisible by heads"
    assert out in ["proj", "mod", None], "out must be one of 'proj', 'mod', or None"

    self.dropout = dropout

    self.dim, self.qk_dim, self.heads, self.kv_heads = dim, qk_dim, heads, kv_heads
    self.head_dim, self.value_dim = qk_dim // heads, dim // heads
    self.q = nn.Linear(dim, qk_dim, bias=False)
    self.kv = nn.Linear(dim, self.kv_heads * (self.head_dim + self.value_dim), bias=False)

    self.q_norm = RMSNorm(self.head_dim)
    self.k_norm = RMSNorm(self.head_dim)

    self.out = out
    match out:
      case "proj":
        self.proj = nn.Linear(dim, dim, bias=False)
      case "mod":
        self.gate = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

  def __call__(self, x:Tensor, kv:Tensor|None=None) -> Tensor:
    b, t, c = x.shape
    if kv is not None: kvt = kv.shape[1]
    else: kvt = t

    # q, k, v
    q = self.q(x)
    k, v = self.kv(x if kv is None else kv).split([self.kv_heads * self.head_dim, self.kv_heads * self.value_dim], dim=-1)
    q = self.q_norm(q.reshape(b, t, self.heads, self.head_dim))
    k = self.k_norm(k.reshape(b, kvt, self.kv_heads, self.head_dim))
    v = v.reshape(b, kvt, self.kv_heads, self.value_dim)

    # sdpa
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    attn = q.scaled_dot_product_attention(k, v, enable_gqa=True, dropout_p=self.dropout)

    # xsa
    if t == kvt:
      vn = v.normalize(dim=-1)
      if self.heads != self.kv_heads: vn = vn.repeat_interleave(self.heads // self.kv_heads, dim=1)
      attn = attn - (attn * vn).sum(axis=-1, keepdim=True) * vn

    attn = attn.transpose(1, 2).reshape(b, t, c)

    # output modulation
    match self.out:
      case "proj":
        return self.proj(attn)
      case "mod":
        return self.proj(attn.hardsigmoid() * self.gate(x).hardsigmoid())
      case _: return attn

class FFN:
  def __init__(self, cin:int, cout:int=0, exp:int=2, norm:bool=True, bias:bool=True, dropout:float=0.0):
    if cout == 0: cout = cin
    self.cin, self.cout, self.exp = cin, cout, exp
    self.dropout = dropout
    if norm: self.norm = RMSNorm(cin)
    self.up = nn.Linear(cin, cout * exp, bias=bias)
    self.gate = nn.Linear(cin, cout * exp, bias=bias)
    self.down = nn.Linear(cout * exp, cout, bias=bias)

  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "norm"): x = self.norm(x)
    x = self.up(x) * self.gate(x).swish()
    x = self.down(x)
    return x.dropout(self.dropout)

class MLP:
  def __init__(self, in_dim:int, out_dim:int, mid_dim:int, blocks:int=1, bias:bool=False):
    self.proj_in = nn.Linear(in_dim, mid_dim, bias=bias)
    self.layers = [nn.Linear(mid_dim, mid_dim, bias=bias) for _ in range(blocks)]
    self.ls = [LayerScale(mid_dim) for _ in range(blocks)]
    self.out = nn.Linear(mid_dim, out_dim, bias=bias)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.proj_in(x).gelu()
    for lin, ls in zip(self.layers, self.ls):
      x = ls(x, lin(x).gelu())
    return self.out(x)
