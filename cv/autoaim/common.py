from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit

@TinyJit
def pred(model, img):
  cl, cl_prob, x, y = model(img)
  return cl, cl_prob, x, y, Tensor([0])
