from tinygrad.engine.jit import TinyJit
from tinygrad.dtype import dtypes

@TinyJit
def pred(model, img):
  oimg, _, _ = model(img)
  return (oimg.permute(0, 2, 3, 1) * 255).cast(dtypes.uint8)
