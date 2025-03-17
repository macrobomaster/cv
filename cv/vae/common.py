from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device
from tinygrad.dtype import dtypes

@TinyJit
def pred(model, img):
  img = img.to(Device.DEFAULT)
  oimg, _, _ = model(img)
  return (oimg.permute(0, 2, 3, 1) * 255).cast(dtypes.uint8).to("CPU")
