import glob

from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import GlobalCounters
import cv2
import numpy as np

from ..autoaim.model import Backbone
from .model import Model
from ..common import BASE_PATH
from ..common.image import rgb_to_yuv420_tensor

@TinyJit
def pred(encoder, sideband, model, x):
  yuv = rgb_to_yuv420_tensor(x.to(Device.DEFAULT))
  yuv = yuv.cast(dtypes.default_float).permute(0, 3, 1, 2).div(255)
  yuv_mean, yuv_std = yuv.mean([2, 3], keepdim=True), yuv.std([2, 3], keepdim=True)
  yuv = yuv.sub(yuv_mean).div(yuv_std.add(1e-6))

  with Tensor.train(False), Tensor.test(True):
    z = encoder(yuv, sideband.expand(yuv.shape[0], -1))[-1].detach()
  x = model(z)
  return x.permute(0, 2, 3, 1).mul(255).clamp(0, 255).cast(dtypes.uint8).to("CPU")

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  encoder = Backbone(cin=6, cstage=[16, 32, 64, 128], stages=[2, 2, 6, 2], sideband=4, sideband_only=True, dropout=0)
  state_dict = safe_load(BASE_PATH / "model.safetensors")
  state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
  load_state_dict(encoder, state_dict)
  sideband = safe_load(BASE_PATH / "model.safetensors")["sideband"].to(Device.DEFAULT)

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "vae.safetensors"))
  load_state_dict(model, state_dict)

  preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  i = 0
  while i < len(preprocessed_train_files):
    GlobalCounters.reset()

    file = preprocessed_train_files[i]
    img = cv2.imread(file)
    img = cv2.resize(img, (512, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # predict
    imgt = Tensor(np.array([img], dtype=np.uint8), device="NPY")
    oimg = pred(encoder, sideband, model, imgt)
    oimg = oimg.numpy()[0]

    oimg = cv2.resize(oimg, (256, 128))
    oimg = cv2.cvtColor(oimg, cv2.COLOR_RGB2BGR)

    stack = np.hstack((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.resize(oimg, (512, 256))))
    cv2.imshow("img", stack)

    key = cv2.waitKey(0)
    if key == ord("q"): break
    elif key == ord("a"): i -= 1
    elif key == ord("f"): i += 100
    else: i += 1

  cv2.destroyAllWindows()
