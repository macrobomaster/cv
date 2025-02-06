import time

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad.helpers import GlobalCounters, getenv
import cv2
import numpy as np

from .model import Model
from .common import pred
from ..common import BASE_PATH
from ..common.camera import setup_aravis, get_aravis_frame
from ..common.image import bgr_to_yuv420

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  cam, strm = setup_aravis()

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)
  if getenv("HALF", 0) == 1:
    for key, param in get_state_dict(model).items():
      if "norm" in key: continue
      if ".n" in key: continue
      param.replace(param.half()).realize()

  st = time.perf_counter()
  while True:
    GlobalCounters.reset()

    spt = time.perf_counter()
    img = get_aravis_frame(cam, strm)
    # resize and convert to yuv
    img = cv2.resize(img, (512, 256))
    # increase brightness
    # img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    yuv = bgr_to_yuv420(img)
    pt = time.perf_counter() - spt
    print(f"frame aquisition time: {pt:.3f}")

    # run model
    yuvt = Tensor(np.expand_dims(yuv, 0), dtype=dtypes.uint8)
    detected, det_prob, x, y, dist = pred(model, yuvt)
    detected, det_prob, x, y, dist = detected.item(), det_prob.item(), x.item(), y.item(), dist.item()

    dt = time.perf_counter() - st
    st = time.perf_counter()
    cv2.putText(img, f"{1/dt:.2f} FPS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 250, 55), 1)

    # draw the annotation
    cv2.putText(img, f"{detected}: {det_prob:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if detected == 1 and det_prob > 0.6:
      x, y = int(x), int(y)
      cv2.circle(img, (x, y), int(max(10 - dist, 2)), (0, 255, 0), -1)
      cv2.putText(img, f"{dist:.3f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("img", img)

    # display
    cv2.imshow("img", img)

    key = cv2.waitKey(1)
    if key == ord("q"): break

  cv2.destroyAllWindows()
  cam.stop_acquisition()
