import glob

from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import GlobalCounters
import cv2
import numpy as np

from .model import Model
from .common import pred
from ..common import BASE_PATH
from ..common.image import bgr_to_yuv420

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)

  preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  i = 0
  while i < len(preprocessed_train_files):
    GlobalCounters.reset()

    file = preprocessed_train_files[i]
    img = cv2.imread(file)
    img = cv2.resize(img, (512, 256))
    yuv = bgr_to_yuv420(img)

    # predict
    yuvt = Tensor(np.array([yuv], dtype=np.uint8))
    detected, det_prob, x, y, dist = pred(model, yuvt)
    detected, det_prob, x, y, dist = detected.item(), det_prob.item(), x.item(), y.item(), dist.item()

    # draw the annotation
    cv2.putText(img, f"{detected}: {det_prob:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if detected == 1 and det_prob > 0.6:
      x, y = int(x), int(y)
      cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
      cv2.putText(img, f"{dist:.3f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)
    if key == ord("q"): break
    elif key == ord("a"): i -= 1
    else: i += 1

  cv2.destroyAllWindows()
