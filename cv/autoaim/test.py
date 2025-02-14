import glob

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad.helpers import GlobalCounters, getenv
import cv2

from .model import Model
from .common import pred
from ..common import BASE_PATH

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)
  if getenv("HALF", 0) == 1:
    for key, param in get_state_dict(model).items():
      if "norm" in key: continue
      if ".n" in key: continue
      param.replace(param.half()).realize()

  preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  i = 0
  while i < len(preprocessed_train_files):
    GlobalCounters.reset()

    file = preprocessed_train_files[i]
    img = cv2.imread(file)
    img = cv2.resize(img, (512, 256))

    # predict
    model_out = pred(model, Tensor(img, device="NPY")).numpy()[0]
    detected, x, y = model_out[0], model_out[1], model_out[2]

    # draw the annotation
    cv2.putText(img, f"{detected:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if detected > 0.0:
      x, y = int(x), int(y)
      cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
      # cv2.putText(img, f"{dist:.3f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)
    if key == ord("q"): break
    elif key == ord("a"): i -= 1
    elif key == ord("f"): i += 100
    else: i += 1

  cv2.destroyAllWindows()
