import glob

from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import GlobalCounters
import cv2
import numpy as np

from .model import Model
from .common import pred
from ..common import BASE_PATH

if __name__ == "__main__":
  Tensor.no_grad = True
  Tensor.training = False

  model = Model()
  state_dict = safe_load(str(BASE_PATH / "vae.safetensors"))
  load_state_dict(model, state_dict)

  preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  i = 0
  while i < len(preprocessed_train_files):
    GlobalCounters.reset()

    file = preprocessed_train_files[i]
    img = cv2.imread(file)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # predict
    imgt = Tensor(np.array([img], dtype=np.uint8))
    oimg = pred(model, imgt)
    oimg = oimg.numpy()[0]

    oimg = cv2.resize(oimg, (512, 512))
    cv2.imshow("img", oimg)

    key = cv2.waitKey(0)
    if key == ord("q"): break
    elif key == ord("a"): i -= 1
    else: i += 1

  cv2.destroyAllWindows()
