from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
import numpy as np
import cv2

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...autoaim.model import Model
from ...autoaim.common import pred

def run():
  Tensor.no_grad = True
  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  model = Model()
  state_dict = safe_load(str(Path(__file__).parent.parent.parent.parent / "weights/model.safetensors"))
  load_state_dict(model, state_dict, verbose=False)
  if getenv("HALF", 0) == 1:
    for key, param in get_state_dict(model).items():
      if "norm" in key: continue
      if ".n" in key: continue
      param.replace(param.half()).realize()

  pub = messaging.Pub(["autoaim"])
  sub = messaging.Sub(["camera_feed"])

  while True:
    sub.update(0)
    GlobalCounters.reset()

    frame = sub["camera_feed"]

    framet = Tensor(frame, dtype=dtypes.uint8, device="PYTHON").reshape(256, 512, 3)
    model_out = pred(model, framet).tolist()[0]
    colorm, colorp, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, numberm, numberp = model_out
    match colorm:
      case 0: colorm = "none"
      case 1: colorm = "red"
      case 2: colorm = "blue"
      case 3: colorm = "blank"

    pub.send("autoaim", {
      "colorm": colorm,
      "colorp": colorp,
      "xc": xc,
      "yc": yc,
      "xtl": xtl,
      "ytl": ytl,
      "xtr": xtr,
      "ytr": ytr,
      "xbl": xbl,
      "ybl": ybl,
      "xbr": xbr,
      "ybr": ybr,
      "numberm": numberm,
      "numberp": numberp,
    })
