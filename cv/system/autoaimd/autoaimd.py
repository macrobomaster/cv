from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict

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
  load_state_dict(model, state_dict)
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
    if frame is None: continue
    framet = Tensor(frame, dtype=dtypes.uint8, device="PYTHON").reshape(256, 512, 3)
    model_out = pred(model, framet).tolist()[0]
    cl, clp, x, y, colorm, colorp, numberm, numberp = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5], model_out[6], model_out[7]
    match colorm:
      case 1: colorm = "red"
      case 2: colorm = "blue"
      case _: colorm = "blank"

    pub.send("autoaim", {
      "cl": cl,
      "clp": clp,
      "x": x,
      "y": y,
      "colorm": colorm,
      "colorp": colorp,
      "numberm": numberm,
      "numberp": numberp
    })
