from pathlib import Path
import pickle

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...autoaim.model import Model
from ...autoaim.common import pred, MODEL_VERSION

HALF = getenv("HALF", 0)
BEAM = getenv("BEAM", 0) or getenv("JITBEAM", 0)

def run():
  pub = messaging.Pub(["autoaim"])
  sub = messaging.Sub(["camera_feed"])

  Tensor.no_grad = True
  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  # cache model jit
  if kv_get("autoaim", f"model_{MODEL_VERSION}_{HALF}_{BEAM}_run") is None:
    logger.info("building cached model")

    model = Model()
    state_dict = safe_load(str(Path(__file__).parent.parent.parent.parent / "weights/model.safetensors"))
    load_state_dict(model, state_dict, verbose=False)
    if HALF:
      for key, param in get_state_dict(model).items():
        if "norm" in key: continue
        if ".n" in key: continue
        param.replace(param.half()).realize()

    # run to initialize jit
    fake_input = Tensor.empty(256, 512, 3, dtype=dtypes.uint8, device="PYTHON").realize()
    for _ in range(3):
      pred(model, fake_input).tolist()

    kv_put("autoaim", f"model_{MODEL_VERSION}_{HALF}_{BEAM}_run", pickle.dumps(pred))

  # load model
  logger.info(f"loading cached model_{MODEL_VERSION}_{HALF}_{BEAM}_run")
  model_pred = pickle.loads(kv_get("autoaim", f"model_{MODEL_VERSION}_{HALF}_{BEAM}_run"))

  while True:
    sub.update(0)
    GlobalCounters.reset()

    camera_feed = sub["camera_feed"]
    if camera_feed is None: continue

    if sub.updated["camera_feed"]:
      frame = camera_feed["frame"]
      framet = Tensor(frame, dtype=dtypes.uint8, device="PYTHON").reshape(256, 512, 3)

      model_out = model_pred(None, framet).tolist()[0]

      colorm, colorp, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, numberm, numberp = model_out
      match colorm:
        case 0: colorm = "none"
        case 1: colorm = "red"
        case 2: colorm = "blue"
        case 3: colorm = "blank"

      valid = colorm != "none" and colorp > 0.6

      pub.send("autoaim", {
        "valid": valid,
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
