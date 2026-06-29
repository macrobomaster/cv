from pathlib import Path
from typing import Callable, Any
import pickle, time, math

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...autoaim.model import Model, CLASS_DECODE_TABLE, QUALITY_TAU, MAL_GAMMA
from ...autoaim.common import pred, TemporalInference, MODEL_VERSION, IMG_H, IMG_W, T

HALF = getenv("HALF", 0)
BEAM = getenv("BEAM", 0) or getenv("JITBEAM", 0)
FUSE = getenv("FUSE", 0)
TARGET_COLOR = getenv("TARGET_COLOR", 1)
COLOR_TO_ID = {"red": 0, "blue": 1}
ID_TO_COLOR = {0: "red", 1: "blue"}

VALID_REL_ERR = 0.10
VALID_CONF_THRESH = math.exp(-MAL_GAMMA * VALID_REL_ERR / QUALITY_TAU)

def run():
  pub = messaging.Pub(["autoaim"])
  sub = messaging.Sub(["camera_feed", "team_color"], poll="camera_feed")

  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  # cache model jit
  model_key = f"model_{MODEL_VERSION}_{HALF}_{BEAM}_{FUSE}_{Device.DEFAULT}"
  if kv_get("autoaim", model_key) is None:
    logger.info("building cached model")

    model = Model(temporal_size=T)
    state_dict = safe_load(str(Path(__file__).parent.parent.parent.parent / "weights/model.safetensors"))
    load_state_dict(model, state_dict, verbose=False, strict=False)
    if FUSE: model.fuse()
    if HALF:
      for param in get_state_dict(model).values():
        if param.ndim < 2: continue
        param.replace(param.half())

    Tensor.realize(*get_state_dict(model).values())

    # warmup jit
    infer = TemporalInference(pred, T=T, model=model)
    kv_put("autoaim", model_key, pickle.dumps(infer.warmup()))

    logger.info("cached model built, requesting restart")
    kv_put("restart", "autoaimd", True)
    return

  # load model
  logger.info(f"loading cached {model_key}")
  logger.info(f"autoaimd backend={Device.DEFAULT} HALF={HALF} FUSE={FUSE} BEAM={BEAM} T={T}")
  model_fn: Callable[[Any, Tensor, Tensor], Tensor] = pickle.loads(kv_get("autoaim", model_key))

  color_names = {0: "none", 1: "red", 2: "blue"}
  target_color = TARGET_COLOR
  team_color = None

  infer = TemporalInference(model_fn, T=T)
  fid = 0

  while True:
    sub.update(0)
    GlobalCounters.reset()

    camera_feed = sub["camera_feed"]
    if camera_feed is None: continue

    if sub.updated["camera_feed"]:
      frame = camera_feed["frame"]
      framet = Tensor(frame, dtype=dtypes.uint8, device="PYTHON")
      ft = time.monotonic()

      if sub.updated["team_color"]:
        team_color = sub["team_color"]
        if team_color in COLOR_TO_ID:
          # team_color is our alliance; the model target is the enemy armor color.
          target_color = 1 - COLOR_TO_ID[team_color]
          logger.info(f"autoaimd: team={team_color}, target={ID_TO_COLOR[target_color]}")
        else:
          logger.warning(f"autoaimd: unknown team_color {team_color!r}; "
                         f"keeping target={ID_TO_COLOR.get(target_color, target_color)}")

      model_out = infer(framet, target_color=target_color)
      mt = time.monotonic()

      class_id = int(model_out[0])
      confidence = model_out[1]
      corners = list(model_out[2:10])
      corner_lo = list(model_out[10:18])
      corner_hi = list(model_out[18:26])

      detected, color_id, number = CLASS_DECODE_TABLE[class_id]
      color_name = color_names.get(color_id, "blank")

      valid = detected == 1 and confidence > VALID_CONF_THRESH

      fid += 1
      pub.send("autoaim", {
        "valid": valid,
        "class_id": class_id,
        "confidence": confidence,
        "detected": detected,
        "color": color_name,
        "team_color": team_color,
        "target_color": ID_TO_COLOR.get(target_color, target_color),
        "number": number,
        "corners": corners,
        "corner_lo": corner_lo,
        "corner_hi": corner_hi,
        "t_capture": camera_feed["ct"],
        "fid": fid,
        "infer_ms": (mt - ft) * 1e3,
      })
