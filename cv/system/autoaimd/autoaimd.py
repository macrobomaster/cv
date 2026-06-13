from pathlib import Path
from typing import Callable, Any
import pickle, time

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict

from ..core import messaging
from ..core.logging import logger
from ..core.keyvalue import kv_get, kv_put
from ...autoaim.model import Model, CLASS_DECODE_TABLE, N_FEAT_TOKENS
from ...autoaim.common import pred_backbone, pred_decoder, TemporalInference, MODEL_VERSION, IMG_H, IMG_W, T

HALF = getenv("HALF", 0)
BEAM = getenv("BEAM", 0) or getenv("JITBEAM", 0)

def run():
  pub = messaging.Pub(["autoaim"])
  sub = messaging.Sub(["camera_feed"])

  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  # cache model jit — separate backbone and decoder
  backbone_key = f"model_{MODEL_VERSION}_{HALF}_{BEAM}_backbone_{Device.DEFAULT}"
  decoder_key = f"model_{MODEL_VERSION}_{HALF}_{BEAM}_decoder_{Device.DEFAULT}"
  if kv_get("autoaim", backbone_key) is None:
    logger.info("building cached model")

    model = Model(temporal_size=T)
    state_dict = safe_load(str(Path(__file__).parent.parent.parent.parent / "weights/model.safetensors"))
    load_state_dict(model, state_dict, verbose=False, strict=False)
    model.fuse()
    if HALF:
      for key, param in get_state_dict(model).items():
        if "norm" in key: continue
        if ".n" in key: continue
        param.replace(param.half()).realize()

    # warmup backbone jit (input is now T frames)
    fake_input = Tensor.empty(1, T, IMG_H, IMG_W, 3, dtype=dtypes.uint8, device="PYTHON").realize()
    for _ in range(3):
      tokens = pred_backbone(model, fake_input)
      tokens.tolist()

    # warmup decoder jit (no T multiplier — temporal is fused at stem)
    fake_tokens = Tensor.empty(1, N_FEAT_TOKENS, 512, device="PYTHON").realize()
    for _ in range(3):
      pred_decoder(model, fake_tokens).tolist()

    kv_put("autoaim", backbone_key, pickle.dumps(pred_backbone))
    kv_put("autoaim", decoder_key, pickle.dumps(pred_decoder))

    # Request restart after building cached model
    logger.info("cached model built, requesting restart")
    kv_put("restart", "autoaimd", True)
    return  # Exit to allow supervisor to restart us

  # load model
  logger.info(f"loading cached {backbone_key}")
  model_backbone: Callable[[Any, Tensor], Tensor] = pickle.loads(kv_get("autoaim", backbone_key))
  model_decoder: Callable[[Any, Tensor], Tensor] = pickle.loads(kv_get("autoaim", decoder_key))

  color_names = {0: "none", 1: "red", 2: "blue"}

  infer = TemporalInference(model_backbone, model_decoder, None, T=T)

  while True:
    sub.update(0)
    GlobalCounters.reset()

    camera_feed = sub["camera_feed"]
    if camera_feed is None: continue

    if sub.updated["camera_feed"]:
      frame = camera_feed["frame"]
      framet = Tensor(frame, dtype=dtypes.uint8, device="PYTHON").reshape(256, 512, 3)
      ft = time.monotonic()

      model_out = infer(framet)
      mt = time.monotonic()

      class_id = int(model_out[0])
      confidence = model_out[1]
      corners = list(model_out[2:10])  # 4 corners in [0, 1] image-normalized coords, TL/TR/BL/BR

      detected, color_id, number = CLASS_DECODE_TABLE[class_id]
      color_name = color_names.get(color_id, "blank")

      valid = detected == 1 and confidence > 0.6

      pub.send("autoaim", {
        "valid": valid,
        "class_id": class_id,
        "confidence": confidence,
        "detected": detected,
        "color": color_name,
        "number": number,
        "corners": corners,
      })
