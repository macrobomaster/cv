import glob

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad.helpers import GlobalCounters, getenv
import cv2

from .model import Model, CLASS_DECODE_TABLE
from .common import pred_backbone, pred_decoder, TemporalInference, T, IMG_H, IMG_W
from ..common import BASE_PATH

if __name__ == "__main__":
  Tensor.training = False
  if getenv("HALF", 0) == 1:
    dtypes.default_float = dtypes.float16

  model = Model(temporal_size=T)
  state_dict = safe_load(str(BASE_PATH / "model.safetensors"))
  load_state_dict(model, state_dict)
  if getenv("HALF", 0) == 1:
    for key, param in get_state_dict(model).items():
      if "norm" in key: continue
      if ".n" in key: continue
      param.replace(param.half()).realize()

  infer = TemporalInference(pred_backbone, pred_decoder, model, T=T)

  # syndata sanity check — confirms the model works on its training distribution.
  # if these are wrong, it's an inference-path bug, not sim-to-real.
  if getenv("SYNCHECK", 1):
    from .syndata import generate_sample
    import numpy as _np
    print("=== syndata sanity check ===")
    for plate_name in ["1_red", "3_blue", "4_blue", "5_red", "1_blank", "5_blank"]:
      syn_img, true_class, true_corners = generate_sample(f"syn:{plate_name}")
      # bypass JIT for diagnostic: call model.backbone + tokenizer step-by-step to find where magnitude explodes
      img_t = Tensor(syn_img, device="NPY").to(Device.DEFAULT).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 3)
      x0_d, x1_d, x2_d, x3_d, sb_d = model.backbone(img_t)
      feat_diag = model.feature_tokenizer(x3_d, sb_d)
      print(f"  {plate_name}:")
      print(f"    x0: mean={x0_d.mean().item():.3f}, std={x0_d.std().item():.3f}, absmax={x0_d.abs().max().item():.3f}")
      print(f"    x1: mean={x1_d.mean().item():.3f}, std={x1_d.std().item():.3f}, absmax={x1_d.abs().max().item():.3f}")
      print(f"    x2: mean={x2_d.mean().item():.3f}, std={x2_d.std().item():.3f}, absmax={x2_d.abs().max().item():.3f}")
      print(f"    x3: mean={x3_d.mean().item():.3f}, std={x3_d.std().item():.3f}, absmax={x3_d.abs().max().item():.3f}")
      print(f"    sb: mean={sb_d.mean().item():.3f}, std={sb_d.std().item():.3f}, absmax={sb_d.abs().max().item():.3f}")
      print(f"    feat: mean={feat_diag.mean().item():.3f}, std={feat_diag.std().item():.3f}, absmax={feat_diag.abs().max().item():.3f}")
      # also still run through the regular path for the class/corners output
      out = infer(Tensor(syn_img, device="NPY"))
      pred_class = int(out[0])
      pred_conf = float(out[1])
      true_pairs = [(true_corners[2*k], true_corners[2*k+1]) for k in range(4)]
      pred_pairs = [(float(out[2 + 2*k]), float(out[3 + 2*k])) for k in range(4)]
      corner_err = float(_np.mean([abs(p[0]-t[0]) + abs(p[1]-t[1]) for p, t in zip(pred_pairs, true_pairs)]))
      class_ok = "OK" if pred_class == true_class else "WRONG"
      print(f"    cls true={true_class} pred={pred_class} conf={pred_conf:.3f} [{class_ok}]")
      print(f"    corners pred: {[(round(x, 3), round(y, 3)) for x, y in pred_pairs]}")
      print(f"    mean per-coord L1 err: {corner_err:.4f}")
      infer.reset()
    print("=== end syndata sanity check ===")

  preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  i = 0
  while i < len(preprocessed_train_files):
    GlobalCounters.reset()

    file = preprocessed_train_files[i]
    img = cv2.imread(file)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # predict — output is [class_id, confidence, c1x,c1y,...,c4x,c4y]
    model_out = infer(Tensor(img, device="NPY"))
    class_id = int(model_out[0])
    confidence = model_out[1]
    corners = [(model_out[2 + 2*k], model_out[3 + 2*k]) for k in range(4)]  # 4 (x,y) pairs in [0,1]

    # diagnostic: dump raw class softmax to see the actual distribution
    if getenv("DIAG_CLS", 0):
      import numpy as _np
      frames_diag = Tensor.stack(*list(infer.frame_buffer), dim=0).unsqueeze(0).to(Device.DEFAULT)
      feat_diag = model.encode(frames_diag)
      class_logits_diag, _, _ = model.decoder(feat_diag)
      probs_diag = class_logits_diag.softmax(-1).to("CPU").numpy()[0]
      top = sorted(enumerate(probs_diag.tolist()), key=lambda kv: -kv[1])[:5]
      ent = float(-(probs_diag * _np.log(probs_diag + 1e-12)).sum())
      print(f"  file={file.rsplit('/',1)[-1]}")
      print(f"    top-5: " + ", ".join(f"cls{k}={v:.3f}" for k, v in top))
      print(f"    argmax={int(probs_diag.argmax())}, entropy={ent:.3f} (uniform={_np.log(17):.3f})")
      print(f"    full dist: " + ", ".join(f"{p:.3f}" for p in probs_diag.tolist()))

    detected, color_id, number = CLASS_DECODE_TABLE[class_id]
    color_names = {0: "none", 1: "red", 2: "blue"}
    color_name = color_names.get(color_id, "blank")

    cv2.putText(img, f"class={class_id} ({confidence:.3f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, f"det={detected} color={color_name} num={number}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # draw predicted corners (TL/TR/BR/BL) with edges connecting them as a quad
    px_corners = [(int(cx * IMG_W), int(cy * IMG_H)) for cx, cy in corners]
    quad_order = [0, 1, 3, 2, 0]  # TL → TR → BR → BL → TL  (corners order is TL/TR/BL/BR)
    for k in range(len(quad_order) - 1):
      cv2.line(img, px_corners[quad_order[k]], px_corners[quad_order[k + 1]], (0, 255, 255), 1)
    for ci, (px, py) in enumerate(px_corners):
      cv2.circle(img, (px, py), 4, (0, 255, 255), -1)
      cv2.putText(img, str(ci), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)
    if key == ord("q"): break
    elif key == ord("a"):
      i -= 1
      infer.reset()
    elif key == ord("f"):
      i += 100
      infer.reset()
    else: i += 1

  cv2.destroyAllWindows()
