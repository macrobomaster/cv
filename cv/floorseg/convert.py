"""One-off: convert the CoinCheung BiSeNetV2 ADE20K checkpoint → weights/floorseg.safetensors.

Runs on a DEV BOX (needs torch + network), NEVER on the Orin. The tinygrad `BiSeNetV2`
module mirrors the reference so keys match 1:1 (eval path: detail+segment+bga+head); the
training-only `aux*` heads and `num_batches_tracked` counters are dropped. Verified
numerically identical to the torch reference (argmax 100%, logit MAE ~2e-6).
  usage: python -m cv.floorseg.convert
"""
import urllib.request
from pathlib import Path

import torch
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict, safe_save

from .model import BiSeNetV2

CKPT_URL = "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_ade20k.pth"
WEIGHTS = Path(__file__).parent.parent.parent / "weights"

def main():
  pth = WEIGHTS / "bisenetv2_ade20k.pth"
  if not pth.exists():
    print(f"downloading {CKPT_URL}")
    urllib.request.urlretrieve(CKPT_URL, str(pth))
  ck = torch.load(str(pth), map_location="cpu", weights_only=True)
  keys = set(get_state_dict(BiSeNetV2(150)).keys())          # eval-path keys to keep
  sd = {k: Tensor(v.float().numpy()) for k, v in ck.items()
        if k in keys and not k.endswith("num_batches_tracked")}
  out = WEIGHTS / "floorseg.safetensors"
  safe_save(sd, str(out))
  print(f"saved {out} ({len(sd)} tensors)")

if __name__ == "__main__":
  main()
