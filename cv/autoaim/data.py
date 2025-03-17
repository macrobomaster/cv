import glob

from tinygrad.helpers import getenv
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import Device
from tinygrad.dtype import dtypes
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import pyvips
import numpy as np
import albumentations as A

from ..common import BASE_PATH
from .common import get_annotation
from .syndata import generate_sample

def get_train_files():
  real_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  real_files = [f"path:{f}" for f in real_files]

  fake_files = [
    "fake:3_blank",
    "fake:4_blank",
    "fake:5_blank",

    "fake:2_red",
    "fake:3_red",
    "fake:4_red",
    "fake:5_red",
    "fake:6_red",

    "fake:2_blue",
    "fake:3_blue",
    "fake:4_blue",
    "fake:5_blue",
    "fake:6_blue",
  ] * len(real_files)

  if getenv("FINETUNE", 0):
    return real_files
  else:
    return fake_files

OUTPUT_PIPELINE = None
def load_single_file(file):
  global OUTPUT_PIPELINE
  if OUTPUT_PIPELINE is None:
    OUTPUT_PIPELINE = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.Perspective(p=0.25),
      A.Affine(translate_percent=(-0.2, 0.2), scale=(0.9, 1.1), rotate=(-45, 45), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
      A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, p=0.5),
      A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
      A.OneOf([
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomToneCurve(p=0.5),
      ], p=0.2),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

  detected, x, y, dist, color, number = 0, 0, 0, -1, -1, -1
  if file.startswith("path:"):
    img = pyvips.Image.new_from_file(file[5:], access="sequential").numpy()
    img = img[..., :3]
    if img.shape[0] != 256 or img.shape[1] != 512:
      img = cv2.resize(img, (512, 256))

    anno = get_annotation(file[5:])
    detected, x, y, dist = anno.detected, anno.x, anno.y, anno.dist

    # transform points
    x, y = x * (img.shape[1] - 1), (1 - y) * (img.shape[0] - 1)
  elif file.startswith("fake:"):
    img, x, y, color, number = generate_sample(file)

  output = OUTPUT_PIPELINE(image=img, keypoints=[(x, y)])
  img = output["image"]
  x, y = output["keypoints"][0]
  if x < 0 or x > img.shape[1] or y < 0 or y > img.shape[0]:
    detected = 0
  else:
    detected = 1

  number -= 1

  return {
    "x": img.tobytes(),
    "y": np.array((detected, x, y, dist, color, number), dtype=_to_np_dtype(dtypes.default_float)).tobytes(),
  }

def single_batch(iter):
  d, c = next(iter)
  return d["x"].to(Device.DEFAULT), d["y"].to(Device.DEFAULT), c

if __name__ == "__main__":
  from tinygrad.helpers import tqdm
  from ..common.dataloader import batch_load, BatchDesc

  BS = 256

  batch_iter = iter(tqdm(batch_load(
    {
      "x": BatchDesc(shape=(256, 512, 3), dtype=dtypes.uint8),
      "y": BatchDesc(shape=(4,), dtype=dtypes.default_float),
    },
    load_single_file, get_train_files, bs=BS, shuffle=True,
  ), total=len(get_train_files())//BS))

  i, proc = 0, single_batch(batch_iter)
  while proc is not None:
    try: next_proc = single_batch(batch_iter)
    except StopIteration: next_proc = None
    proc, next_proc = next_proc, None
