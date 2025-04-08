from tinygrad.dtype import _to_np_dtype, dtypes
import albumentations as A
import cv2
import numpy as np

from .syndata import generate_sample
from ..common.dataloader import DataloaderProc

OUTPUT_PIPELINE = A.Compose([
  A.Perspective(p=0.25),
  A.Affine(translate_percent=(-0.2, 0.2), scale=(0.9, 1.1), rotate=(-45, 45), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
  A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
  A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
  A.Defocus(radius=(1, 5), p=0.1),
  A.MotionBlur(blur_limit=(3, 7), p=0.5),
  A.GaussNoise(std_range=(0.05, 0.2), p=0.25),
  A.PlanckianJitter(p=0.5),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
DEFAULT_NP_DTYPE = _to_np_dtype(dtypes.default_float)

def load_single_file(file):
  if file.startswith("fake:"):
    img, detected, keypoints, color, number = generate_sample(file)
  else:
    raise ValueError("unknown file type")

  output = OUTPUT_PIPELINE(image=img, keypoints=keypoints)
  img = output["image"]
  xc, yc = output["keypoints"][0]
  if detected:
    if xc < 0 or xc > img.shape[1] or yc < 0 or yc > img.shape[0]:
      detected = 0
  xtl, ytl = output["keypoints"][1]
  xtr, ytr = output["keypoints"][2]
  xbl, ybl = output["keypoints"][3]
  xbr, ybr = output["keypoints"][4]

  # numbers start from 2 but model starts from 1
  number -= 1

  # gate number based on detection
  if not detected:
    number = 0

  # gate color based on detection
  if not detected:
    color = 0

  # set all keypoints to 0 if not detected
  if not detected:
    xc = yc = xtl = ytl = xtr = ytr = xbl = ybl = xbr = ybr = 0

  return {
    "x": img.tobytes(),
    "y": np.array((color, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr, number), dtype=DEFAULT_NP_DTYPE).tobytes(),
  }

def run():
  cv2.setNumThreads(0)
  cv2.ocl.setUseOpenCL(False)

  DataloaderProc(load_single_file).start()
