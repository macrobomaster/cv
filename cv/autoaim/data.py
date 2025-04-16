import glob

from tinygrad.dtype import _to_np_dtype, dtypes
from tinygrad.helpers import getenv
import albumentations as A
import cv2
import numpy as np

from .syndata import generate_sample
from ..common.dataloader import DataloaderProc
from ..common import BASE_PATH

OUTPUT_PIPELINE = A.Compose([
  A.Perspective(p=0.25),
  A.Affine(translate_percent=(-0.2, 0.2), scale=(0.9, 1.1), rotate=(-45, 45), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
  A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
  A.OneOf([
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.3),
  ], p=0.2),
  A.OneOf([
    A.Defocus(radius=(1, 5), p=0.1),
    A.MotionBlur(blur_limit=(3, 7), p=0.5),
  ], p=0.25),
  A.OneOf([
    A.GaussNoise(std_range=(0.05, 0.2), p=0.5),
    A.ISONoise(p=0.5),
  ], p=0.25),
  A.PlanckianJitter(p=0.5),
  A.Downscale(scale_range=(0.5, 0.75), interpolation_pair={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_LINEAR}, p=0.1),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
DEFAULT_NP_DTYPE = _to_np_dtype(dtypes.default_float)

def load_single_file(file):
  if file.startswith("fake:"):
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    detected = 0
    keypoints = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    color = 0
    number = 0
  elif file.startswith("syn:"):
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

  # scale keypoints to -1-1 range
  xc = xc / img.shape[1] * 2 - 1
  yc = yc / img.shape[0] * 2 - 1
  xtl = xtl / img.shape[1] * 2 - 1
  ytl = ytl / img.shape[0] * 2 - 1
  xtr = xtr / img.shape[1] * 2 - 1
  ytr = ytr / img.shape[0] * 2 - 1
  xbl = xbl / img.shape[1] * 2 - 1
  ybl = ybl / img.shape[0] * 2 - 1
  xbr = xbr / img.shape[1] * 2 - 1
  ybr = ybr / img.shape[0] * 2 - 1

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
    "y": np.array((color, number, xc, yc, xtl, ytl, xtr, ytr, xbl, ybl, xbr, ybr), dtype=DEFAULT_NP_DTYPE).tobytes(),
  }

def get_train_files():
  real_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  real_files = [f"path:{f}" for f in real_files]

  syn_files = [
    "syn:3_blank",
    "syn:4_blank",
    "syn:5_blank",

    "syn:2_red",
    "syn:3_red",
    "syn:4_red",
    "syn:5_red",
    "syn:6_red",

    "syn:2_blue",
    "syn:3_blue",
    "syn:4_blue",
    "syn:5_blue",
    "syn:6_blue",
  ] * len(real_files)

  fake_files = [
    "fake:"
  ] * len(real_files)

  if getenv("FAKEFILES", 0):
    return fake_files
  if getenv("FINETUNE", 0):
    return real_files
  return syn_files

def run():
  cv2.setNumThreads(0)
  cv2.ocl.setUseOpenCL(False)

  DataloaderProc(load_single_file).start()
