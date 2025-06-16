import json, glob

from tinygrad.helpers import getenv, tqdm, trange
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_save, get_state_dict
import albumentations as A
import cv2

from ...common import BASE_PATH
from ..data import PLATES

# preprocess plate images
if getenv("PLATE"):
  PLATE_PIPELINE = A.Compose([
    A.SmallestMaxSize(max_size=512, p=1),
    A.CenterCrop(height=512, width=640, pad_if_needed=True, p=1),
    A.SafeRotate(limit=(-90, 90), p=0.5),
    A.Perspective(scale=(0.05, 0.2), keep_size=True, fit_output=True, p=1),
    A.RandomScale(scale_limit=(0.05-1, 0.5-1), p=1),
    A.PadIfNeeded(min_height=256, min_width=256, p=1),
    A.LongestMaxSize(max_size=256, p=1),
    A.PadIfNeeded(min_height=256, min_width=256, p=1),
  ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

  plates_proc, keypoints_proc = [], []
  for plate in tqdm(PLATES):
    plates, keypoints = [], []
    for _ in trange(getenv("ITERS", 100)):
      img = cv2.imread(str(BASE_PATH / "armor_plate" / f"{plate}.png"), cv2.IMREAD_UNCHANGED)
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

      # apply augments
      with open(str(BASE_PATH / "armor_plate" / f"{plate}.json"), "r") as f:
        kp = json.load(f)
      resized = PLATE_PIPELINE(image=img, keypoints=[(img.shape[1]//2, img.shape[0]//2)]+kp)
      img = resized["image"]
      kp = resized["keypoints"]

      plates.append(Tensor(img, device="CPU").permute(2, 0, 1))
      keypoints.append(Tensor(kp, device="CPU", dtype=dtypes.float32))
    plates_proc.append(Tensor.stack(*plates).realize())
    keypoints_proc.append(Tensor.stack(*keypoints).realize())

  safe_save({"plates": Tensor.stack(*plates_proc), "keypoints": Tensor.stack(*keypoints_proc)}, str(BASE_PATH / "intermediate" / "plates.safetensors"))

# preprocess background images
if getenv("BACKGROUND"):
  BACKGROUND_PIPELINE = A.Compose([
    A.RandomResizedCrop(size=(256, 512), scale=(0.1, 1.0), ratio=(1.9, 2.1), p=1),
  ])

  background_images = []
  bg_files = glob.glob(str(BASE_PATH / "background" / "*"))
  for f in tqdm(bg_files):
    backgrounds = []
    # take random crops of the image
    bg_img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    for _ in trange(getenv("ITERS", 100)):
      img = BACKGROUND_PIPELINE(image=bg_img)["image"]
      backgrounds.append(Tensor(img, device="CPU").permute(2, 0, 1))
    background_images.append(Tensor.stack(*backgrounds).realize())

  safe_save({"backgrounds": Tensor.stack(*background_images)}, str(BASE_PATH / "intermediate" / "backgrounds.safetensors"))
