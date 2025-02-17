import glob, random

from tinygrad.helpers import tqdm
import cv2
import pyvips
import albumentations as A

from ...common import BASE_PATH
from ...common.image import alpha_overlay

PLATE_PIPELINE = None
PLATE2_PIPELINE = None
BACKGROUND_PIPELINE = None
plate_images = {}
background_images = []
def generate_sample(file) -> tuple[cv2.Mat, float, float]:
  global PLATE_PIPELINE, PLATE2_PIPELINE, BACKGROUND_PIPELINE, plate_images, background_images
  if PLATE_PIPELINE is None:
    PLATE_PIPELINE = A.Compose([
      A.RandomScale(scale_limit=(0.01-1, 0.5-1), p=1),
      A.Perspective(scale=(0.05, 0.15), keep_size=True, fit_output=True, p=1),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    PLATE2_PIPELINE = A.Compose([
      A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
      A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
      A.Defocus(radius=(1, 3), p=0.5),
      A.MotionBlur(blur_limit=(3, 5), p=0.5),
    ])
  if BACKGROUND_PIPELINE is None:
    BACKGROUND_PIPELINE = A.Compose([
      A.RandomResizedCrop(size=(256, 512), scale=(0.1, 1.0), ratio=(1.9, 2.1), p=1),
    ])

  # preload plate image
  plate = int(file.split(":")[1])
  if plate not in plate_images:
    tqdm.write(f"loading plate {plate}")
    plate_img = pyvips.Image.new_from_file(str(BASE_PATH / "armor_plate" / f"{plate}_front.png"), access="sequential").numpy()
    # resize so that max side length is 512
    if plate_img.shape[0] > plate_img.shape[1]:
      plate_img = cv2.resize(plate_img, (int(512 * plate_img.shape[1] / plate_img.shape[0]), 512))
    else:
      plate_img = cv2.resize(plate_img, (512, int(512 * plate_img.shape[0] / plate_img.shape[1])))
    plate_images[plate] = plate_img

  # load background images
  if len(background_images) == 0:
    bg_files = glob.glob(str(BASE_PATH / "background" / "*"))
    tqdm.write(f"loading {len(bg_files)} background images")
    background_images = [pyvips.Image.new_from_file(f, access="sequential").numpy() for f in bg_files]

  raw_plate = plate_images[plate]
  # select a random background image
  raw_background = random.choice(background_images)

  plate_out = PLATE_PIPELINE(image=raw_plate, keypoints=[(raw_plate.shape[1]//2, raw_plate.shape[0]//2)])
  plate = plate_out["image"]
  # keep alpha channel out
  plate_alpha = plate[..., 3]
  plate = PLATE2_PIPELINE(image=plate[:, :, :3])["image"]
  # put alpha channel back
  plate = cv2.merge([plate, plate_alpha])
  img = BACKGROUND_PIPELINE(image=raw_background)["image"]

  # put the plate on the background with alpha blending
  x = random.randint(0, img.shape[1] - plate.shape[1])
  y = random.randint(0, img.shape[0] - plate.shape[0])
  alpha_overlay(plate, img, x, y)

  return img, x + plate_out["keypoints"][0][0], y + plate_out["keypoints"][0][1]
