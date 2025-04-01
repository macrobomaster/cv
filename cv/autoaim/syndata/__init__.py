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
def generate_sample(file) -> tuple[cv2.Mat, list[tuple[float, float]], int, int]:
  global PLATE_PIPELINE, PLATE2_PIPELINE, BACKGROUND_PIPELINE, plate_images, background_images
  if PLATE_PIPELINE is None:
    PLATE_PIPELINE = A.Compose([
      A.RandomScale(scale_limit=(0.05-1, 0.5-1), p=1),
      A.Perspective(scale=(0.05, 0.15), keep_size=True, fit_output=True, p=1),
      A.SafeRotate(limit=(-90, 90), p=0.5),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
  if BACKGROUND_PIPELINE is None:
    BACKGROUND_PIPELINE = A.Compose([
      A.RandomResizedCrop(size=(256, 512), scale=(0.1, 1.0), ratio=(1.9, 2.1), p=1),
    ])

  # preload plate image
  plate = file.split(":")[1]
  number = int(plate.split("_")[0])
  color = plate.split("_")[1]
  if plate not in plate_images:
    tqdm.write(f"loading plate {plate}")
    plate_img = pyvips.Image.new_from_file(str(BASE_PATH / "armor_plate" / f"{plate}.png"), access="sequential").numpy()
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

  # keypoints are center, top-left, top-right, bottom-left, bottom-right
  keypoints = []
  keypoints.append((raw_plate.shape[1]//2, raw_plate.shape[0]//2))
  keypoints.append((0, 0))
  keypoints.append((raw_plate.shape[1], 0))
  keypoints.append((0, raw_plate.shape[0]))
  keypoints.append((raw_plate.shape[1], raw_plate.shape[0]))

  plate_out = PLATE_PIPELINE(image=raw_plate, keypoints=keypoints)
  plate = plate_out["image"]
  img = BACKGROUND_PIPELINE(image=raw_background)["image"]

  # put the plate on the background with alpha blending
  x = random.randint(0, img.shape[1] - plate.shape[1])
  y = random.randint(0, img.shape[0] - plate.shape[0])
  alpha_overlay(plate, img, x, y)

  # turn color into int
  match color:
    case "red": color = 1
    case "blue": color = 2
    case _: color = 3

  center = x + plate_out["keypoints"][0][0], y + plate_out["keypoints"][0][1]
  top_left = x + plate_out["keypoints"][1][0], y + plate_out["keypoints"][1][1]
  top_right = x + plate_out["keypoints"][2][0], y + plate_out["keypoints"][2][1]
  bottom_left = x + plate_out["keypoints"][3][0], y + plate_out["keypoints"][3][1]
  bottom_right = x + plate_out["keypoints"][4][0], y + plate_out["keypoints"][4][1]
  return img, [center, top_left, top_right, bottom_left, bottom_right], color, number
