from pathlib import Path
import random

import albumentations as A
import cv2
import numpy as np

from ...common import BASE_PATH
from ...common.image import alpha_overlay

PLATE_PIPELINE = A.Compose([
  A.RandomScale(scale_limit=(0.002-1, 0.02-1), p=1),
  A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
  A.Perspective(scale=(0.05, 0.2), keep_size=True, fit_output=True, p=1),
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
BACKGROUND_PIPELINE = A.Compose([
  A.RandomResizedCrop(size=(256, 512), scale=(0.1, 1.0), ratio=(1.9, 2.1), p=1),
])
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

if __name__ == "__main__":
  raw_plate = cv2.imread(str(BASE_PATH / "armor_plate" / "4_front.png"), cv2.IMREAD_UNCHANGED)
  raw_background = cv2.imread(str(BASE_PATH / "background" / "room.png"))

  while True:
    plate_out = PLATE_PIPELINE(image=raw_plate, keypoints=[(raw_plate.shape[1]//2, raw_plate.shape[0]//2)])
    plate = plate_out["image"]
    background = BACKGROUND_PIPELINE(image=raw_background)["image"]

    # put the plate on the background with alpha blending
    x = random.randint(0, background.shape[1] - plate.shape[1])
    y = random.randint(0, background.shape[0] - plate.shape[0])
    alpha_overlay(plate, background, x, y)

    output = OUTPUT_PIPELINE(image=background, keypoints=[(x + plate_out["keypoints"][0][0], y + plate_out["keypoints"][0][1])])
    img = output["image"]
    x, y = output["keypoints"][0]
    print(x, y)

    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)
    if key == ord("q"): break
