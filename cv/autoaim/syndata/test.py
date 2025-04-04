import random

import albumentations as A
import cv2

from . import generate_sample

if __name__ == "__main__":
  while True:
    num = random.choice([2, 3, 4, 5, 6])
    color = random.choice(["red", "blue"])
    img, detected, keypoints, color, number = generate_sample(f"fake:{num}_{color}")

    for keypoint in keypoints:
      x, y = keypoint
      cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.putText(img, f"{num}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f"{color}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # convert from rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)
    if key == ord("q"): break
