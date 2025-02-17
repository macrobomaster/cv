import random

import albumentations as A
import cv2

from . import generate_sample

if __name__ == "__main__":
  while True:
    num = random.choice([3, 4, 5])
    img, x, y = generate_sample(f"fake:{num}")

    cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    # convert from rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)
    if key == ord("q"): break
