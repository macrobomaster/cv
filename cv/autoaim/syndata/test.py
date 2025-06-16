import cv2
import numpy as np

from . import generate_sample

if __name__ == "__main__":
  while True:
    imgs, detecteds, keypointss, colors, numbers = generate_sample(1)
    for i in range(imgs.shape[0]):
      img, detected, keypoints, color, number = imgs[i], detecteds[i], keypointss[i], colors[i], numbers[i]
      img = np.ascontiguousarray(img.permute(1, 2, 0).numpy())
      detected = detected.item()
      keypoints = keypoints.tolist()
      color = color.item()
      number = number.item()

      for keypoint in keypoints:
        x, y = keypoint
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
      cv2.putText(img, f"Detected: {detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      cv2.putText(img, f"{number}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      cv2.putText(img, f"{color}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

      # convert from rgb to bgr
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imshow("img", img)

      key = cv2.waitKey(0)
    if key == ord("q"): break
