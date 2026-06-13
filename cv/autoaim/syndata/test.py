import random

import cv2
import numpy as np

from . import generate_sample, generate_sequence
from ..model import CLASS_DECODE_TABLE

if __name__ == "__main__":
  T = 4
  while True:
    num = random.choice([1, 2, 3, 4, 5, 6])
    color = random.choice(["red", "blue"])
    file = f"syn:{num}_{color}"

    images, class_id, corners_8 = generate_sequence(file, T=T)

    detected, color_id, number = CLASS_DECODE_TABLE[class_id] if class_id < len(CLASS_DECODE_TABLE) else (0, 0, 0)
    # Denormalize corners back to pixel coords for drawing on the final frame
    corner_px = [(int(corners_8[2*i] * 512), int(corners_8[2*i+1] * 256)) for i in range(4)]

    # Display T frames side by side
    row = []
    for t, img in enumerate(images):
      frame = img.copy()
      cv2.putText(frame, f"t={t}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      if t == T - 1:
        cv2.putText(frame, f"class={class_id} det={detected} col={color_id} num={number}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if class_id > 0:
          for i, (px, py) in enumerate(corner_px):
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)
            cv2.putText(frame, str(i), (px+4, py-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      row.append(frame)

    # Stack frames: top row = t0,t1; bottom row = t2,t3
    top = np.concatenate(row[:T//2], axis=1)
    bottom = np.concatenate(row[T//2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)

    cv2.imshow("sequence", grid)
    key = cv2.waitKey(0)
    if key == ord("q"): break
