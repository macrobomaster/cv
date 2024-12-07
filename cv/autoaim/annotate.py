import os

import cv2
import glob
from pathlib import Path

from ..common import BASE_PATH

if __name__ == "__main__":
  frame_files = glob.glob(str(BASE_PATH / "data" / "*.png"))
  print(f"there are {len(frame_files)} frames")

  oframe, frame = None, None
  click_pos = None

  def mouse_handler(event, x, y, flags, param):
    global oframe, frame, click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
      click_pos = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
      frame = oframe.copy()  # type: ignore
      cv2.line(frame, (x - 50, y), (x + 50, y), (0, 0, 255), 1)
      cv2.line(frame, (x, y - 50), (x, y + 50), (0, 0, 255), 1)

  cv2.namedWindow("preview")
  cv2.setMouseCallback("preview", mouse_handler)

  i = 0
  flag = False
  back = False
  while not flag and i < len(frame_files):
    # check if the file has already been annotated
    if Path(frame_files[i]).with_suffix(".txt").exists() and not back:
      i += 1
      continue
    back = False

    print(f"annotating frame {i} of {len(frame_files)}")
    frame_file = frame_files[i]
    oframe = frame = cv2.imread(frame_file)

    while click_pos is None:
      cv2.imshow("preview", frame)

      key = cv2.waitKey(1)
      if key == ord("q"):
        flag = True
        break
      elif key == ord("s"):
        with open(Path(frame_file).with_suffix(".txt"), "w") as f:
          f.write(f"0 0 0")
        break
      elif key == ord("a"):
        i -= 2
        back = True
        break
    if click_pos is not None:
      with open(Path(frame_file).with_suffix(".txt"), "w") as f:
        f.write(f"1 {click_pos[0]} {click_pos[1]}")  # type: ignore
      click_pos = None
    i += 1
