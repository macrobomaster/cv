import cv2
import glob

from .common import get_annotation
from ..common import BASE_PATH

preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)

i = 0
while i < len(preprocessed_train_files):
  file = preprocessed_train_files[i]
  img = cv2.imread(file)

  # get the annotation
  anno = get_annotation(file)
  detected, x, y, dist = anno.detected, anno.x, anno.y, anno.dist

  # draw the annotation
  if detected:
    x, y = int(x * img.shape[1]), int((1 - y) * img.shape[0])
    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    cv2.putText(img, f"{dist:.3f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  cv2.imshow("img", img)

  key = cv2.waitKey(0)
  if key == ord("q"): break
  elif key == ord("a"):
    i -= 1
    if i < 0: i = len(preprocessed_train_files) - 1
  elif key == ord("f"):
    i += 100
  else: i += 1

cv2.destroyAllWindows()
