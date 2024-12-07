import cv2
import glob
import csv
from pathlib import Path

from ..common import BASE_PATH

preprocessed_train_files = glob.glob(str(BASE_PATH / "data" / "*.png"))

# open annotation file
# strip the file extension and frame index
basename = ".".join(Path(preprocessed_train_files[0]).name.split(".")[:-2])
with open(BASE_PATH / "data" / f"{basename}.csv", "r") as f:
  print(f"reading annotation file {BASE_PATH / 'data' / f'{basename}.csv'}")
  # read the annotation file
  reader = csv.reader(f)
  # skip the header
  _ = next(reader)
  annotations = [(int(row[0]), int(row[1]), float(row[2]), float(row[3])) for row in reader]

i = 0
while i < len(preprocessed_train_files):
  file = preprocessed_train_files[i]
  img = cv2.imread(file)

  # get the annotation
  detected, x, y = annotations[i][1:]

  # draw the annotation
  if detected:
    x, y = int(x * img.shape[1]), int((1 - y) * img.shape[0])
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
  cv2.imshow("img", img)

  key = cv2.waitKey(0)
  if key == ord("q"): break
  elif key == ord("a"): i -= 1
  else: i += 1

cv2.destroyAllWindows()
