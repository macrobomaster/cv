from pathlib import Path
import csv

from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit

from ..common import BASE_PATH

@TinyJit
def pred(model, img):
  cl, cl_prob, x, y, dist = model(img)
  return cl, cl_prob, x, y, dist

annotations = {}
def get_annotation(img_file):
  global annotations

  # if there is a img_file.txt file, read that
  if Path(img_file).with_suffix(".txt").exists():
    with open(Path(img_file).with_suffix(".txt"), "r") as f:
      line = f.readline().strip()
      line = line.split(" ")
      if len(line) == 3:
        return int(line[0]), float(line[1]), float(line[2]), 0.0
      elif len(line) == 4:
        return int(line[0]), float(line[1]), float(line[2]), float(line[3])
      else: raise ValueError(f"invalid annotation file {img_file}.txt")
  else:
    basename = ".".join(Path(img_file).name.split(".")[:-2])
    if basename not in annotations:
      with open(BASE_PATH / "data" / f"{basename}.csv", "r") as f:
        print(f"reading annotation file {BASE_PATH / 'data' / f'{basename}.csv'}")
        # read the annotation file
        reader = csv.reader(f)
        # skip the header
        _ = next(reader)
        annotations[basename] = [(int(row[0]), int(row[1]), float(row[2]), float(row[3])) for row in reader]

    # get the frame index
    frame_index = int(Path(img_file).name.split(".")[-2]) - 1
    assert frame_index == annotations[basename][frame_index][0]
    return annotations[basename][frame_index][1:] + (0.0,)
