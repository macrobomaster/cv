from functools import partial
from dataclasses import dataclass
from pathlib import Path
import csv

from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device
from tinygrad.helpers import tqdm

from ..common import BASE_PATH
from ..common.image import rgb_to_yuv420_tensor

MODEL_VERSION = 5

@partial(TinyJit, prune=True)
def pred(model, img):
  img = img.to(Device.DEFAULT)
  if img.ndim == 3: img = img.unsqueeze(0)
  # if img.shape[3] == 3:
  #   yuv = rgb_to_yuv420_tensor(img)
  # else:
  #   yuv = img
  return model(img).to("CPU")

@dataclass
class Annotation:
  detected: int
  x: float
  y: float

annotations_csv = {}
def get_annotation(img_file) -> Annotation:
  global annotations_csv

  # default
  detected, x, y = 0, 0.0, 0.0

  # if there is a img_file.txt file, read that
  if Path(img_file).with_suffix(".txt").exists():
    with open(Path(img_file).with_suffix(".txt"), "r") as f:
      line = f.readline().strip()
      line = line.split(" ")
      detected, x, y = int(line[0]), float(line[1]), float(line[2])
  else:
    basename = ".".join(Path(img_file).name.split(".")[:-2])
    if basename not in annotations_csv:
      with open(BASE_PATH / "data" / basename / f"{basename}.csv", "r") as f:
        tqdm.write(f"reading annotation file {BASE_PATH / 'data' / basename / f'{basename}.csv'}")
        # read the annotation file
        reader = csv.reader(f)
        # skip the header
        _ = next(reader)
        annotations_csv[basename] = [(int(row[0]), int(row[1]), float(row[2]), float(row[3])) for row in reader]

    # check that frame index matches
    frame_index = int(Path(img_file).name.split(".")[-2]) - 1
    assert frame_index == annotations_csv[basename][frame_index][0]

    detected, x, y = annotations_csv[basename][frame_index][1:]

  # return annotation
  return Annotation(detected, x, y)
