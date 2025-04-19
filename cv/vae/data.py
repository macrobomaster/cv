import glob

from tinygrad.helpers import getenv
import cv2
import numpy as np
import albumentations as A

from ..common.dataloader import DataloaderProc
from ..common import BASE_PATH

OUTPUT_PIPELINE = A.Compose([
  A.RandomCrop(256, 512, pad_if_needed=True, p=1),
  A.HorizontalFlip(p=0.5),
  A.VerticalFlip(p=0.5),
  A.Perspective(p=0.25),
  A.Affine(translate_percent=(-0.2, 0.2), scale=(0.9, 1.1), rotate=(-45, 45), shear=(-5, 5), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
  A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
  A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-20, 20), val_shift_limit=0, p=0.5),
  A.OneOf([
    A.PlanckianJitter(mode="cied"),
    A.PlanckianJitter(),
  ], p=0.5),
])
def load_single_file(file) -> dict[str, bytes]:
  if file.startswith("fake:"):
    img = np.zeros((256, 512, 3), dtype=np.uint8)
  elif file.startswith("path:"):
    img = cv2.imread(file[5:])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  else:
    raise ValueError("unknown file type")

  # img = OUTPUT_PIPELINE(image=img)["image"]

  return {
    "x": img.tobytes(),
  }

def get_train_files():
  real_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  real_files = [f"path:{f}" for f in real_files]

  fake_files = [
    "fake:"
  ] * len(real_files)

  if getenv("FAKEFILES", 0):
    return fake_files
  return real_files

def run():
  cv2.setNumThreads(0)
  cv2.ocl.setUseOpenCL(False)

  DataloaderProc(load_single_file).start()

if __name__ == "__main__":
  files = get_train_files()
  for file in files:
    data = load_single_file(file)
    img = np.frombuffer(data["x"], dtype=np.uint8)
    img = img.reshape((256, 512, 3))
    cv2.imshow("img", img)
    key = cv2.waitKey(0)
    if key == ord("q"):
      break
