import glob

from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
import cv2

from .syndata import generate_sample
from ..common import BASE_PATH

def load_batch(bs: int) -> tuple[Tensor, Tensor]:
  if getenv("FAKEFILES"):
    return Tensor.zeros(bs, 256, 512, 3, dtype=dtypes.uint8).contiguous(), Tensor.zeros(bs, 18).contiguous()

  img, detected, keypoints, color, number = generate_sample(bs)
  img = img.cast(dtypes.float32)

  # random brightness contrast
  contrast = Tensor.uniform(bs, 1, 1, 1, low=-0.2, high=0.2) + 1
  brightness = Tensor.uniform(bs, 1, 1, 1, low=-0.2, high=0.2) * 255
  img = (Tensor.rand(bs, 1, 1, 1) < 0.5).where((img * contrast + brightness).clip(0, 255), img)

  # random saturation
  saturation = Tensor.uniform(bs, 1, 1, 1, low=-0.2, high=0.2) + 1
  luminance = (0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]).unsqueeze(1)
  img = (Tensor.rand(bs, 1, 1, 1) < 0.5).where(luminance.lerp(img, saturation), img)

  # random gaussian noise
  noise = Tensor.randn(bs, 3, 256, 512) * Tensor.uniform(bs, 1, 1, 1, low=0.05, high=0.2)
  img = (Tensor.rand(bs, 1, 1, 1) < 0.5).where(img + noise, img)

  # no plate detected when plate cetner is outside of frame bounds
  center_outside = (keypoints[:, 0, 0] < 0) | (keypoints[:, 0, 0] > img.shape[3]) | (keypoints[:, 0, 1] < 0) | (keypoints[:, 0, 1] > img.shape[2])
  detected = detected & ~center_outside

  # scale keypoints to (-1, 1) range
  keypoints = keypoints / Tensor.cat(Tensor(img.shape[3]).reshape(1, 1, 1), Tensor(img.shape[2]).reshape(1, 1, 1), dim=-1)
  keypoints = keypoints * 2 - 1

  # numbers start from 2 but model starts from 1
  number = (number > 0).where(number - 1, 0)

  # gate number based on detection
  number = detected.where(number, 0)

  # gate color based on detection
  color = detected.where(color, 0)

  # gate keypoints based on detection
  keypoints = detected.reshape(bs, 1, 1).where(keypoints, 0)

  # loss gates
  has_det = Tensor(True).unsqueeze(0).expand(bs)
  has_color = Tensor(True).unsqueeze(0).expand(bs)
  has_number = Tensor(True).unsqueeze(0).expand(bs)
  has_center = detected
  has_plate = detected

  labels = Tensor.stack(detected, color, number, dim=1)
  keypoints = keypoints.flatten(1)
  labels = labels.cat(keypoints, dim=1)
  has_gates = Tensor.stack(has_det, has_color, has_number, has_center, has_plate, dim=1)
  labels = labels.cat(has_gates, dim=1)

  return img.permute(0, 2, 3, 1).cast(dtypes.uint8), labels

PLATES = [
  "3_blank",
  "4_blank",
  "5_blank",

  "2_red",
  "3_red",
  "4_red",
  "5_red",
  "6_red",

  "2_blue",
  "3_blue",
  "4_blue",
  "5_blue",
  "6_blue",
]

def get_train_files():
  real_files = glob.glob(str(BASE_PATH / "data" / "**" / "*.png"), recursive=True)
  real_files = [f"path:{f}" for f in real_files]

  syn_files = [f"syn:{plate}" for plate in PLATES] * len(real_files)

  fake_files = [
    "fake:"
  ] * 1024

  if getenv("FAKEFILES", 0):
    return fake_files
  if getenv("REALFILES", 0):
    return real_files
  if getenv("SYNFILES", 0):
    return syn_files
  return syn_files + real_files

if __name__ == "__main__":
  cv2.setNumThreads(0)
  cv2.ocl.setUseOpenCL(False)

  x, y = load_batch(8)
  for i in range(x.shape[0]):
    img = x[i].numpy()
    anno = y[i].numpy()
    print(anno)
    cv2.circle(img, (int(((anno[3] + 1) / 2) * 512), int(((anno[4] + 1) / 2) * 256)), 5, (0, 255, 0), -1)
    cv2.imshow("img", img)
    key = cv2.waitKey(0)
    if key == ord("q"):
      break
