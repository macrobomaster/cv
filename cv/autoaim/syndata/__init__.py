from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes

from ...common import BASE_PATH

W, H = 512, 256

def random_crop(img: Tensor, size:tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
  bs, c, h, w = img.shape
  y = Tensor.randint(bs, high=h - size[0])
  y_arange = Tensor.arange(size[0]).reshape(1, 1, size[0], 1).expand(bs, c, size[0], w) + y.reshape(bs, 1, 1, 1)
  x = Tensor.randint(bs, high=w - size[1])
  x_arange = Tensor.arange(size[1]).reshape(1, 1, 1, size[1]).expand(bs, c, size[0], size[1]) + x.reshape(bs, 1, 1, 1)
  return img.gather(2, y_arange).gather(3, x_arange), x, y

plates = None
keypoints = None
backgrounds = None
def generate_sample(bs:int=1) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
  global plates, backgrounds, keypoints
  if plates is None:
    safe = safe_load(str(BASE_PATH / "intermediate" / "plates.safetensors"))
    plates = safe["plates"].to(Device.DEFAULT)
    keypoints = safe["keypoints"].to(Device.DEFAULT)
  assert keypoints is not None
  if backgrounds is None:
    backgrounds = safe_load(str(BASE_PATH / "intermediate" / "backgrounds.safetensors"))["backgrounds"].to(Device.DEFAULT)

  background_index = Tensor.randint(bs, high=backgrounds.shape[0])
  background_instance_index = Tensor.randint(bs, high=backgrounds.shape[1])
  background_tensor = backgrounds[background_index, background_instance_index]

  plate_index = Tensor.randint(bs, high=plates.shape[0])
  plate_instance_index = Tensor.randint(bs, high=plates.shape[1])
  plate_tensor = plates[plate_index, plate_instance_index]
  plate_keypoints_tensor = keypoints[plate_index, plate_instance_index]

  bg_h, bg_w = background_tensor.shape[-2:]
  plate_h, plate_w = plate_tensor.shape[-2:]

  # random transform of the plate
  plate_tensor = plate_tensor.pad((None, None, (bg_h - plate_h//2, bg_h - plate_h//2), (bg_w - plate_w//2, bg_w - plate_w//2)))
  plate_tensor, x, y = random_crop(plate_tensor, (H, W))
  x, y = x.unsqueeze(1), y.unsqueeze(1)
  x = (bg_w - plate_w//2) - x
  y = (bg_h - plate_h//2) - y
  plate_keypoints_tensor = plate_keypoints_tensor + Tensor.stack(x, y, dim=-1)

  plate_img, plate_alpha = plate_tensor.split([3, 1], dim=1)
  plate_alpha = plate_alpha / 255.0
  plate_img = plate_img.float()

  # random brightness contrast
  contrast = Tensor.uniform(bs, 1, 1, 1, low=-0.2, high=0.2) + 1
  brightness = Tensor.uniform(bs, 1, 1, 1, low=-0.2, high=0.2) * 255
  plate_img = (Tensor.rand(bs, 1, 1, 1) < 0.5).where((plate_img * contrast + brightness).clip(0, 255), plate_img)

  # random crop of background
  background_tensor = background_tensor.pad((None, None, (bg_h//2, bg_h//2), (bg_w//2, bg_w//2)), mode="reflect")
  background_tensor, _, _ = random_crop(background_tensor, (H, W))

  # alpha blend plate onto background
  img = (plate_img * plate_alpha + background_tensor.float() * (1 - plate_alpha)).cast(background_tensor.dtype)

  # detected
  detected = Tensor.rand(bs) > 0.2
  img = detected.reshape(bs, 1, 1, 1).where(img, background_tensor)

  color = (plate_index < 3).where(3, (plate_index < 8).where(1, 2))
  number = (plate_index < 3).where(plate_index + 3, (plate_index < 8).where(plate_index - 1, plate_index - 6))

  return img, detected, plate_keypoints_tensor, color, number
