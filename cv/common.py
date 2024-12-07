from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import getenv

import cv2
import numpy as np

import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis

BASE_PATH = Path(getenv("BASE_PATH", "./base/"))

def bgr_to_yuv420(img):
  height, width = img.shape[:2]
  img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)

  # extract U and V channels
  u = img[height:height+height//4].reshape(-1, width//2)
  v = img[height+height//4:height+height//2].reshape(-1, width//2)

  # seperate the Y channel into 4 channels
  y0 = img[:height:2, :width:2]
  y1 = img[1:height:2, :width:2]
  y2 = img[:height:2, 1:width:2]
  y3 = img[1:height:2, 1:width:2]

  # stack the channels
  return np.stack([y0, y1, y2, y3, u, v], axis=-1)

def setup_aravis():
  Aravis.update_device_list()
  devices = Aravis.get_n_devices()
  if devices == 0:
    print("No camera found")
    exit()

  print(f"using {Aravis.get_device_id(0)}")
  cam = Aravis.Camera.new(Aravis.get_device_id(0))
  dev = cam.get_device()
  dev.set_string_feature_value("UserSetSelector", "Default")
  dev.execute_command("UserSetLoad")
  cam.set_pixel_format_from_string("BayerRG8")
  # camera is 1440x1080 or 4:3 crop to be 2:1
  cam.set_region(0, 0, 1440, 720)
  cam.set_exposure_time(10000)
  cam.set_gain(16)
  cam.set_binning(2, 2)
  cam.set_frame_rate(90)
  dev.set_string_feature_value("AcquisitionMode", "Continuous")
  cam.set_trigger("Software")

  strm = cam.create_stream()
  cam.start_acquisition()

  cam.software_trigger()
  payload = cam.get_payload()
  strm.push_buffer(Aravis.Buffer.new_allocate(payload))

  return cam, strm

def get_aravis_frame(cam, strm):
  cam.software_trigger()
  buf = strm.pop_buffer()
  img_data = buf.get_data()
  img_bayer = np.frombuffer(img_data, dtype=np.uint8).reshape(cam.get_region()[3], cam.get_region()[2], 1)
  img = cv2.cvtColor(img_bayer, cv2.COLOR_BAYER_BG2BGR_VNG)
  strm.push_buffer(buf)
  return img

class CLAMB(Optimizer):
  def __init__(self, params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-5, weight_decay=0.0, adam=False):
    super().__init__(params, lr)
    self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, weight_decay, adam
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous() for _ in [b1, b2])
    self.m = [Tensor.zeros(*t.shape, dtype=dtypes.float32, device=t.device, requires_grad=False).contiguous() for t in self.params]
    self.v = [Tensor.zeros(*t.shape, dtype=dtypes.float32, device=t.device, requires_grad=False).contiguous() for t in self.params]

  def _step(self) -> list[Tensor]:
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, t in enumerate(self.params):
      assert t.grad is not None
      self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * t.grad)
      self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (t.grad * t.grad))
      m_hat = self.m[i] / (1.0 - self.b1_t)
      v_hat = self.v[i] / (1.0 - self.b2_t)
      cmask = (m_hat * t.grad > 0).cast(t.dtype)
      cmask = cmask * (cmask.numel() / (cmask.sum() + 1))
      up = ((m_hat * cmask) / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      t.assign((t.detach() - self.lr * r * up).cast(t.dtype))
    return [self.b1_t, self.b2_t] + self.m + self.v
