from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.state import load_state_dict, get_state_dict, safe_load, safe_save

from ..common.tensor import norm

class VGG16Loss:
  def __init__(self):
    self.scaling_layer = ScalingLayer()
    self.net = VGG16()
    self.layer_weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]

  def __call__(self, x:Tensor, y:Tensor, with_gram:bool=True) -> Tensor | tuple[Tensor, Tensor]:
    x = x * 2 - 1
    y = y * 2 - 1

    x, y = self.scaling_layer(x), self.scaling_layer(y)

    features_x, features_y = self.net(x), self.net(y)
    diffs = []
    for fx, fy in zip(features_x, features_y):
      diffs.append((fx - fy).abs())

    loss = self.layer_weights[0] * diffs[0].mean((1, 2, 3))
    for w, d in zip(self.layer_weights[1:], diffs[1:]):
      loss = loss + w * d.mean((1, 2, 3))

    if with_gram:
      gram_diffs = []
      for fx, fy in zip(features_x, features_y):
        gram_diff = self._gram_matrix(fx) - self._gram_matrix(fy)
        gram_diffs.append(gram_diff.square())

      gram_loss = self.layer_weights[0] * gram_diffs[0].mean((1, 2))
      for w, d in zip(self.layer_weights[1:], gram_diffs[1:]):
        gram_loss = gram_loss + w * d.mean((1, 2))

      return loss, gram_loss
    else:
      return loss

  def _gram_matrix(self, x):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h * w)
    x = x @ x.transpose(1, 2)
    return x / (h * w)

  def load_from_pretrained(self):
    self.net.load_from_pretrained()

class ScalingLayer:
  def __init__(self):
    self.shift = Tensor([-0.030, -0.088, -0.188], requires_grad=False)[None, :, None, None]
    self.scale = Tensor([0.458, 0.448, 0.450], requires_grad=False)[None, :, None, None]

  def __call__(self, x:Tensor) -> Tensor:
    return (x - self.shift) / self.scale

class VGG16:
  def __init__(self):
    self.features = [
      nn.Conv2d(3, 64, 3, 1, 1), # 0
      lambda x: x.relu(),
      nn.Conv2d(64, 64, 3, 1, 1),
      lambda x: x.relu(),

      lambda x: x.max_pool2d(2, 2), # 4
      nn.Conv2d(64, 128, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(128, 128, 3, 1, 1),
      lambda x: x.relu(),

      lambda x: x.max_pool2d(2, 2), # 9
      nn.Conv2d(128, 256, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(256, 256, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(256, 256, 3, 1, 1),
      lambda x: x.relu(),

      lambda x: x.max_pool2d(2, 2), # 16
      nn.Conv2d(256, 512, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(512, 512, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(512, 512, 3, 1, 1),
      lambda x: x.relu(),

      lambda x: x.max_pool2d(2, 2), # 23
      nn.Conv2d(512, 512, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(512, 512, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(512, 512, 3, 1, 1),
      lambda x: x.relu(),
    ]

  def __call__(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    for i, layer in enumerate(self.features):
      x = layer(x)
      if i == 3: h_relu1_2 = x
      if i == 8: h_relu2_2 = x
      if i == 15: h_relu3_3 = x
      if i == 22: h_relu4_3 = x
      if i == 29: h_relu5_3 = x

    return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3

  def load_from_pretrained(self):
    state_dict = safe_load(Path(__file__).parent.parent.parent / "weights/vgg16.safetensors")
    load_state_dict(self, state_dict)

if __name__ == "__main__":
  vgg = VGG16()
  import torchvision
  weights = torchvision.models.vgg16(pretrained=True).state_dict()
  print(weights.keys())
  for k,v in get_state_dict(vgg).items():
    v.assign(weights[k].numpy()).realize()
  safe_save(get_state_dict(vgg), (Path(__file__).parent.parent.parent / "weights/vgg16.safetensors").as_posix())

  lpips = VGG16Loss()
  lpips.load_from_pretrained()
