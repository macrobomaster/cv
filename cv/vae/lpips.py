from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.state import load_state_dict, get_state_dict, safe_load, safe_save, torch_load

from ..common.tensor import norm

class VGG16Loss:
  def __init__(self):
    self.scaling_layer = ScalingLayer()
    self.net = VGG16()
    channels = [64, 128, 256, 512, 512]
    self.lin0 = NetLinLayer(channels[0], 1)
    self.lin1 = NetLinLayer(channels[1], 1)
    self.lin2 = NetLinLayer(channels[2], 1)
    self.lin3 = NetLinLayer(channels[3], 1)
    self.lin4 = NetLinLayer(channels[4], 1)

  def __call__(self, x:Tensor, y:Tensor) -> Tensor:
    x = x * 2 - 1
    y = y * 2 - 1

    x, y = self.scaling_layer(x), self.scaling_layer(y)

    features_x, features_y = self.net(x), self.net(y)
    diffs = []
    for fx, fy in zip(features_x, features_y):
      fx, fy = fx.div(norm(fx, axis=1, keepdim=True).add(1e-10)), fy.div(norm(fy, axis=1, keepdim=True).add(1e-10))
      diffs.append((fx - fy).square())

    lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
    loss = lins[0](diffs[0]).mean((2, 3), keepdim=True)
    for lin, d in zip(lins[1:], diffs[1:]):
      loss = loss + lin(d).mean((2, 3), keepdim=True)

    return loss

  def load_from_pretrained(self):
    state_dict = torch_load(Tensor.from_url("https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"))
    load_state_dict(self, state_dict, strict=False)
    self.net.load_from_pretrained()

class ScalingLayer:
  def __init__(self):
    self.shift = Tensor([-0.030, -0.088, -0.188], requires_grad=False)[None, :, None, None]
    self.scale = Tensor([0.458, 0.448, 0.450], requires_grad=False)[None, :, None, None]

  def __call__(self, x:Tensor) -> Tensor:
    return (x - self.shift) / self.scale

class NetLinLayer:
  def __init__(self, cin:int, cout:int):
    self.model = [lambda x: x, nn.Conv2d(cin, cout, 1, 1, 0, bias=False)]

  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential(self.model)

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
