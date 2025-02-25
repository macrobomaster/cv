from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.state import load_state_dict, get_state_dict, safe_save, torch_load

from ..common.tensor import norm

class VGG16Loss:
  def __init__(self):
    self.scaling_layer = ScalingLayer()
    self.net = VGG16()
    channels = [64, 128, 256, 512, 512]
    self.lins = [NetLinLayer(c, 1) for c in channels]

  def __call__(self, x:Tensor, y:Tensor) -> Tensor:
    x = x * 2 - 1
    y = y * 2 - 1

    x = self.scaling_layer(x)
    y = self.scaling_layer(y)

    features_x, features_y = self.net(x), self.net(y)
    diffs = []
    for fx, fy in zip(features_x, features_y):
      fx, fy = fx.div(norm(fx, axis=1, keepdim=True) + 1e-6), fy.div(norm(fy, axis=1, keepdim=True) + 1e-6)
      diffs.append((fx - fy).square())

    loss = self.lins[0](diffs[0]).mean((2, 3))
    for lin, d in zip(self.lins[1:], diffs[1:]):
      loss = loss + lin(d).mean((2, 3))

    return loss

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
      lambda x: x.max_pool2d(2), # 4
      nn.Conv2d(64, 128, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(128, 128, 3, 1, 1),
      lambda x: x.relu(),
      lambda x: x.max_pool2d(2), # 9
      nn.Conv2d(128, 256, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(256, 256, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(256, 256, 3, 1, 1),
      lambda x: x.relu(),
      lambda x: x.max_pool2d(2), # 16
      nn.Conv2d(256, 512, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(512, 512, 3, 1, 1),
      lambda x: x.relu(),
      nn.Conv2d(512, 512, 3, 1, 1),
      lambda x: x.relu(),
      lambda x: x.max_pool2d(2), # 23
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

if __name__ == "__main__":
  lpips = VGG16Loss()

  import torchvision
  weights = torchvision.models.vgg16(pretrained=True).state_dict()
  for k,v in get_state_dict(lpips).items():
    if "net." not in k: continue
    k = k.replace("net.", "")
    v.assign(weights[k].numpy()).realize()

  state_dict = torch_load("weights/vgg.pth")
  state_dict = {k.replace("lin", "lins."): v for k,v in state_dict.items()}
  load_state_dict(lpips, state_dict, strict=False)

  safe_save(get_state_dict(lpips), "weights/lpips.safetensors")
