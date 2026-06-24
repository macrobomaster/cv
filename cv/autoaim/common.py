from functools import partial

import numpy as np
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv

from ..common import BASE_PATH

# input image dims
IMG_H, IMG_W = 384, 768

BACKBONE_STRIDE = 32
X3_H, X3_W = IMG_H // BACKBONE_STRIDE, IMG_W // BACKBONE_STRIDE
N_X3_TOKENS = X3_H * X3_W

X2_STRIDE = 16
X2_H, X2_W = IMG_H // X2_STRIDE, IMG_W // X2_STRIDE
N_X2_TOKENS = X2_H * X2_W

# temporal window
T = getenv("T", 1)

# corner output valid range
BIN_LO, BIN_HI = -0.5, 1.5

MODEL_VERSION = 34

# canonical camera
CANONICAL_FX_FY = 1010
CANONICAL_CX, CANONICAL_CY = IMG_W / 2, IMG_H / 2
CANONICAL_CAMERA_MATRIX = np.array([[CANONICAL_FX_FY, 0, CANONICAL_CX],
                                    [0, CANONICAL_FX_FY, CANONICAL_CY],
                                    [0, 0, 1]], dtype=np.float32)
CANONICAL_DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)

SCREW_DIMS_SMALL = (0.095, 0.10437)
SCREW_DIMS_LARGE = (0.20875, 0.08562)

def plate_screw_dims(number:int) -> tuple[float, float]:
  return SCREW_DIMS_LARGE if number == 1 else SCREW_DIMS_SMALL

def plate_screw_points(number:int) -> np.ndarray:
  w, h = plate_screw_dims(number)
  return np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0], [-w/2, h/2, 0], [w/2, h/2, 0]], dtype=np.float32)

# ballistics model
MUZZLE_VELOCITY = 28.0   # m/s
GRAVITY = 9.81           # m/s^2

# pipeline latencies
DELTA_INPUT = 0.020      # s, uart to control board
DELTA_TRIGGER = 0.060    # s, flywheel delay
GIMBAL_TAU = 0.040       # s, gimbal first-order motor constant.
GIMBAL_OMEGA_MAX = 12.0  # rad/s, gimbal slew rate ceiling.

# Gimbal tracking control (decisiond). aim_error is a RATE/joystick command:
# gimbal_velocity = K_JOYSTICK * aim_error. The controller commands the velocity needed to follow
# the moving aim point: velocity feedforward + PID on the angular position error, then / K_JOYSTICK.
K_JOYSTICK = 10.0        # rad/s per aim_error unit. TODO: calibrate (calib_gimbal: peak_rate/amp).
AIM_KP = 10.0            # 1/s. ≈ K_JOYSTICK reproduces the old proportional behavior at zero feedforward
AIM_KI = 0.0            # leave 0; raise only if a persistent offset remains
AIM_KD = 0.0            # leave 0; raise only if oscillation appears
AIM_I_CLAMP = 2.0       # rad/s, anti-windup ceiling on the integral's velocity contribution
AIM_FF_DT = 0.010       # s, forward finite-difference step for the velocity feedforward
AIM_D_TAU = 0.020       # s, low-pass time constant on the derivative term

# camera frame to gimbal frame
R_MOUNT = np.array([[0,  0,  1],
                    [0, -1,  0],
                    [-1, 0,  0]], dtype=np.float64)
T_MOUNT = np.array([0, 0, 0], dtype=np.float64)

@partial(TinyJit, prune=True)
def pred(model, frame, target_color, frames:Tensor|None=None):
  if frames is None:
    frames = frame.to(Device.DEFAULT).reshape(1, IMG_H, IMG_W, 3)
  else:
    frame = frame.to(Device.DEFAULT).reshape(IMG_H, IMG_W, 3)
    frames.assign(Tensor.cat(frames[1:], frame.unsqueeze(0))).realize()

  target_color = target_color.to(Device.DEFAULT)
  mem, sb = model.encode(frames.unsqueeze(0))
  return model.corner_predict(mem, sb, target_color).float().to("CPU")

class TemporalInference:
  def __init__(self, model_fn, T:int, model=None):
    self.model_fn, self.model = model_fn, model
    self.T = T
    if self.T != 1:
      self.frames = Tensor.zeros(self.T, IMG_H, IMG_W, 3, dtype=dtypes.uint8).clone()
    else:
      self.frames = None

  def __call__(self, img, target_color:int=0) -> list:
    target_color_t = Tensor([target_color], dtype=dtypes.int32, device="PYTHON")
    return self.model_fn(self.model, img, target_color_t, frames=self.frames).tolist()[0]

  def warmup(self):
    for _ in range(3):
      if self.T != 1: fake_frames = Tensor.empty(T, IMG_H, IMG_W, 3, dtype=dtypes.uint8).clone().realize()
      fake_frame = Tensor.empty(IMG_H * IMG_W * 3, dtype=dtypes.uint8, device="PYTHON").clone().realize()
      fake_target = Tensor([0], dtype=dtypes.int32, device="PYTHON").realize()
      self.model_fn(self.model, fake_frame, fake_target, frames=fake_frames if self.T != 1 else None)
    return self.model_fn
