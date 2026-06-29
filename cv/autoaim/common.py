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

MODEL_VERSION = 40

# canonical camera
CANONICAL_FX_FY = 648
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

# pipeline latencies. DELTA_INPUT/TRIGGER are shared (same comms path); TAU/OMEGA_MAX are PER-AXIS
# (different motor + inertia). All from calib_gimbal --axis {yaw,pitch}; feed the lead-settle timing.
DELTA_INPUT = 0.020      # s, uart to control board
DELTA_TRIGGER = 0.060    # s, flywheel delay
GIMBAL_TAU = {"yaw": 0.072, "pitch": 0.072}        # s, velocity-rise const per axis. TODO: pitch
GIMBAL_OMEGA_MAX = {"yaw": 6.78, "pitch": 6.78}    # rad/s, slew ceiling per axis. TODO: pitch

# Gimbal tracking control (decisiond). aim_error is a RATE/joystick command:
# gimbal_velocity = k_joystick * aim_error. The controller commands the velocity to follow the moving
# aim point: velocity feedforward + PID on the angular position error, then / k_joystick.
# PER-AXIS — yaw and pitch differ (pitch carries gravity torque + different inertia → different
# k_joystick / stable kp, and pitch may need ki for gravity droop). Calibrate each with
# `calib_gimbal --axis {yaw,pitch}`. k_joystick = steady-velocity slope; kp = position-loop bandwidth
# (keep below ~1/(2·(DELTA_INPUT+TAU)) ≈ 3 rad/s or it rings; calib_gimbal prints a starting kp).
AIM_GAINS = {
  "yaw":   dict(k_joystick=9.05, kp=3.0, ki=0.0, kd=0.0),
  "pitch": dict(k_joystick=9.05, kp=3.0, ki=0.0, kd=0.0),   # TODO: calib_gimbal --axis pitch
}
AIM_I_CLAMP = 2.0       # rad/s, anti-windup ceiling on the integral's velocity contribution
AIM_FF_DT = 0.010       # s, forward finite-difference step for the velocity feedforward
AIM_D_TAU = 0.020       # s, low-pass time constant on the derivative term

# camera frame (RDF: x-right, y-down, z-forward) → gimbal-inertial frame
# (x-forward, y-up, z-right; right-handed). A camera mount is rigid, so this MUST
# be a proper rotation (det +1). Row 2 was [-1,0,0] (det -1, a reflection) — a
# latent bug that left-handed the gimbal frame and mirrored the lateral/yaw
# channel; the mirror was being undone downstream by decisiond's atan2 sign.
R_MOUNT = np.array([[0,  0,  1],
                    [0, -1,  0],
                    [1,  0,  0]], dtype=np.float64)
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
