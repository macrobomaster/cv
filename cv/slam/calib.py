"""Calibration constants for the SLAM stack.

Camera intrinsics come from the shared canonical-pinhole convention in
`autoaim.common` (camerad pre-warps the real sensor into it). IMU noise
parameters and the camera<-body extrinsic are bench measurements with TODO
markers where a proper calibration tool would supersede them.
"""
import numpy as np

from ..autoaim.common import CANONICAL_CAMERA_MATRIX, IMG_H as _IMG_H, IMG_W as _IMG_W

# Camera intrinsics + resolution come straight from autoaim.common — frontd/
# slamd consume camerad's `camera_feed`, which is pre-warped to exactly this
# canonical pinhole. Deriving (not hardcoding) keeps SLAM in lockstep when the
# canonical resolution/focal changes.
IMG_W, IMG_H = int(_IMG_W), int(_IMG_H)
K = CANONICAL_CAMERA_MATRIX.astype(np.float32)
FX, FY = float(K[0, 0]), float(K[1, 1])
CX, CY = float(K[0, 2]), float(K[1, 2])
K_INV = np.linalg.inv(K).astype(np.float32)

# --- Camera <- IMU extrinsic (pitch-dependent) -----------------------------
# Body frame = the gimbal IMU frame. The gimbal IMU sits on the FULL gimbal
# (yaw+pitch, with the autoaim camera); the SLAM camera is on the YAW-ONLY
# stage. So the IMU frame is the camera/yaw frame with the gimbal PITCH added
# on top — to go from the IMU frame to the SLAM-camera frame you undo pitch,
# then apply a fixed mount offset.
#
# R_MOUNT / T_MOUNT: fixed SLAM-camera <- yaw-stage mount (measure with
# calib_handeye). p_cam = R_MOUNT @ p_yawstage + T_MOUNT.
# TODO: fill in from calibration; identity placeholder for now.
R_MOUNT = np.eye(3, dtype=np.float32)
T_MOUNT = np.zeros(3, dtype=np.float32)
# Pitch rotation axis in the IMU frame and sign (TODO: verify against hardware;
# pitch about the camera x-axis is the usual convention).
PITCH_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
PITCH_SIGN = 1.0

def cam_from_imu(pitch_rad: float) -> tuple[np.ndarray, np.ndarray]:
  """Return (R_ic, t_ic): IMU-frame <- camera-frame rotation/translation at the
  given gimbal pitch, i.e. a camera-frame point maps to the IMU frame via
  p_imu = R_ic @ p_cam + t_ic. (This is the "body <- camera" extrinsic the
  MSCKF clones/updates use; the SLAM camera is the gimbal-yaw frame, the IMU
  adds pitch on top.)"""
  a = PITCH_SIGN * pitch_rad
  c, s = np.cos(a), np.sin(a)
  ax = PITCH_AXIS
  K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], np.float32)
  R_pitch = (np.eye(3, dtype=np.float32) + s*K + (1-c)*(K@K)).astype(np.float32)  # IMU <- yaw-stage
  R_ic = (R_pitch @ R_MOUNT.T).astype(np.float32)                          # IMU <- camera
  t_ic = (-R_ic @ T_MOUNT).astype(np.float32)
  return R_ic, t_ic

# IMU noise (continuous-time) — measurements `n` model: u_meas = u_true + b + n
# White-noise stddev per axis. Replace with values from the IMU datasheet
# (e.g. MPU9250: gyro ~0.01 rad/s/sqrt(Hz), accel ~0.0008 m/s^2/sqrt(Hz)).
GYRO_NOISE   = 1.0e-2            # rad / s / sqrt(Hz)
ACCEL_NOISE  = 1.0e-3            # m / s^2 / sqrt(Hz)
# Bias random walk stddev.
GYRO_BIAS_RW  = 1.0e-4           # rad / s^2 / sqrt(Hz)
ACCEL_BIAS_RW = 1.0e-4           # m / s^3 / sqrt(Hz)

# World gravity (we use world frame with +z up).
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)

# Pixel measurement noise stddev (used for MSCKF update).
PIXEL_NOISE = 1.0                # px

# MSCKF clone window size — fixed for static-shape JIT'd inner kernels.
N_CLONES = 15

# --- AprilTags (absolute pose correction) ----------------------------------
# Field has AprilTags at known locations; they replace loop closure. Detection
# uses cv2.aruco DICT_APRILTAG_36h11 (built into opencv, no extra dependency).
APRILTAG_DICT = "DICT_APRILTAG_36h11"
TAG_SIZE = 0.15                   # m, side length of the printed tag (black border)

# Field map: tag id -> world pose (R_world_tag (3x3), t_world_tag (3,)).
# The tag frame is the standard cv2 convention: origin at tag center, +x right,
# +y down, +z out of the tag toward the viewer. World frame is +z up.
# TODO: fill in with the real surveyed field tag layout.
TAG_FIELD_MAP: dict[int, tuple[np.ndarray, np.ndarray]] = {
  0: (np.eye(3, dtype=np.float32), np.array([0.0, 0.0, 0.5], dtype=np.float32)),
}

# Measurement noise stddev for an AprilTag absolute pose fix.
TAG_POS_NOISE = 0.05             # m
TAG_ROT_NOISE = 0.05            # rad
# Reject tag detections whose PnP reprojection is implausible / too far.
TAG_MAX_RANGE = 8.0              # m
