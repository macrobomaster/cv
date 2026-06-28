"""Shared constants + calibration for the field-localization stack.

Camera intrinsics come from the shared canonical-pinhole convention in
`autoaim.common` (camerad pre-warps the real sensor into it). IMU noise
parameters and the camera<-body extrinsic are bench measurements with TODO
markers where a proper calibration tool would supersede them.
"""
import math

import numpy as np

from ..autoaim.common import CANONICAL_CAMERA_MATRIX, IMG_H as _IMG_H, IMG_W as _IMG_W

# Camera intrinsics + resolution come straight from autoaim.common — slamd
# consumes camerad's `camera_feed`, which is pre-warped to exactly this
# canonical pinhole. Deriving (not hardcoding) keeps SLAM in lockstep when the
# canonical resolution/focal changes.
IMG_W, IMG_H = int(_IMG_W), int(_IMG_H)
K = CANONICAL_CAMERA_MATRIX.astype(np.float32)
FX, FY = float(K[0, 0]), float(K[1, 1])
CX, CY = float(K[0, 2]), float(K[1, 2])

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

# Body(camera) orientation at gimbal (yaw=0, pitch=0), as world<-body. The
# camera is RDF (x-right, y-down, z-forward); the world is z-up. A level,
# forward-looking camera maps: z(fwd)->world+x, y(down)->world-z, x(right)->-y.
# WITHOUT this, world<-body at level is identity → the camera "faces straight
# up" in the world view (RDF z = world z). TODO: verify columns vs hardware
# (which way is robot-forward, and the yaw/pitch signs in _gimbal_R_wb).
CAM_BASE_R = np.array([[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]], dtype=np.float32)

def cam_from_imu(pitch_rad: float) -> tuple[np.ndarray, np.ndarray]:
  """Return (R_ic, t_ic): IMU-frame <- camera-frame rotation/translation at the
  given gimbal pitch, i.e. a camera-frame point maps to the IMU frame via
  p_imu = R_ic @ p_cam + t_ic. (This is the "body <- camera" extrinsic for tag
  PnP; the SLAM camera is the gimbal-yaw frame, the IMU adds pitch on top.)"""
  a = PITCH_SIGN * pitch_rad
  c, s = np.cos(a), np.sin(a)
  ax = PITCH_AXIS
  K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], np.float32)
  R_pitch = (np.eye(3, dtype=np.float32) + s*K + (1-c)*(K@K)).astype(np.float32)  # IMU <- yaw-stage
  R_ic = (R_pitch @ R_MOUNT.T).astype(np.float32)                          # IMU <- camera
  t_ic = (-R_ic @ T_MOUNT).astype(np.float32)
  return R_ic, t_ic

# Accelerometer noise (continuous-time). Orientation comes from the gimbal, so
# the filter only models the accelerometer (no gyro). Replace with datasheet
# values (e.g. MPU9250 accel ~0.0008 m/s^2/sqrt(Hz)).
ACCEL_NOISE   = 1.0e-3           # m / s^2 / sqrt(Hz)  — white noise
ACCEL_BIAS_RW = 1.0e-4           # m / s^3 / sqrt(Hz)  — bias random walk
# Explicit velocity random-walk process noise. The accel-only process noise above
# is tiny, so the wheel/planar velocity updates shrink P[v] toward zero and the
# filter becomes overconfident in its own (coasting) velocity — the updates lose
# Kalman gain and stop pinning (velocity + the vertical channel drift even though
# the updates fire). This keeps P[v] from collapsing so the wheel measurements
# stay authoritative. Tune: too low → drifts when stopped; too high → noisy v.
VEL_PROCESS_NOISE = 0.3          # m / s / sqrt(s)

# World gravity (we use world frame with +z up).
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)
# Does RAW_ACCEL report raw specific force (≈9.81 magnitude at rest, includes
# gravity) or already-gravity-compensated linear acceleration (≈0 at rest)?
#   True  → predict uses a_w = R_wb·accel + GRAVITY   (raw specific force)
#   False → predict uses a_w = R_wb·accel             (gravity already removed)
# Wrong choice makes the robot free-fall ("sink into the floor") or rocket up.
# Check the slamd flow log's accel magnitude at rest: ~9.81 → True, ~0 → False.
ACCEL_INCLUDES_GRAVITY = False

# --- Wheel-odometry velocity fusion ----------------------------------------
# Chassis velocity (2-D horizontal, m/s, referenced to the gimbal heading) is
# fused as a velocity measurement on v_w. The accelerometer can't observe
# constant velocity (it reads ~0 at steady speed), so without this the estimate
# coasts/drifts whenever there's no parallax or tag. Because the board gives it
# in the gimbal frame, it's a full vector (direction + magnitude); the v=0 case
# (stationary) is just the same update, no special ZUPT branch needed.
WHEEL_VEL_NOISE = 0.1            # m/s — measurement stddev incl. slip + lever-arm slack; tune
# The robot drives on the ground, so its world-vertical velocity is ~0. The wheel
# update also pins v_w.z to 0 (planar-motion constraint) — wheels only observe the
# horizontal plane, so without this the vertical channel drifts (the "drifts
# upward" bug) and b_a.z stays unobservable. Loose enough to allow ramps.
VERT_VEL_NOISE = 0.05            # m/s — vertical (planar) velocity constraint stddev
# Reject wheel readings above this as a sensor fault / garbage. Moderate slip is
# absorbed by WHEEL_VEL_NOISE instead. Velocity is NOT Mahalanobis-gated: the
# accelerometer can't observe DC velocity, so P[v] understates the true
# uncertainty and a chi-square gate would reject the good wheel readings that
# make velocity observable (the overconfidence trap).
WHEEL_SPEED_MAX = 5.0            # m/s — above the robot's top speed
# Let the wheel-velocity update also correct gimbal-yaw drift (δψ) via its
# heading coupling. Powerful ONCE the wheel vx/vy axes are verified, but during
# bring-up an unverified wheel-axis sign feeds wrong yaw corrections and spirals
# the heading (a straight drive curves into a scribble). Keep OFF until the
# wheel axes + gimbal-yaw sign are confirmed.
VEL_OBSERVES_YAW = False

# --- AprilTags (absolute pose correction) ----------------------------------
# Field has AprilTags at known locations; they replace loop closure. Detection
# uses cv2.aruco DICT_APRILTAG_36h11 (built into opencv, no extra dependency).
APRILTAG_DICT = "DICT_APRILTAG_36h11"
TAG_SIZE = 0.15                   # m, side length of the printed tag (black border)

# Fuse AprilTags into the filter (absolute position + yaw). Keep this OFF until
# TAG_FIELD_MAP holds the real surveyed layout AND the gimbal/camera extrinsics
# are calibrated — otherwise a tag "fix" snaps to a made-up spot through a wrong
# R_wb and corrupts the state (esp. vertical). Detection still runs for the viz
# overlay regardless; this only gates the state update.
FUSE_APRILTAGS = False
def wall_tag(x:float, y:float, z:float, facing_deg:float) -> tuple[np.ndarray, np.ndarray]:
  """(R_world_tag, t_world_tag) for a tag mounted vertically + upright on a wall,
  its FACE pointing along facing_deg in the field horizontal plane (0°=+x, 90°=+y,
  CCW). cv2's tag frame is +x right, +y down, +z out of the face — so here tag +z
  = facing direction and tag +y = world-down. Floor/tilted/rotated tags need a
  hand-built R instead.

  Survey each tag off the field drawing: (x, y) = its centre in your chosen field
  frame, z = mounting height above the floor, facing = the way the face points
  INTO the play area (the side a robot sees it from)."""
  a = math.radians(facing_deg)
  z_axis = np.array([math.cos(a), math.sin(a), 0.0])   # tag +z (out of the face)
  y_axis = np.array([0.0, 0.0, -1.0])                  # tag +y (down)
  x_axis = np.cross(y_axis, z_axis)                    # tag +x = +y × +z (right-handed)
  R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
  return R, np.array([x, y, z], dtype=np.float32)

# Field map: tag id -> (R_world_tag, t_world_tag) in YOUR field frame (z up, RH).
# Build each with wall_tag(...). TODO: survey the 9 tags from the field drawings.
# Empty until then — no tag fuses (and FUSE_APRILTAGS is gated off above anyway).
TAG_FIELD_MAP: dict[int, tuple[np.ndarray, np.ndarray]] = {
  # 0: wall_tag(0.0, 4.0, 0.30, facing_deg=-90),   # e.g. on the +y wall, facing -y
  0: wall_tag(2.250, 3.679, 0.288, facing_deg=180),
  1: wall_tag(2.420, 2.666, 0.288, facing_deg=0),
  2: wall_tag(4.999, 1.200, 0.288, facing_deg=180),
  3: wall_tag(7.001, 1.200, 0.288, facing_deg=0),
  4: wall_tag(9.580, 2.666, 0.288, facing_deg=180),
  5: wall_tag(9.750, 3.679, 0.288, facing_deg=0),
  6: wall_tag(11.000, 7.999, 0.288, facing_deg=-90),
  7: wall_tag(6.000, 1.701, 0.288, facing_deg=90),
  8: wall_tag(6.000, 6.9839, 0.288, facing_deg=-90),
  9: wall_tag(1.000, 7.999, 0.288, facing_deg=-90),
}

# Measurement noise stddev for an AprilTag absolute pose fix.
TAG_POS_NOISE = 0.05             # m
TAG_ROT_NOISE = 0.05            # rad
# Reject tag detections whose PnP reprojection is implausible / too far.
TAG_MAX_RANGE = 8.0              # m
# Gimbal yaw is gyro-integrated and drifts (no absolute yaw reference); the
# filter carries a yaw-bias state δψ that random-walks at this rate and is
# corrected by the tag's absolute yaw (a Kalman update with TAG_YAW_NOISE).
YAW_DRIFT_RW = 5.0e-3            # rad / s / sqrt(Hz) — gimbal yaw drift rate
TAG_YAW_NOISE = 0.03            # rad — AprilTag yaw measurement stddev
